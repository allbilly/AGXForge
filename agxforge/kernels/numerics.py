"""The fp32 arithmetic the decode kernels perform, as numpy models (MM 25.132, 25.136).

Every reference in agxforge.kernels states its values with these functions, and every builder emits the same
operations in the same order, so each kernel is checked bit for bit against its reference:

    ftz, fadd, fmul      the ALU: round to nearest even, subnormal inputs and results flushed to a signed zero
                         (recon section 138 part 5)
    recip, rsqrt         rounded once; the GPU corrects its hardware seed with the exact midpoint test
                         (mid_below_*, round_once), which agxforge.kernels.emit emits instruction for instruction
    rsqrt_seed           op3850 itself, bit for bit (MM 25.141.16)
    exp2_soft            a fixed fp32 sequence the GPU reproduces by construction (op1272 is not correctly rounded)
    narrow, to_bits16    the 16-bit storage rounding (op1016, RNE)
    gemm, truncate19     the section-136 MMA, vectorised, and its fp32-A operand quantisation
    rope_rotate, q_scale, silu    the decode step's rounding points that the kernels fix

LayerSpec is the decoder layer's shape the references take; MILESTONE is InternLM2.5-1.8B's.

Moved from tools/g17decodestep.py, which keeps the MM 25.132 decode step's stage references and dispatch plan.
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

# THE REPOSITORY ROOT, for the one measured table this module reads (isa/g17-rsqrt-seed.npz)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GELU_NEG_INV_LN2 = np.float32(-1.4426950408889634)     # -1/ln 2, the GELU register step's (tlower.GELU_CONSTANTS)

F32, F64 = np.float32, np.float64
LOG2E = 1.4426950408889634
KEY_BLOCK = 16                             # keys per score tile (runtime.ATTENTION_BLOCK)


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """One decoder layer's decode step. `storage` is the 16-bit type of every stored tensor (weights,
    activations, cache); accumulation is fp32 throughout. `k_chunk` 0 is the single-chain class (one
    MMA chain over the whole K); k_chunk > 0 is split-K on every GEMM with slices of that length, under
    gemm_reference's contract (single-chain partials, ascending fp32 left fold, the residual C last). The
    host-orchestrated pipeline needs it: a straight-line gemm_generic body admits K <= 256 (the K loop is
    a loop, and wide_n excludes it)."""
    d_model: int = 2048
    n_heads: int = 16
    head_dim: int = 128
    ffn_dim: int = 8192
    kv_len: int = 256
    storage: str = "half"                  # "half" or "bfloat"
    norm_eps: float = 1.0e-5
    rope_base: float = 10000.0
    k_chunk: int = 0
    # k_route True: each projection at projection_route's split_k, the GPU pipeline's route (G 8 for the
    # 2048-wide qkv, o_proj and ffn_down at the milestone, G 1 for gate/up), instead of k_chunk on every
    # GEMM. It is the reference tools/g17decodestep_gpu.py checks the whole step against when k_chunk is 0.
    k_route: bool = False
    # GQA (MM 25.138.2): n_kv_heads KV heads, each shared by n_heads / n_kv_heads query heads; 0 is n_heads (MHA).
    # The qkv projection is then N = d_model + 2 n_kv_heads head_dim, and the RoPE append writes the cache of
    # every QUERY head from its KV head, so the attention sees the MHA layout unchanged
    n_kv_heads: int = 0

    @property
    def kv_heads(self):
        return self.n_kv_heads or self.n_heads

    @property
    def qkv_width(self):
        return self.d_model + 2 * self.kv_heads * self.head_dim

    def __post_init__(self):
        if self.storage not in ("half", "bfloat"):
            raise ValueError("storage is half or bfloat")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        for name in ("d_model", "head_dim", "ffn_dim"):
            if getattr(self, name) % 16:
                raise ValueError("%s must be a multiple of 16 (the MMA issue)" % name)
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head_dim")
        if self.kv_len < 0:
            raise ValueError("kv_len must be non-negative")
        if self.k_chunk and (self.k_chunk < 0 or self.k_chunk % 16 or self.d_model % self.k_chunk
                             or self.ffn_dim % self.k_chunk):
            raise ValueError("k_chunk must be a multiple of 16 dividing d_model and ffn_dim")

    @property
    def n_keys(self):
        return self.kv_len + 1

    @property
    def key_blocks(self):
        return -(-self.n_keys // KEY_BLOCK)

    def as_dict(self):
        return dataclasses.asdict(self)


MILESTONE = LayerSpec()


_TINY32 = F32(2.0 ** -126)


def ftz(v):
    v = np.array(v, dtype=F32, copy=True)
    tiny = (v != 0) & (np.abs(v) < _TINY32)
    v[tiny] = np.copysign(F32(0.0), v[tiny])
    return v


def fadd(x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(x) + ftz(y))


def fmul(x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(x) * ftz(y))


def exp2(x):
    with np.errstate(over="ignore"):
        return ftz(np.exp2(ftz(x).astype(F64)).astype(F32))


def _recip64(x):
    with np.errstate(divide="ignore", over="ignore"):
        return ftz((1.0 / ftz(x).astype(F64)).astype(F32))


def _rsqrt64(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return ftz((1.0 / np.sqrt(ftz(x).astype(F64))).astype(F32))


# THE EXACT MIDPOINT TEST (MM 25.136). For a positive normal candidate a, the midpoint between a and its
# successor is (2 Ma + 1) 2^(Ea - 151) (Ma the 24-bit significand, Ea the biased exponent, also across a
# binade edge). 1/sqrt(x) is below it exactly when Mx (2 Ma + 1)^2 > 2^(452 - Ex - 2 Ea), and 1/g exactly
# when Mg (2 Ma + 1) > 2^(301 - Eg - Ea). The products are formed in 15-bit limbs, every intermediate
# below 2^31, keeping only the bits from 2^60 (2^30) up: the comparison with a power of two at least
# that large needs no more. Equality is impossible (2 Ma + 1 is odd and at least 2^24), so the test never
# ties. tools/g17decodeops.py emits the same limbs in the GPU program, instruction for instruction.
_M15 = 0x7FFF


def _fields(bits):
    bits = np.asarray(bits, np.int64)
    return (bits & 0x7FFFFF) | 0x800000, (bits >> 23) & 0xFF


def mid_below_rsqrt(bx, ba):
    """True where 1/sqrt(x) < mid(a, a+) (x and a as float32 bit patterns)."""
    Mx, Ex = _fields(bx)
    Ma, Ea = _fields(ba)
    q = 2 * Ma + 1
    q0, q1 = q & _M15, q >> 15
    t0 = q0 * q0
    s0, c = t0 & _M15, t0 >> 15
    t1 = ((q0 * q1) << 1) + c
    s1, c = t1 & _M15, t1 >> 15
    t2 = q1 * q1 + c
    s2, s3 = t2 & _M15, t2 >> 15
    m0, m1 = Mx & _M15, Mx >> 15
    k = (s0 * m0) >> 15
    k = (s1 * m0 + s0 * m1 + k) >> 15
    k = (s2 * m0 + s1 * m1 + k) >> 15
    k = (s3 * m0 + s2 * m1 + k) >> 15
    top = s3 * m1 + k                                     # floor(Mx q^2 / 2^60)
    sh = 392 - Ex - 2 * Ea                                # S - 60
    if np.any((sh < 0) | (sh > 30)):
        raise ValueError("rsqrt midpoint test: the candidate is not within a binade of 1/sqrt(x)")
    return top >= (np.int64(1) << sh)


def mid_below_recip(bg, ba):
    """True where 1/g < mid(a, a+)."""
    Mg, Eg = _fields(bg)
    Ma, Ea = _fields(ba)
    q = 2 * Ma + 1
    q0, q1 = q & _M15, q >> 15
    m0, m1 = Mg & _M15, Mg >> 15
    k = (q0 * m0) >> 15
    k = (q1 * m0 + q0 * m1 + k) >> 15
    top = q1 * m1 + k                                     # floor(Mg q / 2^30)
    sh = 271 - Eg - Ea                                    # S - 30
    if np.any((sh < 0) | (sh > 30)):
        raise ValueError("recip midpoint test: the candidate is not within a binade of 1/g")
    return top >= (np.int64(1) << sh)


def round_once(below, x, seed):
    """The correctly rounded value from any seed within one ulp of it: y0 - 1 where f lies below
    mid(y0 - 1, y0), y0 + 1 where it lies above mid(y0, y0 + 1), else y0 (the GPU program's rule)."""
    bx = np.asarray(x, F32).view(np.uint32)
    by = np.asarray(seed, F32).view(np.uint32).astype(np.int64)
    down = below(bx, by - 1)
    up = ~below(bx, by)
    return np.where(down, by - 1, np.where(up, by + 1, by)).astype(np.uint32).view(F32)


def recip(x):
    """1/x rounded once (correctly rounded), for a positive normal x whose reciprocal is normal: the
    float64 value, within one ulp, put through the exact midpoint test, so no double rounding
    survives. On the GPU, op3658's seed through the same test (MM 25.136: op3658 is within one ulp
    and not correctly rounded)."""
    x = np.asarray(x, F32)
    return round_once(mid_below_recip, x, _recip64(x))


def rsqrt(x):
    """1/sqrt(x) rounded once (correctly rounded), for a positive normal x: as `recip`. On the GPU,
    op3850's seed through the same test (MM 25.136 measured op3850: within one ulp, not correctly
    rounded)."""
    x = np.asarray(x, F32)
    return round_once(mid_below_rsqrt, x, _rsqrt64(x))


_SEED = None


def rsqrt_seed(x):
    """op3850 itself (cc `b.rsqrt`, the hardware rsqrt seed) for a positive normal x, bit for bit (MM 25.141.16).
    It is faithful (within 0.82 ulp) and equals `rsqrt` on 95.7 percent of inputs. Measured as an exact function:
    2^24 inputs per band, and seed(x 4^k) == seed(x) 2^-k on every input at k = -8 and k = +5. So the seed is
    `rsqrt` except at the 725,821 (parity, mantissa) keys in isa/g17-rsqrt-seed.npz, where it takes the stored
    [1, 4) value scaled by the same power of two."""
    global _SEED
    if _SEED is None:
        import os
        z = np.load(os.path.join(_REPO_ROOT, "isa", "g17-rsqrt-seed.npz"))
        _SEED = (z["index"].astype(np.int64), z["seed"].astype(np.int64))
    x = np.asarray(x, F32)
    out = rsqrt(x).view(np.uint32).astype(np.int64)
    bits = x.view(np.uint32).astype(np.int64)
    e = (bits >> 23) & 0xFF
    if np.any((e == 0) | (e == 0xFF) | (bits >> 31 != 0)):
        raise ValueError("rsqrt_seed: positive normal inputs only")
    par = (e - 127) & 1
    key = (par << 23) | (bits & 0x7FFFFF)
    k = (e - 127 - par) >> 1                          # x = x0 4^k, x0 in [1, 4)
    pos = np.searchsorted(_SEED[0], key)
    pos = np.minimum(pos, len(_SEED[0]) - 1)
    hit = _SEED[0][pos] == key
    out = np.where(hit, _SEED[1][pos] - (k << 23), out)
    return out.astype(np.uint32).view(F32).reshape(x.shape)


# SILU'S EXP2 (MM 25.136). op1272 is within one ulp both ways and not correctly rounded, and no cheap
# exact test decides 2^t against a midpoint, so no reference that rounds exp2 once can be matched bit
# for bit (the GELU register step passes by an enclosure for this reason, 25.109). exp2_soft is a fixed
# sequence of RNE fp32 and integer steps the GPU reproduces by construction.
EXP2_MAGIC = F32(12582912.0)                 # 1.5 * 2^23: t + MAGIC holds rint(t) in its low bits
# [-125, 125]: p 2^n stays normal and finite, and SiLU's 1/(1 + 2^t) stays normal, so the exact
# midpoint test (which needs a normal result) applies wherever SiLU calls recip
EXP2_LO, EXP2_HI = F32(-125.0), F32(125.0)
EXP2_COEF = tuple(F32(math.log(2.0) ** k / math.factorial(k)) for k in range(8))


def exp2_soft(t):
    """t clamped to [-125, 125]; s = t + 1.5 * 2^23; n = s - 1.5 * 2^23 (rint(t), exact); f = t - n
    (exact, |f| <= 1/2); p = the degree-7 Taylor polynomial of 2^f in Horner form (fmul then fadd,
    RNE); then n added to p's exponent field as an integer. Within one ulp of 2^t."""
    t = np.asarray(t, F32)
    t = np.minimum(np.maximum(t, EXP2_LO), EXP2_HI)
    s = fadd(t, EXP2_MAGIC)
    nf = fadd(s, -EXP2_MAGIC)
    f = fadd(t, -nf)
    p = np.full(f.shape, EXP2_COEF[7], F32)
    for c in EXP2_COEF[6::-1]:
        p = fadd(fmul(p, f), c)
    n = s.view(np.uint32).astype(np.int64) - 0x4B400000
    return ((p.view(np.uint32).astype(np.int64) + (n << 23)) & 0xFFFFFFFF).astype(np.uint32).view(F32)


def narrow(x, storage="half"):
    """fp32 -> the 16-bit storage type, RNE (op1016 for half), returned as the fp32 value it holds."""
    x = np.asarray(x, dtype=F32)
    if storage == "half":
        with np.errstate(over="ignore"):
            return x.astype(np.float16).astype(F32)
    u = x.view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    out = rounded.astype(np.uint32).view(F32)
    return np.where(np.isnan(x), x, out).astype(F32)


def to_bits16(x, storage="half"):
    """The stored 16-bit patterns of values already representable in `storage`."""
    x = np.asarray(x, dtype=F32)
    if storage == "half":
        return x.astype(np.float16).view(np.uint16)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def truncate19(x):
    """The measured fp32-A operand quantisation (chapter 5): the low 13 bits cleared."""
    u = np.asarray(x, dtype=F32).view(np.uint32) & np.uint32(0xFFFFE000)
    return u.view(F32)


def gemm(a, b, c=None, *, truncate_a=False, truncate_b=False):
    """D = A B (+ C once, after the chain), A (M x K), B (K x N), fp32 out: _gemm_mma vectorised.

    Each 16-wide issue: products rounded to fp32 (exact for 16-bit operands), p_i = RNE32(p_2i +
    p_2i+1), q_j = RNE32(p_j + p_j+4), acc = C (the previous issue's D; the first issue is the no-C
    form, acc = q_0) then acc = RNE32(acc + q_j). A float64 sum of two fp32 values rounded to fp32 is
    the correctly rounded fp32 sum, which is what `_rne32` of a Python float computes."""
    a = np.asarray(a, dtype=F32)
    b = np.asarray(b, dtype=F32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm: A is M x K and B is K x N")
    M, K = a.shape
    N = b.shape[1]
    if K % 16:
        raise ValueError("gemm: K must be a multiple of 16 (the MMA issue)")
    if truncate_a:
        a = truncate19(a)
    if truncate_b:
        b = truncate19(b)
    a64, b64 = a.astype(F64), b.astype(F64)
    r = lambda v: v.astype(F32).astype(F64)
    acc = None
    with np.errstate(over="ignore", invalid="ignore"):
        for s in range(0, K, 16):
            prod = r(a64[:, s:s + 16, None] * b64[None, s:s + 16, :])        # M x 16 x N
            p = r(prod[:, 0::2] + prod[:, 1::2])                                # p_0..p_7
            q = r(p[:, 0:4] + p[:, 4:8])                                        # q_0..q_3
            if acc is None:
                acc, rest = q[:, 0], range(1, 4)
            else:
                rest = range(4)
            for j in rest:
                acc = r(acc + q[:, j])
        if c is not None:
            acc = r(acc + np.asarray(c, dtype=F32).astype(F64).reshape(M, N))
    return acc.astype(F32)


def rope_rotate(t, cos, sin):
    """Rotate-half RoPE (pairs i and i + head_dim/2), per output two fmuls and one fadd."""
    h = t.shape[-1] // 2
    x1, x2 = t[..., :h], t[..., h:]
    o1 = fadd(fmul(x1, cos), -fmul(x2, sin))
    o2 = fadd(fmul(x2, cos), fmul(x1, sin))
    return np.concatenate([o1, o2], axis=-1)


def q_scale(spec):
    """The softmax scale 1/sqrt(head_dim), in base 2 (the attention class exponentiates with exp2)."""
    return F32(LOG2E / math.sqrt(spec.head_dim))


def silu(x):
    """The GELU register step (tlower 'gelu', gelu_model) without its 1.702 multiply: t = x*(-1/ln 2),
    t = 2**t, t = t + 1, t = 1/t, y = x*t. CHANGED (MM 25.136): 2**t is exp2_soft, not the exact value
    rounded once, because op1272 is not correctly rounded and nothing cheap corrects it; 1/t stays
    rounded once (the GPU corrects op3658 exactly)."""
    t = fmul(x, GELU_NEG_INV_LN2)
    t = exp2_soft(t)
    t = fadd(t, F32(1.0))
    t = recip(t)
    return fmul(x, t)
