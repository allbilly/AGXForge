"""IR emission helpers the decode kernels share.

    _c, _cf, _and                  integer and fp32 constants, bitwise and
    emit_constants, emit_rn        the correctly rounded rsqrt and recip: the hardware seed, then the exact midpoint
                                   test of agxforge.kernels.numerics, emitted in the same 15-bit limbs
    emit_exp2_constants, emit_exp2_soft    numerics.exp2_soft, instruction for instruction
    _function                      the three-buffer function every decode kernel is (the common worker's contract)
    _carrier, _carrier_bytes       the 16 x 16 x 16 MMA a kernel runs first so the common worker admits it
    _transport                     the gemm_generic buffer rule the worker sizes the three buffers by

Moved from tools/g17decodeops.py, which keeps the MM 25.136 decode-step kernels and their dispatch harness.
"""
from __future__ import annotations

import numpy as np

from agxforge.kernels.numerics import EXP2_COEF, EXP2_HI, EXP2_LO, EXP2_MAGIC

F32 = np.float32
M15 = 0x7FFF
KV_LENGTH_BYTE = 6144                        # runtime.KV_LENGTH_BYTE (P9's uniform)
CARRIER = 16                                 # the carrier MMA's M, N and K per threadgroup


def _bits(x):
    return int(np.asarray(x, F32).reshape(()).view(np.uint32))


def _c(b, v, name):
    from agxforge.g17 import ir
    return b.const(int(v) & 0xFFFFFFFF, type=ir.I32, name=name)


def _cf(b, v, name):
    from agxforge.g17 import ir
    return b.const(_bits(v), type=ir.F32, name=name)


def _and(b, x, m, name):
    return getattr(b, "and")(x, m, name=name)


def _emit_fields(b, bits, K, tag):
    """(M, E): the 24-bit significand with its hidden bit, and the biased exponent."""
    m = getattr(b, "or")(_and(b, bits, K["mant"], tag + "_m23"), K["hidden"], name=tag + "_m")
    e = _and(b, b.shr(bits, K["s23"], name=tag + "_esh"), K["ff"], name=tag + "_e")
    return m, e


def _emit_mid_top(b, kind, x_m, x_e, cand, K, tag):
    """1 where f(x) < mid(cand, cand+), as the 0/1 value of an unsigned compare (see the models)."""
    a_m, a_e = _emit_fields(b, cand, K, tag + "a")
    q = b.add(b.shl(a_m, K["s1"], name=tag + "_2m"), K["one"], name=tag + "_q")
    q0 = _and(b, q, K["m15"], tag + "_q0")
    q1 = b.shr(q, K["s15"], name=tag + "_q1")
    m0 = _and(b, x_m, K["m15"], tag + "_x0")
    m1 = b.shr(x_m, K["s15"], name=tag + "_x1")
    if kind == "rsqrt":
        t0 = b.mul(q0, q0, name=tag + "_t0")
        s0 = _and(b, t0, K["m15"], tag + "_s0")
        c = b.shr(t0, K["s15"], name=tag + "_c0")
        t1 = b.add(b.shl(b.mul(q0, q1, name=tag + "_q01"), K["s1"], name=tag + "_2q01"), c, name=tag + "_t1")
        s1 = _and(b, t1, K["m15"], tag + "_s1")
        c = b.shr(t1, K["s15"], name=tag + "_c1")
        t2 = b.add(b.mul(q1, q1, name=tag + "_q11"), c, name=tag + "_t2")
        s2 = _and(b, t2, K["m15"], tag + "_s2")
        s3 = b.shr(t2, K["s15"], name=tag + "_s3")
        k = b.shr(b.mul(s0, m0, name=tag + "_p00"), K["s15"], name=tag + "_k0")
        col = b.add(b.add(b.mul(s1, m0, name=tag + "_p10"), b.mul(s0, m1, name=tag + "_p01"), name=tag + "_c1s"), k, name=tag + "_col1")
        k = b.shr(col, K["s15"], name=tag + "_k1")
        col = b.add(b.add(b.mul(s2, m0, name=tag + "_p20"), b.mul(s1, m1, name=tag + "_p11"), name=tag + "_c2s"), k, name=tag + "_col2")
        k = b.shr(col, K["s15"], name=tag + "_k2")
        col = b.add(b.add(b.mul(s3, m0, name=tag + "_p30"), b.mul(s2, m1, name=tag + "_p21"), name=tag + "_c3s"), k, name=tag + "_col3")
        k = b.shr(col, K["s15"], name=tag + "_k3")
        top = b.add(b.mul(s3, m1, name=tag + "_p31"), k, name=tag + "_top")
        # sh = 392 - Ex - 2 Ea
        sh = b.sub(b.sub(K["c392"], x_e, name=tag + "_sh1"), b.shl(a_e, K["s1"], name=tag + "_2ea"), name=tag + "_sh")
    else:
        k = b.shr(b.mul(q0, m0, name=tag + "_p00"), K["s15"], name=tag + "_k0")
        col = b.add(b.add(b.mul(q1, m0, name=tag + "_p10"), b.mul(q0, m1, name=tag + "_p01"), name=tag + "_c1s"), k, name=tag + "_col1")
        k = b.shr(col, K["s15"], name=tag + "_k1")
        top = b.add(b.mul(q1, m1, name=tag + "_p11"), k, name=tag + "_top")
        # sh = 271 - Eg - Ea
        sh = b.sub(b.sub(K["c271"], x_e, name=tag + "_sh1"), a_e, name=tag + "_sh")
    bound = b.shl(K["one"], sh, name=tag + "_bound")
    below = b.icmp(top, bound, rel="ult", name=tag + "_below")        # 1 where P < 2^S
    return below


def emit_constants(b):
    from agxforge.g17 import ir
    return dict(mant=_c(b, 0x7FFFFF, "k_mant"), hidden=_c(b, 0x800000, "k_hidden"), s23=_c(b, 23, "k_s23"),
                ff=_c(b, 0xFF, "k_ff"), s1=_c(b, 1, "k_s1"), one=_c(b, 1, "k_one"), m15=_c(b, M15, "k_m15"),
                s15=_c(b, 15, "k_s15"), c392=_c(b, 392, "k_c392"), c271=_c(b, 271, "k_c271"),
                zero=_c(b, 0, "k_zero"))


def emit_rn(b, kind, x, K, tag):
    """The correctly rounded rsqrt(x) or recip(x): the hardware seed, then the exact midpoint test."""
    from agxforge.g17 import ir
    y0 = (b.rsqrt if kind == "rsqrt" else b.recip)(x, type=ir.I32, name=tag + "_seed")
    x_m, x_e = _emit_fields(b, x, K, tag + "x")
    ym1 = b.sub(y0, K["one"], name=tag + "_ym1")
    below_lo = _emit_mid_top(b, kind, x_m, x_e, ym1, K, tag + "L")      # 1: f >= mid(y0-1, y0)
    below_hi = _emit_mid_top(b, kind, x_m, x_e, y0, K, tag + "H")       # 1: f > mid(y0, y0+1)
    # result = y0 - 1 + below_lo + below_hi: below_hi implies below_lo (the midpoints are ordered)
    return b.add(b.add(ym1, below_lo, name=tag + "_lo"), below_hi, name=tag + "_rn")


def emit_exp2_soft(b, t, K2, tag):
    from agxforge.g17 import ir
    I = ir.I32                                  # float bits carried in integer-typed values
    t = b.fmin(b.fmax(t, K2["lo"], type=I, name=tag + "_cl"), K2["hi"], type=I, name=tag + "_ch")
    s = b.fadd(t, K2["magic"], type=I, name=tag + "_s")
    nf = b.fadd(s, K2["nmagic"], type=I, name=tag + "_nf")
    f = b.fadd(t, b.fneg(nf, type=I, name=tag + "_nnf"), type=I, name=tag + "_f")
    p = K2["c7"]
    for i in range(6, -1, -1):
        p = b.fadd(b.fmul(p, f, type=I, name="%s_m%d" % (tag, i)), K2["c%d" % i], type=I, name="%s_p%d" % (tag, i))
    n = b.sub(s, K2["mbits"], name=tag + "_n")
    return b.add(p, b.shl(n, K2["s23"], name=tag + "_n23"), name=tag + "_e")


def emit_exp2_constants(b):
    K2 = dict(lo=_cf(b, EXP2_LO, "e_lo"), hi=_cf(b, EXP2_HI, "e_hi"), magic=_cf(b, EXP2_MAGIC, "e_magic"),
              nmagic=_cf(b, -EXP2_MAGIC, "e_nmagic"), mbits=_c(b, 0x4B400000, "e_mbits"), s23=_c(b, 23, "e_s23"))
    for i, cf in enumerate(EXP2_COEF):
        K2["c%d" % i] = _cf(b, cf, "e_c%d" % i)
    return K2


def _function(name="tensor_gemm_generic_runtime_demo"):
    from agxforge.g17 import ir
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function(name, [a, bb, c])
    return fn, ir.Builder(fn, fn.block("entry")), a, bb, c


def _carrier(b, a, bb, c, lay):
    """One 16 x 16 x 16 MMA per threadgroup, at offset 0 of every buffer (the grid split applies no
    operand offsets): A = buffer-1 rows [0, 16 G) of 16 halves, B = buffer-2's first 16 x 16 halves, D
    into buffer-3 rows [0, 16 G) of 16 floats. This path reads A from buffer 1 whatever buffer the IR
    names (the probe's first carrier named buffer 2 and read buffer 1, MM 25.136.3). Runs first: no
    scalar is live across a tensor body."""
    b.tensor_matmul(a, bb, c, M=CARRIER * lay["groups"], N=CARRIER, K=CARRIER, threadgroups=lay["groups"])


def _transport(groups, a_bytes, b_bytes, c_bytes, N=256):
    """gemm_generic's buffer rule (the worker's): A = M K 2, B = K N 2, C = M N 4, M a multiple of 16
    groups and at most 1,024 rows per group."""
    unit = 16 * groups
    M = max(unit, -(-(-(-c_bytes // (4 * N))) // unit) * unit)
    if M // groups > 1024:
        raise ValueError("transport: %d rows per threadgroup exceed the worker's 1,024" % (M // groups))
    K = max(16, -(-(-(-a_bytes // (2 * M))) // 16) * 16)
    if K * N * 2 < b_bytes:
        K = max(K, -(-(-(-b_bytes // (2 * N))) // 16) * 16)
    # gemm_generic's class table refuses K above 256 without a K loop (the wide_n rule at N 256), so
    # grow M instead until A and B fit with K <= 256
    while K > 256:
        M += unit
        if M // groups > 1024:
            raise ValueError("transport: no M x N x K with K <= 256 holds these buffers")
        K = max(16, -(-(-(-a_bytes // (2 * M))) // 16) * 16, -(-(-(-b_bytes // (2 * N))) // 16) * 16)
    return dict(M=M, N=N, K=K, a_bytes=M * K * 2, b_bytes=K * N * 2, c_bytes=M * N * 4)


def _align(v, a=256):
    return -(-v // a) * a


def _carrier_bytes(groups):
    """Bytes the carrier occupies at the start of buffer 3 (16 G rows of 16 floats); its A tile at the
    start of buffer 1 (16 G rows of 16 halves) and its B tile in buffer 2 are smaller, so one offset
    clears all three."""
    return _align(4 * CARRIER * CARRIER * groups)
