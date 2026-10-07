"""The quantized prefill GEMM (MM 25.144.1): y[M][N] = x[M][K] . W^T, W the model's affine q4/q8 weights (group 64,
bf16 scales and biases, the layout g17qmv reads), on the tensor units.

Form 1, two dispatches:
  dequant   one thread per packed word, n fastest: w[k][n] = fp16_rne(fp32(q) * s + b) written as W16[K][N] (the
            GEMM's row-major B; the K loop refuses transposed operands). fp32(q) * s is exact (a bf16 scale times an
            integer below 256 fits fp32), the + b rounds once, then fp16 rounds: the reference does the same three
            steps in numpy.
  gemm      gemm_generic (tlower) on A = x fp16 [M][K] rows and B = W16, C = [M][N] fp32 rows (ldc = N), the
            K loop in 1-4 simdgroups, columns over grid_n, K over split_k (w2). The reference is the pinned MMA
            model (_gemm_mma / _gemm_mma_fast) on the dequant's own output.

    python3 tools/g17qmm.py verify --bits 4 --m 512 [--roles qkv,wo,w1,w2]

The GEMM itself is gemm_generic, the compiler's tensor GEMM class, built by tools/g17tensorcommonruntime.py
(author_generic) from role_spec's spec: this module is the dequantization kernel and the spec, not the GEMM program.

Moved from tools/g17qmm.py, which keeps the GPU verification and timing (verify, bench) and the command line.
"""

import numpy as np

from agxforge.kernels import bundle, emit, qmv


SENT = b"\x7f"
# role: (N, K) of the milestone model (d 2048, ffn 8192, wqkv N 4096)
ROLES = {"qkv": (4096, 2048), "wo": (2048, 2048), "w1": (8192, 2048), "w3": (8192, 2048), "w2": (2048, 8192)}


def dequant_layout(N, K, bits):
    """Offsets: the B buffer holds packed words at W (bytes), bf16 scales at S and biases at B (bytes); the C buffer
    holds W16[K][N] fp16 at OUT (bytes)."""
    pw = 32 // bits
    wbytes = N * (K // pw) * 4
    gbytes = N * (K // 64) * 2
    return dict(N=N, K=K, bits=bits, pw=pw, W=0, S=wbytes, B=wbytes + gbytes, b_bytes=wbytes + 2 * gbytes,
                OUT=0, c_bytes=K * N * 2, threads=N * (K // pw))


def build_dequant(lay):
    """The dequant kernel (cc IR): thread gid = tg * 32 + lane; n = gid mod N, j = gid / N (the packed word of row n);
    its pw values land at W16[(j pw + i) N + n], i = 0..pw-1."""
    from agxforge.g17 import cc, ir
    N, K, bits, pw = lay["N"], lay["K"], lay["bits"], lay["pw"]
    if N & (N - 1) or K % 64:
        raise ValueError("dequant: N a power of two, K a multiple of 64")
    fn, b, a, bb, c = emit._function()
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")
    gid = b.add(b.shl(tg, emit._c(b, 5, "k5"), name="tg32"), lane, name="gid")
    n = getattr(b, "and")(gid, emit._c(b, N - 1, "nmask"), name="n")
    j = b.shr(gid, emit._c(b, N.bit_length() - 1, "nsh"), name="j")
    wi = b.add(b.add(b.mul(n, emit._c(b, K // pw, "wrow"), name="nw"), j, name="nwj"), emit._c(b, lay["W"] // 4, "wb"), name="wi")
    word = b.load(bb, wi, type=ir.I32, name="word")
    g = b.shr(j, emit._c(b, (64 // pw).bit_length() - 1, "gsh"), name="g")           # group = (j pw) / 64
    si = b.add(b.add(b.mul(n, emit._c(b, K // 64, "grow"), name="ng"), g, name="ngg"), emit._c(b, lay["S"] // 2, "sb"), name="si")
    bi = b.add(si, emit._c(b, (lay["B"] - lay["S"]) // 2, "bdiff"), name="bi")
    s32 = b.shl(b.load(bb, si, width="half", name="sh"), emit._c(b, 16, "s16"), name="s32")
    b32 = b.shl(b.load(bb, bi, width="half", name="bh"), emit._c(b, 16, "b16"), name="b32")
    base = b.add(b.add(b.mul(j, emit._c(b, pw * N, "jrow"), name="jN"), n, name="jNn"), emit._c(b, lay["OUT"] // 2, "ob"),
                 name="obase")
    mask = (1 << bits) - 1
    for i in range(pw):
        q = getattr(b, "and")(b.shr(word, ir.Imm(bits * i), name="ws%d" % i) if i else word, ir.Imm(mask), name="q%d" % i)
        v = b.fmul(b.u32_to_f32(q, name="qf%d" % i), s32, type=ir.F32, name="v%d" % i)
        w = b.fadd(v, b32, type=ir.I32, name="w%d" % i)   # float bits typed I32, as the norms carry them into the fp16 convert
        oi = b.add(base, emit._c(b, i * N, "io%d" % i), name="oi%d" % i) if i else base
        b.store_at(c, oi, b.f32_to_f16_rte(w, name="h%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def dequant_reference(q, s16, b16, bits):
    """W16[K][N]: fp16_rne(fp32(q) * s + b), each fp32 op rounded to nearest even (numpy float32)."""
    N, K = q.shape
    s = qmv._from_bf16(s16).astype(np.float32)
    bb = qmv._from_bf16(b16).astype(np.float32)
    srep = np.repeat(s, 64, axis=1)
    brep = np.repeat(bb, 64, axis=1)
    w = (q.astype(np.float32) * srep).astype(np.float32)
    w = (w + brep).astype(np.float32)
    return w.astype(np.float16).T.copy()


def weights(N, K, bits, seed):
    rng = np.random.default_rng(seed)
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    return qmv.quantize(Wf, bits=bits)          # packed [N][K/pw] u32, s16, b16 [N][K/64], q [N][K]


def dequant_io(lay, packed, s16, b16):
    b = bytearray(lay["b_bytes"])
    bundle._place(b, lay["W"], packed.astype("<u4"))
    bundle._place(b, lay["S"], s16.astype("<u2"))
    bundle._place(b, lay["B"], b16.astype("<u2"))
    c = SENT * lay["c_bytes"]
    return bytes(1024), bytes(b), c


# THE DELIVERED FORM per (role, M) (MM 25.144.1): K slices per trip (kloop_unroll) and loop-carried row bases, chosen
# by the warm-clock protocol (g17projwarm, 9 rounds, bit-exact every unit). M 512 bodies split register groups, so a
# longer prompt runs faster as M 256 chunks than as one M 512 dispatch.
FORM = {"qkv": {128: dict(u=4, gn=128), 256: dict(u=4, sg=4, gn=128, tg=2), 512: dict()},
        "wo": {128: dict(u=4, gn=64), 256: dict(u=4, sg=2, gn=64, tg=4), 512: dict()},
        "w1": {128: dict(u=4), 256: dict(u=4, sg=8), 512: dict()},
        "w3": {128: dict(u=4), 256: dict(u=4, sg=8), 512: dict()},
        "w2": {128: dict(u=4, sk=1, gn=64), 256: dict(u=4, sg=2, sk=1, gn=64, tg=4), 512: dict()}}
# MLX's per-simdgroup tile, 2 x 2 (MM 25.144.1): simdgroups x grid_n chosen so each simdgroup owns 2 row tiles and 2
# column tiles, 4 slices per trip; 1.1-1.3x faster warm than the 4 x 1 / 2 x 1 tiles at every role but w1 (already 2 x 2).
# sk overrides gemm_shape's split_k: w2 (K 8192) runs its whole K in one threadgroup (127 trips at 4 slices).
# tg is the M-block grid (MM 25.157): tg row groups beside the grid_n column groups (rows the low bits of the id, so
# the row groups sharing a column block of W16 run adjacently), each simdgroup still a 2 x 2 tile (32 rows; tlower's
# simdgroup split is along rows, so tg x sg x 32 = M). qkv, wo and w2 reach 256 threadgroups at M 256: -2.5 percent of
# the 1,024-token prefill's GPU time, -3.9 ms wall. Timed in the graph, not alone: alone the weights stay cached.


def gemm_spec(M, N, K, sg, grid_n, split_k, unroll=1, bases=False, tg=1):
    spec = dict(M=M, N=N, K=K, threadgroups=tg, simdgroups=sg, grid_n=grid_n, split_k=split_k, kloop=True)
    if unroll > 1:
        spec["kloop_unroll"] = unroll
    if bases:
        spec["kloop_bases"] = True
    return spec


def role_spec(role, M):
    """The delivered spec for (role, M): gemm_shape's launch plus FORM's loop form (M outside FORM: the plain loop)."""
    N, K = ROLES[role]
    sg, gn, sk = gemm_shape(M, N, K)
    f = FORM.get(role, {}).get(M, {})
    if "sk" in f:
        sk = f["sk"]
        gn = min(N // 16, 256 // sk)
    gn = f.get("gn", gn)
    return gemm_spec(M, N, K, f.get("sg", sg), gn, sk, f.get("u", 1), f.get("bases", False), f.get("tg", 1))


def gemm_shape(M, N, K):
    """(sg, grid_n, split_k) for a projection: 16 rows x 16 columns per MMA tile; 4 simdgroups own M/4 rows each
    (M/64 tile rows per simdgroup), one tile column per threadgroup where N allows <= 256 threadgroups (w1 two),
    and K over two threadgroup groups when K/16 exceeds the loop's 256 slices."""
    sg = 4 if M >= 64 else 1
    split_k = 2 if K // 16 > 256 else 1
    grid_n = min(N // 16, 256 // split_k)
    return sg, grid_n, split_k
