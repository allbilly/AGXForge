#!/usr/bin/env python3
"""THE PREFILL SECTION OF THE GRAPH (MM 25.142.10): the whole prompt as M-row passes, run once before decode.

tools/g17q4graph.py calls `section(...)` when the model config has {"prefill": {"M": 128|512|1024}}. It returns the arenas,
initial files and the `prefill` steps decodegen runs (host writes, then one serial command buffer per step), so that
after the last step the KV cache holds positions 0..L-1, the token log holds the first generated token at L, region R
holds that token's embedding, and the q0 word is L: decode continues exactly where token-by-token feeding would.

Per chunk of M rows (positions p0..p0+M-1), per layer, kernels from the deliver index (Piece B's milestones, 25.144):
    norm_rows(attn_norm, fp16 out) -> qmm_dequant(qkv) + qmm(qkv) -> prefill_append -> prefill_attn(out16)
    -> qmm_dequant(wo) + qmm(wo) -> residual_rows(add16) -> norm_rows(ffn_norm, fp16 out)
    -> qmm(w1), qmm(w3) -> swiglu_rows -> qmm(w2, split-K) -> fold_rows = fp16((p0 + p1) + h)
Then the tail, once, on the last row: a host copy of x[L-1] into R's input row, q0 = L - 1, and decode's own
final_norm, head, argmax and gen_step dispatches (which write log[L] and advance q0 to L).

Every kernel kind this needs must be in the index, or `section` refuses naming the missing (kind, role, variant). Region
placement uses tools/g17modelgraph.Solver in a separate arena "PF"; attention binds decode's own region3 (grown to the
prefill kernels' prefill_region3_bytes) and GEN region.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))           # tools/ first, then the checkout root


# THE SECTION'S ASSEMBLY MOVED to agxforge.inference.prefill (the graph assembler calls it); this module keeps the CPU
# reference of the whole prefill below, and forwards the moved names
from agxforge.inference import prefill as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, (
    "PF",
    "PE",
    "Missing",
    "need",
    "REGO_OPT",
    "_rego_attn",
    "kernels",
    "tail_step",
    "region3_bytes",
    "_al",
    "_block",
    "section",
)))


# ---------------------------------------------------------------------------------------------------------------------
# THE CPU REFERENCE of the whole prefill: every op in its kernel's own arithmetic (M1 dequant and MMA order, M2 append and
# attention, M3 row norms and elementwise), composed per layer. It is what the GPU prefill must reproduce bit for bit
# (the KV cache rows and the last row's x), independent of placement.

def dequant_reference(npz, bits, N, K):
    """W16 [N][K] = fp16_rne(fp32(fp32(q) * s) + b), bf16 s/b widened (M1's contract, g17qmm.dequant_reference)."""
    import numpy as np
    from g17q4graph import _unpack
    q = _unpack(npz["W"], bits, N, K).astype(np.float32)
    s = (np.asarray(npz["S"], np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(N, K // 64)
    b = (np.asarray(npz["B"], np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(N, K // 64)
    s = np.repeat(s, 64, axis=1); b = np.repeat(b, 64, axis=1)
    return ((q * s).astype(np.float32) + b).astype(np.float32).astype(np.float16)


def gemm_reference(a16, w16, split_k=1):
    """y fp32 [M][N] = A fp16 [M][K] x W16^T, in the tensor unit's issue order (g17tensorcommonruntime._gemm_mma_fast);
    split_k halves K and sums the partials p0 + p1 in fp32 (M1's w2)."""
    import numpy as np
    import g17tensorcommonruntime as TCR
    A = np.asarray(a16, np.float16).astype(np.float32)
    B = np.asarray(w16, np.float16).astype(np.float32).T          # [K][N]
    M, K = A.shape
    N = B.shape[1]
    if split_k == 1:
        return np.asarray(TCR._gemm_mma_fast(A, B, None, M, N, K), np.float32)
    h = K // split_k
    parts = [np.asarray(TCR._gemm_mma_fast(A[:, i * h:(i + 1) * h], B[i * h:(i + 1) * h], None, M, N, h), np.float32)
             for i in range(split_k)]
    return parts


def reference(bits, prompt_ids, cap, weights_dir, layers=None, ffn16=False):
    """The prefill of `prompt_ids` through `layers` (default all): per layer's K/V rows, and the last row's x fp16 after
    the last layer. Returns dict(x=[L][2048] fp16 after the last layer run, K=[l] [8][cap][128], V=...). ffn16: the w1 and
    w3 outputs rounded to fp16 (RNE) before the SwiGLU, as the half-epilogue GEMMs store them (MM 25.183)."""
    import numpy as np
    import g17decodeops as O
    import g17prefillattn as P
    import g17rows as RW
    import g17realmodel as M
    W = Path(weights_dir)
    L = len(prompt_ids)
    spec = M.layer_spec(0)
    emb = np.load(W / "embed.npy", mmap_mode="r")
    x16 = np.asarray(emb[list(prompt_ids)], np.float16)
    lay = P.prefill_layout(cap, cap, out16=True)
    pos = np.arange(cap)[:, None] * (M.ROPE_THETA ** (-np.arange(0, 128, 2, dtype=np.float64) / 128))[None, :]
    cos, sin = np.cos(pos).astype(np.float32), np.sin(pos).astype(np.float32)
    out = dict(K=[], V=[])
    for l in (range(M.LAYERS) if layers is None else layers):
        g1 = np.load(W / ("L%d_g1.npy" % l)).astype(np.float32)
        g2 = np.load(W / ("L%d_g2.npy" % l)).astype(np.float32)
        n1 = np.asarray(O.rmsnorm_wide_rows_reference(x16.astype(np.float32), g1, spec), np.float16)
        qkv = gemm_reference(n1, dequant_reference(np.load(W / ("L%d_qkv.npz" % l)), bits, 4096, 2048))
        Kc = np.zeros((8, cap, 128), np.float16); Vc = np.zeros((8, cap, 128), np.float16)
        attn16, _q16, K, V = P.prefill_reference(lay, qkv, cos, sin, Kc, Vc, 0, L)
        out["K"].append(np.asarray(K, np.float16)); out["V"].append(np.asarray(V, np.float16))
        y = gemm_reference(np.asarray(attn16, np.float16).reshape(L, 2048),
                           dequant_reference(np.load(W / ("L%d_wo.npz" % l)), bits, 2048, 2048))
        h = RW.rows_reference(dict(kind="residual"), (y, x16))["h"]
        n2 = np.asarray(O.rmsnorm_wide_rows_reference(h, g2, spec), np.float16)
        ffn = np.load(W / ("L%d_ffn.npz" % l))
        w13 = dequant_reference(ffn, bits, 16384, 2048)
        g = gemm_reference(n2, w13[:8192]); u = gemm_reference(n2, w13[8192:])
        if ffn16:
            g, u = np.asarray(g, np.float16), np.asarray(u, np.float16)
        act = RW.rows_reference(dict(kind="swiglu", in16=ffn16), (g, u))["act"]
        p0, p1 = gemm_reference(act, dequant_reference(np.load(W / ("L%d_w2.npz" % l)), bits, 2048, 8192), split_k=2)
        x16 = RW.rows_reference(dict(kind="fold"), (np.stack([p0, p1]).reshape(2, -1), h.reshape(-1)))["x"].reshape(L, 2048)
    out["x"] = x16
    return out
