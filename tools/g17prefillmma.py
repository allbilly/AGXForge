#!/usr/bin/env python3
"""Compatibility entry point for agxforge.kernels.prefill_mma: the prefill attention on the tensor units.

This file keeps probe_layout, build_probe, mma_reference and forwards every other attribute to the package module.
"""
import numpy as np

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.kernels import prefill_mma as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

_sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
import g17attn as A  # noqa: E402
import g17decodeops as O  # noqa: E402


def probe_layout(nb, p0):
    """Buffer 3 (fp32 words unless said): S at 0 (32 x 16), eight O tiles at 2048 + 2048 sl, M at 18432, L at 18560,
    out fp16 [32][128] at 18688, K cache blocks [nb][16][128] fp16 at 26880 (+ 4096 j). Buffer 1: Q fp16 [32][128].
    Buffer 2: V block-major [nb][8][16][16] fp16."""
    return dict(nb=nb, p0=p0, S=0, OT=2048, M=18432, L=18560, OUT=18688, KC=26880,
                c_bytes=26880 + nb * 4096, a_bytes=32 * D * 2, b_bytes=nb * 8 * 512)


def build_probe(lay):
    """The straight-line probe: init, then per block QK, row stage, eight PV bodies; then the normalisation."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    K2 = O.emit_exp2_constants(b)
    _init_state(b, ir, c, lane, lay)
    for j in range(lay["nb"]):
        b.tensor_matmul(a, c, c, M=32, N=16, K=D, transB=True, offsetA=0, offsetB=lay["KC"] + j * 4096, offsetC=lay["S"])
        _row_stage(b, ir, c, lane, lay, j, K2)
        for sl in range(8):
            b.tensor_matmul(c, bb, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                            offsetA=lay["S"], offsetB=j * 4096 + sl * 512, offsetC=lay["OT"] + sl * 2048)
    _finish(b, ir, c, lane, lay)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ------------------------------------------------------------------------------------------------------ reference
def mma_reference(Q16, Kb, Vb, p0):
    """Q16 [32][128], Kb [nb][16][128], Vb [nb][8][16][16] (fp16 values). Returns (out fp16 [32][128], the final O, l)."""
    import g17decodestep as DS
    import g17tensorcommonruntime as T
    nb = Kb.shape[0]
    Q = np.asarray(Q16, F32)
    Ot = np.zeros((8, 32, 16), F32)
    m = np.full(32, A.NEG_MAX, F32)
    l = np.zeros(32, F32)
    q0 = p0 + (np.arange(32) & 15)
    for j in range(nb):
        S = T._gemm_mma(Q, np.asarray(Kb[j], F32).T, None, 32, 16, D).astype(F32)
        keys = 16 * j + np.arange(16)
        valid = keys[None, :] <= q0[:, None]
        sv = np.where(valid, S, A.NEG_MAX).astype(F32)
        mb = sv[:, 0].copy()
        for k in range(1, 16):
            mb = np.maximum(mb, sv[:, k])
        mn = np.maximum(m, mb)
        nmn = (mn * F32(-1.0)).astype(F32)
        al = DS.exp2_soft((m + nmn).astype(F32)).astype(F32)
        P = DS.exp2_soft((S + nmn[:, None]).astype(F32)).astype(F32)
        P = np.where(valid, P, F32(0.0)).astype(F32)
        ssum = P[:, 0].copy()
        for k in range(1, 16):
            ssum = (ssum + P[:, k]).astype(F32)
        l = ((l * al).astype(F32) + ssum).astype(F32)
        m = mn
        Ot = (Ot * al[None, :, None]).astype(F32)
        for sl in range(8):
            Ot[sl] = T._gemm_mma(P, np.asarray(Vb[j, sl], F32), Ot[sl], 32, 16, 16, truncate_a=True)
    rl = DS.recip(l).astype(F32)
    O_ = np.concatenate([Ot[sl] for sl in range(8)], axis=1)             # [32][128]
    return (O_ * rl[:, None]).astype(F32).astype(np.float16), O_, l
