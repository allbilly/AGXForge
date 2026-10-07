#!/usr/bin/env python3
"""Compatibility entry point for agxforge.kernels.qmm: the quantized prefill GEMM's dequantization and spec.

This file keeps verify, bench, main and forwards every other attribute to the package module.

    python3 tools/g17qmm.py verify --bits 4 --m 512 [--roles qkv,wo,w1,w2]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.kernels import qmm as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

# THE MODULE ALIASES this module had before the move, for callers that reach through them (g17qmm.Q, g17prefillattn.O)
if _os.path.join(_REPO_ROOT, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
import g17qmv as Q  # noqa: E402
import g17decodeops as O  # noqa: E402


def verify(bits, M, roles, work, seed=11):
    """Dequant then GEMM on hardware for each role; returns {role: dict}. Every output is sentinel-filled first."""
    import g17deliver as D
    import g17tensorcommonruntime as R
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = {}
    for role in roles:
        N, K = ROLES[role]
        lay = dequant_layout(N, K, bits)
        packed, s16, b16, q = weights(N, K, bits, seed + bits * 10 + len(role))
        prog = build_dequant(lay)
        a, bbuf, c = dequant_io(lay, packed, s16, b16)
        d = D.author(work / ("dequant_q%d_%s" % (bits, role)), prog, a, bbuf, c, lay)
        (work / "dq_run").mkdir(exist_ok=True)
        got = D.dispatch([dict(tag="dq", dir=d, threads=lay["threads"], group=32, base=1)], work / "dq_run")["dq"]
        w16 = np.frombuffer(got, "<u2", K * N, lay["OUT"]).reshape(K, N)
        want = dequant_reference(q, s16, b16, bits).view("<u2")
        dq_diff = int((w16 != want).sum())
        _sp = role_spec(role, M)
        sg, gn, sk = _sp["simdgroups"], _sp["grid_n"], _sp["split_k"]
        g = work / ("gemm_q%d_%s_m%d" % (bits, role, M))
        if g.exists():
            shutil.rmtree(g)
        R.author_generic(g, role_spec(role, M))
        rng = np.random.default_rng(seed + M)
        x = np.asarray(rng.standard_normal((M, K)), np.float16)
        (g / "a.f16").write_bytes(x.astype("<f2").tobytes())
        (g / "b.f16").write_bytes(np.asarray(w16).tobytes())          # the dequant's OWN output feeds the GEMM
        rep = R.run(g, queries=1, composition="generic")
        qrec = rep["queries"][0]
        out[role] = dict(N=N, K=K, M=M, dequant_mismatch=dq_diff, dequant_code_bytes=len(prog.code),
                         gemm_status=rep["status"], gemm_mismatch=qrec.get("mismatched_elements"),
                         gemm_shape=dict(sg=sg, grid_n=gn, split_k=sk),
                         gemm_code_bytes=len((g / "program.bin").read_bytes()))
        print("q%d %-3s M %d: dequant %d mismatched (%d B), gemm %s %s mismatched (sg %d gn %d sk %d, %d B)" % (
            bits, role, M, dq_diff, len(prog.code), rep["status"], qrec.get("mismatched_elements"), sg, gn, sk,
            out[role]["gemm_code_bytes"]), flush=True)
    return out


def bench(bits, ms, roles, work, rounds=15, warm=5, seed=11):
    """GPU time of ours per (role, M): the dequant (runner rounds, first `warm` dropped) and the GEMM (common-worker
    queries after `warm` warm-ups), medians. Every GEMM bundle is verified (status passed) before its time counts."""
    import statistics
    import subprocess
    import g17deliver as D
    import g17tensorcommonruntime as R
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for role in roles:
        N, K = ROLES[role]
        lay = dequant_layout(N, K, bits)
        packed, s16, b16, q = weights(N, K, bits, seed + bits * 10 + len(role))
        prog = build_dequant(lay)
        a, bbuf, c = dequant_io(lay, packed, s16, b16)
        d = D.author(work / ("dequant_q%d_%s" % (bits, role)), prog, a, bbuf, c, lay)
        run = work / "dq_bench"
        run.mkdir(exist_ok=True)
        (run / "out").mkdir(exist_ok=True)
        plan = run / "plan.json"
        plan.write_text(json.dumps(dict(configs=[dict(tag="dq", bundle=str(d), threads=lay["threads"], group=32, base=1,
                                                      rounds=rounds + warm)])))
        r = subprocess.run([str(D.runner()), str(plan), str(run / "out")], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("dequant run failed: " + r.stderr[-500:])
        dq_us = statistics.median([float(line.split()[3]) for line in r.stdout.splitlines() if line.startswith("time ")][warm:])
        w16 = np.frombuffer((run / "out" / "dq.out").read_bytes(), "<u2", K * N, lay["OUT"]).reshape(K, N)
        if int((w16 != dequant_reference(q, s16, b16, bits).view("<u2")).sum()):
            raise SystemExit("dequant not bit-exact: nothing timed")
        for M in ms:
            _sp = role_spec(role, M)
            sg, gn, sk = _sp["simdgroups"], _sp["grid_n"], _sp["split_k"]
            g = work / ("gemm_q%d_%s_m%d" % (bits, role, M))
            if g.exists():
                shutil.rmtree(g)
            R.author_generic(g, role_spec(role, M))
            rng = np.random.default_rng(seed + M)
            (g / "a.f16").write_bytes(np.asarray(rng.standard_normal((M, K)), np.float16).astype("<f2").tobytes())
            (g / "b.f16").write_bytes(np.asarray(w16).tobytes())
            rep = R.run(g, queries=rounds + warm, composition="generic")
            if rep["status"] != "passed":
                raise SystemExit("gemm %s M %d not bit-exact: nothing timed" % (role, M))
            gemm_us = statistics.median([qq["gpu_seconds"] * 1e6 for qq in rep["queries"]][warm:])
            tf = 2.0 * M * N * K / ((dq_us + gemm_us) * 1e-6) / 1e12
            rows.append(dict(bits=bits, role=role, M=M, N=N, K=K, dequant_us=dq_us, gemm_us=gemm_us,
                             total_us=dq_us + gemm_us, tflops_total=tf, gemm_shape=dict(sg=sg, grid_n=gn, split_k=sk)))
            print("ours q%d %-3s M %4d: dequant %8.1f us  gemm %8.1f us  total %8.1f us  %5.2f TFLOP/s (gemm alone %5.2f)" % (
                bits, role, M, dq_us, gemm_us, dq_us + gemm_us, tf, 2.0 * M * N * K / (gemm_us * 1e-6) / 1e12), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "bench"))
    ap.add_argument("--ms", default="128,256,512")
    ap.add_argument("--out")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--roles", default="qkv,wo,w1,w2")
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    work = a.work or os.path.join("/tmp", "g17qmm-%d" % os.getpid())
    if a.cmd == "bench":
        rows = bench(a.bits, [int(v) for v in a.ms.split(",")], a.roles.split(","), work)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(dict(clock="GPU (command buffer GPUEnd - GPUStart), medians", rows=rows), fh, indent=1)
        return
    rep = verify(a.bits, a.m, a.roles.split(","), work)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
