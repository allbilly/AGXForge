#!/usr/bin/env python3
"""Compatibility entry point and command line for agxforge.kernels.qmv: the quantized matrix-vector product (qmv) for one-token decode.

The kernel moved to the package whole. This file forwards every attribute to it, so `import g17qmv` keeps
working, and keeps the command line: the GPU check: build one qmv, dispatch it through the common worker and compare it with the reference.

    python3 tools/g17qmv.py check [--bits 4] [--N 2048] [--K 2048] [--rows 4]   build, dispatch, compare
"""
import argparse
import json
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

from agxforge.kernels import qmv as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

ROOT = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "check"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "g17-qmv-v1")
    ap.add_argument("--fault", default=None)
    ap.add_argument("--hoist", action="store_true")
    ap.add_argument("--premask", action="store_true")
    ap.add_argument("--fma", action="store_true")
    ap.add_argument("--wpt", type=int, default=0, help="words per trip: selects build_qmv2 (1, 2 or 4)")
    ap.add_argument("--interleave", action="store_true")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--coalesced", action="store_true")
    ap.add_argument("--shand", action="store_true")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--a16", action="store_true")
    args = ap.parse_args(argv)
    lay, x, packed, s16, b16, q = case(args.N, args.K, args.bits, args.rows, hoist=args.hoist, premask=args.premask,
                                       fma=args.fma)
    if args.wpt:
        lay = dict(lay, wpt=args.wpt, interleave=args.interleave, lean=args.lean, coalesced=args.coalesced,
                   shand=args.shand, chunk=args.chunk, a16=args.a16)
    prog = build_qmv2(lay) if args.wpt else build_qmv(lay, fault=args.fault)
    import hashlib
    name = "qmv%s_b%d_n%d_k%d_r%d%s%s%s%s" % ("2w%d%s%s%s%s" % (args.wpt, "i" if args.interleave else "", "L" if args.lean else "",
                                                           "C" if args.coalesced else "", "S" if args.shand else "") + ("c%d" % args.chunk if args.chunk != 8 else "")
                                       + ("A" if args.a16 else "")
                                       if args.wpt else "",
                                       args.bits, args.N, args.K, args.rows,
                                       "_h" if args.hoist else "",
                                       "_pm" if args.premask else "", "_fma" if args.fma else "",
                                       "_" + args.fault if args.fault else "")
    print(name, len(prog.code), "bytes", hashlib.sha256(prog.code).hexdigest()[:16], flush=True)
    if args.cmd == "build":
        return 0
    y = qmv2_reference(lay, x, q, s16, b16) if args.wpt else qmv_reference(lay, x, q, s16, b16)
    a, b, c, want = qmv_io(lay, x, packed, s16, b16, y)
    bundle = args.out / name
    if not bundle.exists():
        args.out.mkdir(parents=True, exist_ok=True)
        O.author(bundle, lay, prog, a, b, c, extra={"arm": name})
    worker = args.out / "common-worker"
    if not worker.exists():
        import g17tensorcommonruntime as TCR
        TCR.build_worker(worker)
    outs = O.dispatch(bundle, worker, queries=3, inputs=(a, b, c))
    res = [O.compare_words(o, want) for o in outs]
    got = np.frombuffer(outs[0], "<f4", lay["Nout"], lay["OUT"])
    exact = np.asarray(q, np.float64) * np.repeat(_from_bf16(s16), lay["group"], 1) + np.repeat(_from_bf16(b16), lay["group"], 1)
    ref64 = exact @ np.asarray(x, np.float64)
    print(json.dumps(dict(words_differing=res, max_rel_vs_f64=float(np.max(np.abs(got - ref64)) / np.max(np.abs(ref64))))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
