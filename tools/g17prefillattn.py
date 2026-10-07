#!/usr/bin/env python3
"""Compatibility entry point for agxforge.kernels.prefill_attention: the prefill's KV append and scalar attention.

This file keeps main and forwards every other attribute to the package module.

    python3 tools/g17prefillattn.py check --m 16 --p0 0 [--cap 272]
"""
import argparse

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.kernels import prefill_attention as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

# THE MODULE ALIASES this module had before the move, for callers that reach through them (g17qmm.Q, g17prefillattn.O)
if _os.path.join(_REPO_ROOT, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
import g17attn as A  # noqa: E402
import g17decodeops as O  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--p0", type=int, default=0)
    ap.add_argument("--cap", type=int, default=272)
    args = ap.parse_args(argv)
    lay = prefill_layout(args.cap, args.m)
    print("append %d bytes, attention %d bytes" % (len(build_prefill_append(lay).code), len(build_prefill_attn(lay).code)))


if __name__ == "__main__":
    main()
