#!/usr/bin/env python3
"""Compatibility entry point and command line for agxforge.kernels.attention: the decode attention (RoPE, KV append, split and merge in one dispatch).

The kernel moved to the package whole. This file forwards every attribute to it, so `import g17attn` keeps
working, and keeps the command line: the compile check: build the split and merge forms and print their sizes.

    python3 tools/g17attn.py check
"""
import argparse

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.kernels import attention as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

# THE MODULE ALIASES this module had before the move, for callers that reach through them (g17qmm.Q, g17prefillattn.O)
if _os.path.join(_REPO_ROOT, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
import g17decodeops as O  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--q0", type=int, default=128)
    ap.add_argument("--cap", type=int, default=272)
    args = ap.parse_args(argv)
    lay = attn_layout(cap=args.cap)
    print("split: %d bytes, merge: %d bytes" % (len(build_attn_split(lay).code), len(build_attn_merge(lay).code)))


if __name__ == "__main__":
    main()
