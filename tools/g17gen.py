#!/usr/bin/env python3
"""Compatibility entry point and command line for agxforge.kernels.argmax: device-side greedy decoding (the two-pass argmax and the generation step).

The kernel moved to the package whole. This file forwards every attribute to it, so `import g17gen` keeps
working, and keeps the command line: the compile check: build both passes and print their sizes.

    python3 tools/g17gen.py check
"""
import argparse

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.kernels import argmax as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

# THE MODULE ALIASES this module had before the move, for callers that reach through them (g17qmm.Q, g17prefillattn.O)
if _os.path.join(_REPO_ROOT, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
import g17decodeops as O  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check",))
    a = ap.parse_args(argv)
    lay = argmax_layout()
    print("pass1 %d bytes, pass2 %d bytes, G %d" % (len(build_pass1(lay).code), len(build_pass2(lay).code), lay["G"]))


if __name__ == "__main__":
    main()
