#!/usr/bin/env python3
"""Compatibility entry point for agxforge.kernels.rows: the prefill's elementwise row kernels.

Every attribute forwards to the package module.
"""

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.kernels import rows as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

globals().update(_compat.install(__name__, _impl, [n for n in vars(_impl) if not n.startswith("__")]))

# THE MODULE ALIASES this module had before the move, for callers that reach through them (g17qmm.Q, g17prefillattn.O)
if _os.path.join(_REPO_ROOT, "tools") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
import g17decodeops as O  # noqa: E402
