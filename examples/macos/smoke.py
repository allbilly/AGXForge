#!/usr/bin/env python3
"""Run the handwritten G13 suite on base M1; --backend metal|iogpu."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.asahi.smoke import main
from examples.macos.backend import select_backend, source_identity

if __name__ == "__main__":
    _, Executor, _, argv = select_backend()
    main(executor_factory=Executor, identity_factory=source_identity, argv=argv)
