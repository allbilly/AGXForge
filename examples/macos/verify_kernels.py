#!/usr/bin/env python3
"""Check compiled G13 kernels on base M1; --backend metal|iogpu."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.asahi.verify_kernels import run
from examples.macos.backend import select_backend, source_identity

if __name__ == "__main__":
    _, Executor, name, argv = select_backend()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.output, executor_factory=Executor, identity_factory=source_identity)
    print(f"PASS: {len(report['checks'])} compiled G13 kernel checks; {name}")
