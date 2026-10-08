#!/usr/bin/env python3
"""Check compiled G13 kernels on base M1; --backend metal|iogpu."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.asahi.verify_kernels import run
from examples.macos.backend import select_backend, select_compiler, source_identity, add_backend_options

if __name__ == "__main__":
    argv = sys.argv[1:]
    _, Executor, name, _ = select_backend(argv)
    compiler_factory, _ = select_compiler(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_backend_options(parser)
    args = parser.parse_args(argv)
    report = run(args.output, executor_factory=Executor, identity_factory=source_identity,
                 compiler_factory=compiler_factory)
    print(f"PASS: {len(report['checks'])} {report['compiler']} kernel checks; {name}")
