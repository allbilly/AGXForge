#!/usr/bin/env python3
"""Check base M1 resources without dispatch; --backend metal|iogpu."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agxforge.g13.smoke import program
from examples.macos.backend import select_backend

if __name__ == "__main__":
    backend, Executor, _, argv = select_backend()
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for cycle in range(3):
        with Executor(va_slot=cycle) as gpu:
            buf = gpu.buffer(128, data=bytes(range(128)))
            assert buf.read() == bytes(range(128)) and buf.canaries_ok()
            gpu.prepare(program("store", 32))
            if cycle == 0: print(json.dumps(gpu.platform(), indent=2))
    print(f"PASS: 3 {backend} resource/profile cycles; no GPU commands submitted")
