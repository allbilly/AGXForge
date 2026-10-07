#!/usr/bin/env python3
"""Run and validate BF16 Qwen2.5-0.5B through G13 Metal or direct IOGPU."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.asahi.qwen import main
from examples.macos.backend import select_backend, source_identity

if __name__ == "__main__":
    backend, Executor, name, argv = select_backend()
    if backend == "iogpu":
        NativeExecutor = Executor
        def Executor(**kwargs):
            return NativeExecutor(arena_bytes=1152*1024*1024, **kwargs)
    main(executor_factory=Executor, identity_factory=source_identity, executor_name=name, argv=argv)
