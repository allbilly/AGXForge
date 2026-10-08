#!/usr/bin/env python3
"""Validate BF16 Qwen2.5-0.5B on M1 with AGXForge G13 or Mesa AGX code generation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.asahi.qwen import main
from examples.macos.backend import select_backend, select_compiler, source_identity, add_backend_options

if __name__ == "__main__":
    argv = sys.argv[1:]
    backend, Executor, name, _ = select_backend(argv)
    compiler_factory, _ = select_compiler(argv)
    if backend == "iogpu":
        NativeExecutor = Executor
        def Executor(**kwargs):
            return NativeExecutor(arena_bytes=1152*1024*1024, **kwargs)
    main(executor_factory=Executor, identity_factory=source_identity, executor_name=name, argv=argv,
         compiler_factory=compiler_factory, argument_extensions=add_backend_options)
