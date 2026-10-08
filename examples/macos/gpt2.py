#!/usr/bin/env python3
"""Validate FP32 GPT-2 on M1 macOS with AGXForge G13 or Mesa AGX code generation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agxforge.g13.gpt2 import GPT2Plan, GPT2
from agxforge.g13.gpt2_reference import Reference
from examples.asahi.qwen import main
from examples.macos.backend import select_backend, select_compiler, source_identity, add_backend_options

if __name__ == "__main__":
    argv = sys.argv[1:]
    backend, Executor, name, _ = select_backend(argv)
    compiler_factory, _ = select_compiler(argv)
    if backend == "iogpu":
        NativeExecutor = Executor
        def Executor(**kwargs):
            return NativeExecutor(arena_bytes=640*1024*1024, **kwargs)
    main(executor_factory=Executor, identity_factory=source_identity, executor_name=name, argv=argv,
         plan_factory=GPT2Plan, model_factory=GPT2, reference_factory=Reference,
         default_checkpoint=ROOT/"models/macos/gpt2", description=__doc__, compiler_factory=compiler_factory,
         argument_extensions=add_backend_options)
