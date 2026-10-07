#!/usr/bin/env python3
"""Run and independently validate FP32 GPT-2 through authored G13 on M1 macOS."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agxforge.g13.gpt2 import GPT2Plan, GPT2
from agxforge.g13.gpt2_reference import Reference
from examples.asahi.qwen import main
from examples.macos.backend import select_backend, source_identity

if __name__ == "__main__":
    backend, Executor, name, argv = select_backend()
    if backend == "iogpu":
        NativeExecutor = Executor
        def Executor(**kwargs):
            return NativeExecutor(arena_bytes=640*1024*1024, **kwargs)
    main(executor_factory=Executor, identity_factory=source_identity, executor_name=name, argv=argv,
         plan_factory=GPT2Plan, model_factory=GPT2, reference_factory=Reference,
         default_checkpoint=ROOT/"models/macos/gpt2", description=__doc__)
