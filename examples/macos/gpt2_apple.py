#!/usr/bin/env python3
"""Validate GPT-2 with Apple's MSL compiler, using the same scalar graph and FP64 oracle."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agxforge.g13.gpt2 import GPT2Plan, GPT2
from agxforge.g13.gpt2_reference import Reference
from examples.asahi.qwen import main
from examples.macos.apple_metal import Executor, EXECUTOR_NAME
from examples.macos.backend import source_identity

if __name__ == "__main__":
    plans = []
    def planner(checkpoint, capacity):
        plan = GPT2Plan(checkpoint, capacity); plans.append(plan); return plan
    def executor(**kwargs):
        return Executor(plans[-1], **kwargs)
    main(executor_factory=executor, identity_factory=source_identity, executor_name=EXECUTOR_NAME,
         plan_factory=planner, model_factory=GPT2, reference_factory=Reference,
         default_checkpoint=ROOT/"models/macos/gpt2", description=__doc__)
