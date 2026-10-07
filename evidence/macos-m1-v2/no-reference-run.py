#!/usr/bin/env python3
"""Block CPU-reference access during native inference, then compare recorded GPU tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from examples.asahi import qwen
from agxforge.runtime.iogpu import Executor as NativeExecutor, EXECUTOR_NAME
from agxforge.runtime.macos import source_identity


def Executor(**kwargs):
    return NativeExecutor(arena_bytes=1152*1024*1024,**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"models/asahi/qwen2.5-0.5b")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference-bundle",type=Path,required=True)
    options = parser.parse_args()
    args = argparse.Namespace(output=options.output,checkpoint=options.checkpoint,capacity=32,
                              prompt="",token_ids="9707",generate=1,verify=False,va_slot=11)
    forbidden = AssertionError("CPU tensor reference reached during GPU-only inference")
    with patch.object(qwen,"Reference",side_effect=forbidden), \
         patch.object(qwen.Checkpoint,"floats",side_effect=forbidden):
        report = qwen.run(args,executor_factory=Executor,identity_factory=source_identity,
                          executor_name=EXECUTOR_NAME)
    baseline = json.loads((options.reference_bundle/"summary.json").read_text())
    if baseline["status"] != "PASS" or not baseline["verified"]:
        raise ValueError("baseline must be a verified model bundle")
    files = sorted((options.reference_bundle/"token-000").glob("*.f32"))
    if len(files) != 459: raise ValueError("baseline requires all 459 tensors")
    for file in files:
        if file.read_bytes() != (options.output/"token-000"/file.name).read_bytes():
            raise ValueError(f"GPU tensor differs: {file.name}")
    if report["generated_tokens"][0] != baseline["checks"][0]["next_token"]:
        raise ValueError("GPU argmax differs from verified baseline")
    result = dict(status="PASS against recorded verified GPU tensors",run_status=report["status"],
                  dispatches=report["dispatches"],reference_entry_points="Reference and Checkpoint.floats raise if called",
                  tensors_bit_identical=len(files),va_slot=11,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (options.output/"no-reference-check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
