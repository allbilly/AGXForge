#!/usr/bin/env python3
"""Native BF16 Qwen2.5-0.5B: tokenwise prefill, decode, GPU argmax, FP64 validation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tokenizers import Tokenizer
from agxforge.g13.qwen import Checkpoint, QwenPlan, Qwen
from agxforge.g13.qwen_reference import Reference
from agxforge.runtime.asahi import Executor, source_identity

# Fixed before dispatch; validation runs an independent FP64 graph on the SAME history.
BOUNDS = {
    "embedding": (0, 0), "norm": (.003, .003), "projection": (.005, .003),
    "rope": (.005, .003), "cache": (.005, .003), "scores": (.01, .005),
    "probs": (.0005, .005), "attention": (.005, .003), "residual": (.01, .003),
    "activation": (.01, .005), "logits": (.05, .003)
}


def category(name):
    if name in ("embedding", "logits"): return name
    for key in ("norm", "rope", "cache", "scores", "probs", "attention", "residual", "activation"):
        if key in name: return key
    return "projection"


def run(args, *, executor_factory=Executor, identity_factory=source_identity, executor_name="native Asahi DRM",
        plan_factory=QwenPlan, model_factory=Qwen, reference_factory=Reference, compiler_factory=None,
        execution_compiler=None):
    output = args.output; output.mkdir(parents=True, exist_ok=False)
    checkpoint = Checkpoint(args.checkpoint)
    compiler = compiler_factory(output/"compiler") if compiler_factory is not None else None
    plan = (plan_factory(checkpoint, args.capacity, compiler=compiler) if compiler is not None else
            plan_factory(checkpoint, args.capacity))
    tokenizer = Tokenizer.from_file(str(args.checkpoint/"tokenizer.json"))
    history = [int(t) for t in args.token_ids.split(",")] if args.token_ids else tokenizer.encode(args.prompt).ids
    if not history or any(t < 0 or t >= plan.vocab for t in history) or len(history)+args.generate-1 > plan.capacity:
        raise ValueError("invalid input history or generation exceeds KV capacity")
    (output/"checkpoint.json").write_text(json.dumps(checkpoint.receipt(), indent=2)+"\n")
    (output/"plan.json").write_text(json.dumps(plan.descriptor(), indent=2)+"\n")
    (output/"bounds.json").write_text(json.dumps(BOUNDS, indent=2)+"\n")
    identity = identity_factory()
    (output/"source-sha256.json").write_text(json.dumps(identity, indent=2)+"\n")
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(), status="RUNNING",
                  input_tokens=list(history), generated_tokens=[], checks=[], tensor_fallbacks=[],
                  executor=executor_name, prefill="tokenwise", verified=args.verify, bounds=BOUNDS)
    report["compiler"] = execution_compiler or getattr(plan, "compiler_name", "AGXForge G13")
    oracle = reference_factory(checkpoint, plan) if args.verify else None
    try:
        with executor_factory(timeout_ms=30000, va_slot=args.va_slot) as gpu:
            (output/"platform.json").write_text(json.dumps(gpu.platform(), indent=2)+"\n")
            model = model_factory(gpu, checkpoint, plan, output/"launches")
            while True:
                position = model.position
                token = history[position]
                print(f"position {position}, token {token}: reference {'on' if oracle else 'off'}", flush=True)
                expected = oracle.consume(token) if oracle else None
                token_dir = output/f"token-{position:03d}"; token_dir.mkdir()
                token_checks = []
                def observer(name, raw):
                    actual = np.frombuffer(raw, np.float32)
                    (token_dir/(name+".f32")).write_bytes(raw)
                    if expected is None: return
                    ref = expected[name].reshape(-1)
                    (token_dir/(name+".f64")).write_bytes(ref.tobytes())
                    finite = np.isfinite(ref)
                    atol, rtol = BOUNDS[category(name)]
                    error = np.abs(actual[finite].astype(np.float64)-ref[finite])
                    passed = np.all(np.isfinite(actual[finite])) and np.all(error <= atol+rtol*np.abs(ref[finite]))
                    passed = passed and np.array_equal(actual[~finite], ref[~finite])
                    if name == "embedding": passed = passed and actual.tobytes() == ref.astype(np.float32).tobytes()
                    row = dict(tensor=name, status="PASS" if passed else "WRONG_OUTPUT", atol=atol, rtol=rtol,
                               max_abs_error=float(np.max(error, initial=0)), elements=len(actual),
                               actual_sha256=hashlib.sha256(raw).hexdigest())
                    token_checks.append(row)
                    if not passed:
                        gpu.poisoned = True
                        raise RuntimeError(f"position {position}, {name}: outside declared bounds; sequence stopped")
                start = time.monotonic()
                try: next_token = model.consume(token, observer)
                finally: (token_dir/"checks.json").write_text(json.dumps(token_checks, indent=2)+"\n")
                top_match = oracle is None or next_token == int(np.argmax(expected["logits"]))
                if not top_match: gpu.poisoned = True; raise RuntimeError("GPU argmax disagrees with FP64 oracle")
                report["checks"].append(dict(position=position, token=token, next_token=next_token,
                    tensors=len(token_checks), status="PASS" if oracle else "COMPLETED_UNVERIFIED",
                    top_token_matches=top_match if oracle else None, host_execution_seconds=time.monotonic()-start))
                print(f"position {position}: {len(token_checks)} tensors checked, next token {next_token}", flush=True)
                (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
                if model.position < len(report["input_tokens"]): continue
                report["generated_tokens"].append(next_token)
                if len(report["generated_tokens"]) >= args.generate: break
                history.append(next_token)
            report.update(dispatches=model.dispatches, completed_positions=model.position,
                          text=tokenizer.decode(report["generated_tokens"]))
        if identity_factory() != identity: raise RuntimeError("sources changed during run")
        report.update(status="PASS" if oracle else "COMPLETED_UNVERIFIED", teardown="clean")
    except BaseException as error:
        report.update(status="FAIL", error=str(error)); raise
    finally:
        (output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main(*, executor_factory=Executor, identity_factory=source_identity, executor_name="native Asahi DRM", argv=None,
         plan_factory=QwenPlan, model_factory=Qwen, reference_factory=Reference,
         default_checkpoint=None, description=None, compiler_factory=None, execution_compiler=None,
         argument_extensions=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint or ROOT/"models/asahi/qwen2.5-0.5b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--token-ids", help="fixed token history instead of text")
    parser.add_argument("--generate", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--va-slot", type=int, choices=range(16), default=0)
    parser.add_argument("--verify", action="store_true", help="check intermediate tensors and logits against FP64")
    if argument_extensions is not None: argument_extensions(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.generate <= args.capacity: parser.error("generate must be 1..capacity")
    result = run(args, executor_factory=executor_factory, identity_factory=identity_factory,
                 executor_name=executor_name, plan_factory=plan_factory, model_factory=model_factory,
                 reference_factory=reference_factory, compiler_factory=compiler_factory,
                 execution_compiler=execution_compiler)
    print(f"{result['status']}: {result['completed_positions']} token positions, {result['dispatches']} native dispatches")
    print(result["text"])


if __name__ == "__main__": main()
