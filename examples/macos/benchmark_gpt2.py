#!/usr/bin/env python3
"""Matched GPT-2 comparison of G13 Metal, direct IOGPU and Apple-compiled Metal.

Requires audited, verified bundles for each backend with the same history and prompt.
Warm pipelines, allocations, CPU reference work, disk capture and logits reads are
outside measured intervals. GPU waits, guards and per-token argmax reads are timed.
Each backend runs in a separate child process; round order alternates to reduce drift.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agxforge.g13.qwen import Checkpoint
from agxforge.g13.gpt2 import GPT2Plan, GPT2
from agxforge.runtime.macos import source_identity
from tools.asahi_evidence import audit

EXECUTORS = {"metal": "G13 machine code through Metal", "iogpu": "direct IOGPU G13",
             "apple": "Apple-compiled scalar Metal"}


def validate_baseline(summary, backend):
    if (summary.get("status") != "PASS" or summary.get("verified") is not True
            or summary.get("executor") != EXECUTORS[backend]):
        raise ValueError("verified baseline executor differs from requested backend")
    prompt = summary["input_tokens"]
    history = [row["token"] for row in summary["checks"]]
    if not prompt or history[:len(prompt)] != prompt:
        raise ValueError("verified prompt differs from token history")
    if len(history) <= len(prompt): raise ValueError("benchmark requires at least one decode position")


def worker(args):
    import numpy as np
    sources = source_identity()
    checkpoint = Checkpoint(args.checkpoint)
    baseline = json.loads((args.baseline/"summary.json").read_text())
    validate_baseline(baseline, args.backend)
    expected_tokens = [row["next_token"] for row in baseline["checks"]]
    history = [row["token"] for row in baseline["checks"]]
    prompt_length = len(baseline["input_tokens"])
    plan = GPT2Plan(checkpoint, args.capacity)
    if args.backend == "apple":
        from examples.macos.apple_metal import Executor
        kwargs = dict(plan=plan)
    elif args.backend == "iogpu":
        from agxforge.runtime.iogpu import Executor
        kwargs = dict(arena_bytes=640*1024*1024)
    else:
        from agxforge.runtime.macos import Executor
        kwargs = {}
    result = dict(status="RUNNING", backend=args.backend, round=args.round,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), samples=[])
    try:
        start = time.perf_counter()
        with Executor(timeout_ms=30000, **kwargs) as gpu:
            result["platform"] = gpu.platform()
            model = GPT2(gpu, checkpoint, plan)
            for program in plan.programs.values(): gpu.prepare(program)
            result["setup_seconds"] = time.perf_counter()-start
            for iteration in range(args.warmup+1):
                # Active positions are overwritten before use; future positions are masked.
                model.position = 0
                tokens, seconds = [], []
                for token in history:
                    start = time.perf_counter_ns()
                    tokens.append(model.consume(token))
                    seconds.append((time.perf_counter_ns()-start)/1e9)
                if tokens != expected_tokens:
                    gpu.poisoned = True
                    raise RuntimeError("timed GPU argmax differs from the independently verified history")
                raw = model.temp["logits"].read()
                actual = np.frombuffer(raw, np.float32).astype(np.float64)
                expected = np.frombuffer((args.baseline/f"token-{len(history)-1:03d}"/"logits.f64").read_bytes(), np.float64)
                if not (np.all(np.isfinite(actual)) and np.all(np.abs(actual-expected) <= .05+.003*np.abs(expected))):
                    gpu.poisoned = True
                    raise RuntimeError("timed final logits differ from FP64 reference")
                if iteration == args.warmup:
                    decode_seconds = sum(seconds[prompt_length:])
                    result["samples"].append(dict(position_seconds=seconds, next_tokens=tokens,
                        prefill_seconds=sum(seconds[:prompt_length]), decode_seconds=decode_seconds,
                        decode_positions=len(history)-prompt_length,
                        decode_tokens_per_second=(len(history)-prompt_length)/decode_seconds,
                        total_seconds=sum(seconds), final_logits_sha256=hashlib.sha256(raw).hexdigest(),
                        final_logits_max_abs_error=float(np.max(np.abs(actual-expected)))))
            result["dispatches"] = model.dispatches
        if source_identity() != sources: raise RuntimeError("sources changed during benchmark")
        result.update(status="PASS", source_sha256=sources)
    except BaseException as error:
        result.update(status="FAIL", error=str(error)); raise
    finally:
        args.output.write_text(json.dumps(result, indent=2)+"\n")


def compare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_identity()
    receipts = {}
    checkpoint_receipt = Checkpoint(args.checkpoint).receipt()
    folders = {"metal": args.metal}
    if args.iogpu: folders["iogpu"] = args.iogpu
    if args.apple: folders["apple"] = args.apple
    for backend, folder in folders.items():
        result = audit(folder, expected_executor=EXECUTORS[backend])
        summary = json.loads((folder/"summary.json").read_text())
        validate_baseline(summary, backend)
        if json.loads((folder/"checkpoint.json").read_text()) != checkpoint_receipt:
            raise ValueError("benchmark checkpoint differs from verified weights")
        if json.loads((folder/"source-sha256.json").read_text()) != sources:
            raise ValueError("benchmark sources differ from verified sources")
        plan = json.loads((folder/"plan.json").read_text())
        if plan["architecture"] != "gpt2" or plan["capacity"] != args.capacity:
            raise ValueError("benchmark architecture/capacity differs from verified plan")
        receipts[backend] = dict(path=str(folder.resolve()), summary=summary,
            summary_sha256=hashlib.sha256((folder/"summary.json").read_bytes()).hexdigest(),
            audited_launches=result["launches"], audited_tensors=result["tensor_checks"])
    histories = [[(r["token"], r["next_token"]) for r in entry["summary"]["checks"]] for entry in receipts.values()]
    if any(history != histories[0] for history in histories[1:]): raise ValueError("verified histories differ between backends")
    prompt = receipts["metal"]["summary"]["input_tokens"]
    if any(entry["summary"]["input_tokens"] != prompt for entry in receipts.values()):
        raise ValueError("verified prompts differ between backends")
    report = dict(schema_version=1, status="RUNNING", checkpoint=checkpoint_receipt,
        scope="matched scalar graph: authored G13 Metal, direct IOGPU, and optional Apple-compiled MSL",
        timing="wall-clock token execution including GPU waits, guards, scheduling and argmax reads",
        excluded=["checkpoint download", "model upload", "pipeline creation", "FP64 reference", "evidence disk writes", "logits readback"],
        rounds=args.rounds, warmup_sequences_per_worker=args.warmup, capacity=args.capacity,
        baselines=receipts, samples={backend: [] for backend in folders}, order=[])
    try:
        for round_id in range(args.rounds):
            backends = list(folders)
            offset = round_id % len(backends)
            order = backends[offset:]+backends[:offset]
            if (round_id//len(backends)) % 2: order.reverse()
            for backend in order:
                output = args.output/f"round-{round_id:02d}-{backend}.json"
                folder = folders[backend]
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--backend", backend,
                    "--checkpoint", str(args.checkpoint), "--baseline", str(folder), "--capacity", str(args.capacity),
                    "--round", str(round_id), "--warmup", str(args.warmup), "--output", str(output)]
                subprocess.run(command, check=True)
                row = json.loads(output.read_text())
                if row["status"] != "PASS": raise ValueError("benchmark worker failed")
                report["samples"][backend].extend(row["samples"])
                report["order"].append(dict(round=round_id, backend=backend))
                print(f"round {round_id+1}, {backend}: {row['samples'][0]['decode_tokens_per_second']:.3f} decode tokens/s", flush=True)
                (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
        stats = {}
        for backend, samples in report["samples"].items():
            rate = [s["decode_tokens_per_second"] for s in samples]
            stats[backend] = dict(median_decode_tokens_per_second=statistics.median(rate),
                min_decode_tokens_per_second=min(rate), max_decode_tokens_per_second=max(rate),
                median_prefill_seconds=statistics.median(s["prefill_seconds"] for s in samples),
                median_total_seconds=statistics.median(s["total_seconds"] for s in samples))
        report["statistics"] = stats
        if "iogpu" in stats:
            report["iogpu_over_metal_decode_ratio"] = stats["iogpu"]["median_decode_tokens_per_second"]/stats["metal"]["median_decode_tokens_per_second"]
        if "apple" in stats:
            report["apple_over_g13_metal_decode_ratio"] = stats["apple"]["median_decode_tokens_per_second"]/stats["metal"]["median_decode_tokens_per_second"]
        if source_identity() != sources: raise RuntimeError("sources changed during comparison")
        report.update(status="PASS", source_sha256=sources)
    except BaseException as error:
        report.update(status="FAIL", error=str(error)); raise
    finally:
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["statistics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"models/macos/gpt2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--metal", type=Path)
    parser.add_argument("--iogpu", type=Path)
    parser.add_argument("--apple", type=Path, help="optional verified Apple-compiled scalar baseline")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("metal", "iogpu", "apple"), help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1: parser.error("positive rounds and at least one warmup sequence required")
    if args.worker: worker(args)
    else:
        if not args.metal or not (args.iogpu or args.apple): parser.error("--metal and at least one of --iogpu/--apple verified evidence directories required")
        compare(args)
