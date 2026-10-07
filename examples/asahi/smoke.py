#!/usr/bin/env python3
"""Native smoke suite: audited completion, exact results, canaries, fresh VAs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agxforge.g13.smoke import OPERATIONS, program
from agxforge.runtime.asahi import Executor, source_identity


def f32(value): return struct.unpack("<f", struct.pack("<f", value))[0]
def pack(values): return struct.pack("<" + "f" * len(values), *values)


def inputs(operation, n, seed):
    rng = random.Random(seed)
    a = [f32(rng.randint(-128, 128) / 16) for _ in range(n)]
    b = [f32(rng.randint(-64, 64) / 8) for _ in range(n)]
    if operation == "fma":
        a[:2] = [1 + 2**-12, 1 + 2**-23]
        b[:2] = [1 - 2**-12, 1 - 2**-23]
    if operation in ("lane", "group", "global"):
        values = [i % 32 if operation == "lane" else i // 32 if operation == "group" else i for i in range(n)]
        expected = struct.pack("<" + "I" * n, *values)
    else:
        expected = pack([1 if operation == "store" else a[i] if operation == "copy" else
                         f32(a[i] + b[i]) if operation == "add" else f32(a[i] * b[i]) if operation == "mul" else
                         f32(a[i] - b[i]) if operation == "sub" else f32(math.fma(a[i], b[i], -1)) for i in range(n)])
    return pack(a), pack(b), expected


def run(output, operations=OPERATIONS, sizes=(32, 64, 128), repeats=3, device=None,
        *, executor_factory=Executor, identity_factory=source_identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = identity_factory()
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  status="RUNNING", scope="handwritten native G13G scalar execution", checks=[])
    (output / "source-sha256.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        for operation in operations:
            for n in sizes:
                for repetition in range(repeats):
                    seed = 101 + repetition * 17 + n
                    a, b, expected = inputs(operation, n, seed)
                    folder = output / f"{operation}-{n}-{repetition}"
                    row = dict(operation=operation, threads=n, seed=seed, repetition=repetition, status="NOT_RUN")
                    report["checks"].append(row)
                    # Every test owns a fresh VM/queue/BO set; VA slots vary.
                    with executor_factory(device, va_slot=repetition) as gpu:
                        if not (output / "platform.json").exists():
                            (output / "platform.json").write_text(json.dumps(gpu.platform(), indent=2) + "\n")
                        out = gpu.buffer(n * 4, "write")
                        bb = gpu.buffer(n * 4, "read", b)
                        aa = gpu.buffer(n * 4, "read", a)
                        launch = gpu.dispatch(program(operation, n), [out, bb, aa], n, evidence=folder)
                        actual = out.read()
                        (folder / "expected.bin").write_bytes(expected)
                        passed = (launch["fence_completed"] and actual == expected and aa.read() == a and bb.read() == b)
                        row.update(status="PASS" if passed else "WRONG_OUTPUT", fence_completed=True,
                                   expected_sha256=hashlib.sha256(expected).hexdigest(), actual_sha256=hashlib.sha256(actual).hexdigest(),
                                   canaries_intact=launch["canaries_intact"], teardown="pending")
                        (folder / "check.json").write_text(json.dumps(row, indent=2) + "\n")
                        if not passed:
                            gpu.poisoned = True
                            raise RuntimeError(f"{operation} n={n} seed={seed}: wrong output; suite stopped")
                    row["teardown"] = "clean"
                    (folder / "check.json").write_text(json.dumps(row, indent=2) + "\n")
        if identity_factory() != identity: raise RuntimeError("sources changed during run")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(operation=None, *, executor_factory=Executor, identity_factory=source_identity, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--operation", choices=OPERATIONS, default=operation)
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100 or any(n < 32 or n > 4096 or n % 32 for n in args.sizes):
        parser.error("repeats 1..100; sizes complete SIMD groups in 32..4096")
    report = run(args.output, (args.operation,) if args.operation else OPERATIONS, args.sizes, args.repeats, args.device,
                 executor_factory=executor_factory, identity_factory=identity_factory)
    print(f"{report['status']}: {len(report['checks'])} native dispatches; evidence: {args.output}")


if __name__ == "__main__": main()
