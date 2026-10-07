#!/usr/bin/env python3
"""Verify compact artifact hashes and rederive the two recorded performance cohorts."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
def read(path): return json.loads(path.read_text())


def verify():
    index = read(ROOT/"artifact-sha256.json")
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ROOT.rglob("*") if p.is_file() and p.name != "artifact-sha256.json"}
    if actual != index: raise ValueError("compact evidence files changed or are missing")
    receipt = read(ROOT/"receipt.json")
    for entry in receipt["failures"]:
        if entry["status"] != "FAIL": raise ValueError("failed experiment relabeled")
    for folder in (ROOT/"failures").iterdir():
        if read(folder/"summary.json")["status"] != "FAIL": raise ValueError("failure summary changed")
    for cohort in ("compiler", "earlier-native-transport"):
        folder = ROOT/(cohort+"-benchmark")
        summary = read(folder/"summary.json")
        if summary["status"] != "PASS" or summary["rounds"] != 5: raise ValueError("incomplete cohort")
        for backend, samples in summary["samples"].items():
            if len(samples) != 5: raise ValueError("missing timing rounds")
            rows = [read(p) for p in sorted(folder.glob(f"round-*-{backend}.json"))]
            if len(rows) != 5 or any(r["status"] != "PASS" for r in rows): raise ValueError("failed/missing worker")
            if [r["samples"][0] for r in rows] != samples: raise ValueError("worker and aggregate differ")
            stats = summary["statistics"][backend]
            rates = []
            expected = [c["next_token"] for c in summary["baselines"][backend]["summary"]["checks"]]
            for row, sample in zip(rows, samples):
                if row["source_sha256"] != summary["source_sha256"]: raise ValueError("worker source changed")
                if sample["next_tokens"] != expected: raise ValueError("timed argmax differs from verification")
                if len(sample["position_seconds"]) != 12 or sample["decode_positions"] != 7:
                    raise ValueError("history/measurement scope changed")
                decode = sum(sample["position_seconds"][5:])
                if sample["decode_seconds"] != decode or sample["decode_tokens_per_second"] != 7/decode:
                    raise ValueError("rate calculation differs")
                rates.append(sample["decode_tokens_per_second"])
            if stats["median_decode_tokens_per_second"] != statistics.median(rates): raise ValueError("median differs")
    compiler = read(ROOT/"compiler-benchmark/summary.json")
    if set(compiler["samples"]) != {"metal", "apple"}: raise ValueError("compiler cohort scope changed")
    earlier = read(ROOT/"earlier-native-transport-benchmark/summary.json")
    if set(earlier["samples"]) != {"metal", "iogpu"}: raise ValueError("transport cohort scope changed")
    for bundle in ("metal-model", "apple-model", "native-model"):
        summary = read(ROOT/bundle/"summary.json")
        if summary["status"] != "PASS" or summary["dispatches"] != 3264 or sum(c["tensors"] for c in summary["checks"]) != 2508:
            raise ValueError("incomplete declared model coverage")
    control = read(ROOT/"cross-backend.json")
    if control["status"] != "PASS" or control["g13_tensors_bit_identical"] != 2508:
        raise ValueError("native byte comparison changed")
    print(f"PASS: {len(index)} compact artifact hashes, two five-round cohorts, 3 model receipts, 6 retained failures")


if __name__ == "__main__": verify()
