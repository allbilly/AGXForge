"""Workflow 3: inspect the matched decode study (MM 25.211) and re-derive its numbers from the retained receipts.

    python3 examples/decode_study.py

READS RETAINED EVIDENCE ONLY. Nothing is compiled or dispatched. The study itself ran THROUGH METAL on one M5 Pro;
reproducing it needs the GPU, mlx-lm and the InternLM2.5-1.8B-chat weights (examples/README.md, workflow 3).

The question the study answers: would this project's decode design - its algorithms, fusions and execution
structure - run as fast if Apple's compiler built the kernels? Each kernel has a Metal twin (tools/twins/*.metal) with
the same threads, work partition and fp32 operation order, compiled by `xcrun metal -O3 -fno-fast-math
-ffp-contract=off`. Three arms decode mlx-lm's own 128-token greedy sequence after a 1,792-token prompt.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence" / "g17-matched-study-v1"
ARMS = (("mlx", "mlx-lm generate_step"), ("graph_forced", "AGXForge design, AGXForge compiler"),
        ("graph_twin_all_forced", "AGXForge design, Apple's compiler"))
SOURCES = {
    "the route, step by step": ["agxforge/inference/__init__.py"],
    "graph configuration": ["tools/models/internlm2_q4_spec_code.json"],
    "kernel forms and layouts": ["agxforge/inference/catalog.py"],
    "decode kernels": ["agxforge/kernels/qmv.py", "agxforge/kernels/attention.py", "agxforge/kernels/norm.py",
                       "agxforge/kernels/argmax.py"],
    "prefill kernels": ["agxforge/kernels/prefill_mma.py", "agxforge/kernels/prefill_attention.py",
                        "agxforge/kernels/qmm.py", "agxforge/kernels/rows.py"],
    "arithmetic the references share": ["agxforge/kernels/numerics.py"],
    "verification and delivery": ["tools/g17deliver.py", "agxforge/inference/index.py"],
    "graph assembly": ["agxforge/inference/assemble.py", "agxforge/inference/prefill.py",
                       "agxforge/inference/graph.py"],
    "execution": ["agxforge/inference/execute.py", "tools/g17decodegen.m"],
    "Metal twins": ["tools/twins/qmv_twin.metal", "tools/twins/attn_twin.metal", "tools/twins/misc_twin.metal"],
    "comparison harness": ["tools/g17twin.py", "tools/g17twinrun.m", "tools/g17model_mlx.py"],
    "reproduction helpers": ["tools/g17forcehistory.py", "tools/g17promptids.py"],
}


def main():
    shared = json.loads((EV / "decode-shared-history.json").read_text())
    kernels = json.loads((EV / "kernels.json").read_text())
    print("receipt:", (EV / "decode-shared-history.json").relative_to(ROOT))
    print("instrument:", shared["instrument"])
    print("shared history:", shared["shared_history"][:160], "...\n")

    by_arm = {}
    for row in shared["rows"]:
        by_arm.setdefault(row["arm"], {})[row["rep"]] = row["tok_s"]
    medians = {arm: statistics.median(reps.values()) for arm, reps in by_arm.items()}
    base = medians["mlx"]
    print("%-50s %10s %22s" % ("arm (all through Metal)", "median", "per-repetition / mlx-lm"))
    failures = []
    for arm, label in ARMS:
        reps = by_arm[arm]
        ratios = [reps[r] / by_arm["mlx"][r] for r in sorted(reps)]
        print("%-50s %10.1f %22s" % (label, medians[arm], "%.2f-%.2f" % (min(ratios), max(ratios))))
        if abs(round(medians[arm], 1) - round(shared["tok_s"][arm]["median"], 1)) > 1e-9:
            failures.append("median of %s does not match the receipt's summary" % arm)
    for arm, want in (("mlx", 162.2), ("graph_forced", 184.5), ("graph_twin_all_forced", 213.4)):
        if round(medians[arm], 1) != want:
            failures.append("%s median %.1f, published %.1f" % (arm, medians[arm], want))
    print("\ntoken agreement under the shared history:", shared["forcing_check"][:220], "...")
    print("disagreements with mlx-lm:", [d["index"] for d in shared["disagreements_vs_mlx_full_forward"]])

    print("\nper kernel (kernels.json): Apple-compiled twin time / ours, and whether the outputs are identical")
    for name, row in list(kernels["qmv"].items()) + [("attention q0=" + k, v) for k, v in kernels["attn"].items()]:
        identical = row.get("bytes_differ", row.get("out_bytes_differ")) == 0
        if not identical:
            failures.append("%s outputs differ" % name)
        print("  %-22s ours %7.1f us  twin %7.1f us  ratio %.2f  outputs %s"
              % (name, row["ours_us"], row["apple_us"], row["apple_over_ours"], "identical" if identical else "DIFFER"))

    twin_gain = [by_arm["graph_twin_all_forced"][r] / by_arm["graph_forced"][r] for r in sorted(by_arm["graph_forced"])]
    print("\nwhat this establishes: the speed comes from the decode implementation, which survives Apple compilation;"
          "\nit does not establish any benefit from this project's native instruction control (Apple's compiler is"
          "\n%.0f-%.0f percent faster on the same kernels in this receipt, 10-15 percent at the median across the"
          "\nstudy's runs). One model, one context, four repetitions; at a 196-token context this project's median is"
          "\nslightly below mlx-lm's (0.91-1.02x per repetition in free generation, MM 25.211)." % (100 * (min(twin_gain) - 1),
                                                                                100 * (max(twin_gain) - 1)))
    print("\nsources:")
    for what, paths in SOURCES.items():
        missing = [p for p in paths if not (ROOT / p).is_file()]
        failures += ["missing source %s" % p for p in missing]
        print("  %-32s %s" % (what, ", ".join(paths)))
    rec = json.loads((EV / "inputs-recovery.json").read_text())
    print("\nreproduction:", rec["conclusion"])
    if failures:
        print("\nFAILED:", *failures, sep="\n  ")
        return 1
    print("\nall published numbers re-derived from the per-repetition rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
