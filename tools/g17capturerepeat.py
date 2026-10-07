#!/usr/bin/env python3
"""Repeat the retained passive capture of Metal's own Submit on the running OS (MM 25.210's authored A/B control).

    python3 tools/g17capturerepeat.py OUT_DIR

Runs tools/g17authoredpaircontrol (built here) on the retained pair-1 and pair-2 archives through METAL, with
tools/g17authoredpaircapture.js attached by Frida; nothing is submitted below Metal. The archives are the retained
ones with only the archive's loader tool version set for the running OS (agxforge/g17/container.py), since Metal on
another OS refuses the retained bytes. Each run takes the machine GPU lock and fails if the GPU's recovery or event
counters change. Writes OUT_DIR/passive-submit-with-requests.json in the retained capture's format; compare it with
the retained one with tools/g17capturediff.py.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]

import g17firstpaircapture as C  # noqa: E402
import g17gpulock  # noqa: E402
from g17submittrial import gpu_events, recovery  # noqa: E402
from agxforge.g17 import container  # noqa: E402

RETAINED = ROOT / "evidence/g17-belowmetal-preflight/authored-pair-metal"


def main(out):
    out = Path(out)
    if (out / "passive-submit-with-requests.json").exists():
        raise SystemExit("refusing to overwrite a capture")
    out.mkdir(parents=True, exist_ok=True)
    for b in (1, 2):
        for ext in ("lib", "object"):
            (out / ("pair-%d.%s" % (b, ext))).write_bytes((RETAINED / ("pair-%d.%s" % (b, ext))).read_bytes())
        arc = bytearray((RETAINED / ("pair-%d.arc" % b)).read_bytes())
        old, new = container.build_cmd(26), container.build_cmd()
        if arc.count(old) != 1:
            raise SystemExit("pair-%d.arc: the retained build record is not where expected" % b)
        i = arc.find(old)
        arc[i:i + len(old)] = new
        (out / ("pair-%d.arc" % b)).write_bytes(bytes(arc))
    binary = out / "g17authoredpaircontrol"
    subprocess.run(["xcrun", "clang", "-Wall", "-Wextra", "-Werror", "-fobjc-arc", "-I", str(ROOT / "tools"),
                    str(ROOT / "tools/g17authoredpaircontrol.m"), "-framework", "Foundation", "-framework", "Metal",
                    "-o", str(binary)], check=True)
    C.TARGET, C.EVIDENCE = binary, out
    script = C.SCRIPT.read_text()
    import platform
    report = dict(scope="the retained authored A/B through Metal, passive Submit capture, repeated on macOS %s; "
                        "pair-N.arc differs from the retained archive only in the loader tool version" % platform.mac_ver()[0],
                  script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                  source_sha256=hashlib.sha256((ROOT / "tools/g17authoredpaircontrol.m").read_bytes()).hexdigest(),
                  runs=[])
    with g17gpulock.acquire("exclusive", timeout=60):
        for bias in (1, 2):
            before, prior = recovery(), gpu_events()
            row = C.capture(bias, script)
            row.update(recovery_before=before, recovery_after=recovery(),
                       new_events=sorted(set(gpu_events()) - set(prior)))
            row["passed"] &= row["recovery_before"] == row["recovery_after"] and not row["new_events"]
            report["runs"].append(row)
            print("bias", bias, "PASS" if row["passed"] else "FAIL", row["answers"], flush=True)
            if not row["passed"]:
                break
    (out / "passive-submit-with-requests.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(r["passed"] for r in report["runs"]) and len(report["runs"]) == 2 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1]))
