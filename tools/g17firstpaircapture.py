#!/usr/bin/env python3
"""Passively capture two authorized authored Metal controls, one per process."""
import hashlib
import json
from pathlib import Path
import threading
import time

import frida
import g17gpulock
from g17submittrial import recovery, gpu_events

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/g17-belowmetal-preflight/authored-pair-metal"
SCRIPT = ROOT / "tools/g17authoredpaircapture.js"
TARGET = Path("/tmp/g17authoredpaircontrol")


def capture(bias, script):
    device = frida.get_local_device()
    output, messages = [], []
    done = threading.Event()
    argv = [str(TARGET), str(bias), str(EVIDENCE / f"pair-{bias}.lib"),
            str(EVIDENCE / f"pair-{bias}.arc")]
    pid = device.spawn(str(TARGET), argv=argv, stdio="pipe")
    def on_output(owner, fd, data):
        if owner == pid:
            output.append(dict(fd=fd, text=data.decode(errors="replace")))
    device.on("output", on_output)
    session = device.attach(pid)
    session.on("detached", lambda *args: done.set())
    agent = session.create_script(script)
    agent.on("message", lambda msg, data: messages.append(msg))
    agent.load()
    device.resume(pid)
    if not done.wait(30):
        device.kill(pid)
        done.wait(2)
        raise TimeoutError("authored Metal capture timed out; stop the batch")
    time.sleep(0.2)
    device.off("output", on_output)
    stdout = "".join(o["text"] for o in output if o["fd"] == 1)
    answers = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    events = [m["payload"] for m in messages if m.get("type") == "send"]
    entered = [e for e in events if e.get("kind") == "submit_enter"]
    left = [e for e in events if e.get("kind") == "submit_leave"]
    errors = [e for e in events if e.get("kind") in ("capture_error", "install_error")]
    passed = (len(answers) == 1 and answers[0].get("bias") == bias and
              answers[0].get("status") == 4 and answers[0].get("exact_64") == 64 and
              answers[0].get("sentinel_192") == 192 and len(entered) == 1 and
              len(left) == 1 and left[0]["status"] == 0 and not errors)
    return dict(pid=pid, bias=bias, output=output, messages=messages,
                answers=answers, passed=passed)


def main():
    script = SCRIPT.read_text()
    out = EVIDENCE / "passive-submit-with-requests.json"
    if out.exists():
        raise ValueError("refusing to overwrite capture")
    report = dict(scope="exact authored A/B through Metal; passive Submit capture",
                  script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                  binary_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256((ROOT / "tools/g17authoredpaircontrol.m").read_bytes()).hexdigest(),
                  runs=[])
    with g17gpulock.acquire("exclusive", timeout=60):
        for bias in (1, 2):
            before, prior_events = recovery(), gpu_events()
            row = capture(bias, script)
            row.update(recovery_before=before, recovery_after=recovery(),
                       new_events=sorted(set(gpu_events()) - set(prior_events)))
            row["passed"] &= row["recovery_before"] == row["recovery_after"] and not row["new_events"]
            report["runs"].append(row)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print("bias", bias, "PASS" if row["passed"] else "FAIL", flush=True)
            if not row["passed"]:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
