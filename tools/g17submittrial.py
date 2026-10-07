#!/usr/bin/env python3
"""Prepared three-run Metal reference capture. Requires explicit user approval.

Default prints the plan and launches nothing. --execute is an operational switch,
not a substitute for that approval. This does not perform below-Metal submission.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import g17gpulock
from g17submitabi import decode_records

ROOT = Path(__file__).resolve().parents[1]


def recovery():
    text = subprocess.check_output(["ioreg", "-l", "-r", "-c", "AGXAcceleratorG17X"], text=True)
    matches = re.findall(r'"recoveryCount"\s*=\s*(\d+)', text)
    if not matches:
        raise RuntimeError("GPU recovery counter unavailable; refusing to run")
    return [int(x) for x in matches]


def gpu_events():
    return sorted(glob.glob("/Library/Logs/DiagnosticReports/gpuEvent-*.ips"))


def capture(binary, bias, script_source):
    import frida
    device = frida.get_local_device()
    done = threading.Event()
    messages, output, detach = [], [], []
    pid = None

    def on_output(owner, fd, data):
        if owner == pid:
            output.append({"fd": fd, "text": data.decode(errors="replace")})

    def on_detach(*args):
        detach.append(str(args))
        done.set()

    device.on("output", on_output)
    session = None
    resumed = False
    try:
        # Explicit argv is required; a positional list did not pass the extra
        # argument in the CPU-only smoke check on this installed Frida binding.
        pid = device.spawn(str(binary), argv=[str(binary), str(bias)], stdio="pipe")
        session = device.attach(pid)
        session.on("detached", on_detach)
        agent = session.create_script(script_source)
        agent.on("message", lambda msg, data: messages.append(msg))
        agent.load()
        resumed = True
        device.resume(pid)
        started = time.monotonic()
        while not done.wait(5):
            print(f"Waiting for reference pid={pid}, elapsed={time.monotonic()-started:.0f}s; lock remains held", flush=True)
        # Output delivery may trail the detached notification.
        time.sleep(0.2)
    except Exception:
        if pid is not None and not resumed:
            device.kill(pid)  # still spawn-suspended, no program/GPU work ran
        raise
    finally:
        device.off("output", on_output)
    stdout = "".join(o["text"] for o in output if o["fd"] == 1)
    answers = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "producer" in value:
                answers.append(value)
        except ValueError:
            pass
    events = [m["payload"] for m in messages if m.get("type") == "send"]
    errors = [m for m in messages if m.get("type") == "error" or m.get("payload", {}).get("kind") == "capture_error"]
    submits = [e for e in events if e.get("kind") == "submit_enter"]
    decoded = [decode_records(bytes.fromhex(e["hex"]), e["count"], e["stride"]) for e in submits]
    passed = (len(answers) == 1 and answers[0].get("output") == [3*i+bias for i in range(64)]
              and answers[0].get("guards_correct") == 32 and answers[0].get("status") == 4
              and not errors and len(submits) == 1
              and any(e.get("kind") == "instrumentation_ready" for e in events)
              and any(e.get("kind") == "submit_leave" and e.get("status") == 0 for e in events))
    return {"pid": pid, "bias": bias, "passed": passed, "answers": answers,
            "messages": messages, "output": output, "detached": detach, "decoded_records": decoded}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    print("Plan: fresh-process Metal reference A(1), A-repeat(1), B(2); one dispatch each.")
    if not args.execute:
        print("No process launched. Explicit user authorization is required before --execute.")
        return 0
    if args.out.exists():
        raise RuntimeError("refusing to overwrite trial directory")
    binary = args.binary.resolve(strict=True)
    script = (ROOT / "tools/g17submitcapture.js").read_text()
    args.out.mkdir(parents=True)
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "capture_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "scope": "Apple-compiled Metal reference; not authored-code or below-Metal execution", "runs": []}
    for path in ("tools/g17submitcontrol.m", "tools/g17submittrial.py", "tools/g17submitabi.py"):
        manifest.setdefault("source_hashes", {})[path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    with g17gpulock.acquire("exclusive", timeout=60):
        for name, bias in (("A", 1), ("A-repeat", 1), ("B", 2)):
            before = recovery()
            events_before = gpu_events()
            row = capture(binary, bias, script)
            row.update(name=name, recovery_before=before, recovery_after=recovery(),
                       gpu_events_before=events_before, gpu_events_after=gpu_events())
            row["passed"] &= (row["recovery_before"] == row["recovery_after"]
                               and row["gpu_events_before"] == row["gpu_events_after"])
            manifest["runs"].append(row)
            (args.out / "report.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print(name, "PASS" if row["passed"] else "FAIL; batch stopped", flush=True)
            if not row["passed"]:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
