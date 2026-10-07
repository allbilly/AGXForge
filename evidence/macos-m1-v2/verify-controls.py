#!/usr/bin/env python3
"""Check native relocation, exact Metal agreement, and fresh launch IDs from full bundles."""
import argparse
import json
from pathlib import Path
import struct


def check(baseline, relocated, metal):
    for bundle in (baseline,relocated,metal):
        report = json.loads((bundle/"summary.json").read_text())
        if report["status"] != "PASS" or not report["verified"]:
            raise ValueError("controls require verified model bundles")
    plans = [json.loads((p/"plan.json").read_text())["programs"] for p in (baseline,relocated,metal)]
    if plans[0] != plans[1] or plans[0] != plans[2]: raise ValueError("compiled model plans differ")
    count = 0
    for token in sorted(baseline.glob("token-*")):
        for file in sorted(token.glob("*.f32")):
            raw = file.read_bytes()
            if raw != (relocated/token.name/file.name).read_bytes(): raise ValueError(f"relocation tensor: {file}")
            if raw != (metal/token.name/file.name).read_bytes(): raise ValueError(f"Metal tensor: {file}")
            count += 1
    if count != 2754: raise ValueError("all six positions and 459 tensors per position required")
    counts = []
    for bundle in (baseline,relocated):
        seen = set()
        folders = sorted((bundle/"launches").iterdir())
        if len(folders) != 3780: raise ValueError("full native launch count required")
        for folder in folders:
            launch = json.loads((folder/"launch.json").read_text())
            command,encoder = launch["trace_ids"]
            if not command or encoder != command+1 or command in seen or encoder in seen:
                raise ValueError("reused or invalid native trace IDs")
            pages = (folder/"iogpu-pages.bin").read_bytes()
            for offset in (0,0x18):
                if struct.unpack_from("<Q",pages,offset)[0] != command: raise ValueError("command ID mismatch")
            if (struct.unpack_from("<Q",pages,0x28)[0] != encoder or
                    struct.unpack_from("<I",pages,0x4000+0x23c)[0] != encoder):
                raise ValueError("encoder ID mismatch")
            seen.update((command,encoder))
        counts.append(len(seen))
    first = "launches/000001-embedding/launch.json"
    addresses = [[b["gpu_address"] for b in json.loads((p/first).read_text())["buffers"]]
                 for p in (baseline,relocated)]
    if addresses[0] == addresses[1]: raise ValueError("GPU addresses did not relocate")
    return dict(status="PASS",tensors_bit_identical=count,metal_tensors_bit_identical=count,
                baseline_va_slot=0,relocated_va_slot=7,baseline_gpu_addresses=addresses[0],
                relocated_gpu_addresses=addresses[1],launches=3780,
                unique_command_and_encoder_ids=counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","relocated","metal","output"):parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    result = check(args.baseline,args.relocated,args.metal)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
