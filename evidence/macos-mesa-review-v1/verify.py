#!/usr/bin/env python3
"""Verify the retained review regressions using integer input/output oracles."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from tools.asahi_evidence import audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def words(path, width):
    data = path.read_bytes()
    require(len(data) % width == 0, "partial integer element")
    return [entry[0] for entry in struct.iter_unpack("<H" if width == 2 else "<I", data)]


def expected_output(folder, launch):
    name = launch["program"]["name"]
    n = launch["logical_threads"]
    memory = re.fullmatch(r"(load|store)_i(16|32)_offset([01])_disp([02])", name)
    shift = re.fullmatch(r"(shl|shr)_i(16|32)_to_i(16|32)", name)
    immediate = re.fullmatch(r"(shl|shr)_immediate_to_i16", name)
    if memory:
        kind, bits, offset, disp = memory.groups()
        width = int(bits)//8
        displacement = int(offset)+int(disp)
        source = words(folder/"slot-1-before.bin", width)
        if kind == "load":
            result = source[displacement:displacement+n]
            require(len(result) == n, "load oracle exceeds input")
        else:
            result = words(folder/"slot-0-before.bin", width)
            require(len(source) == n and displacement+n <= len(result), "store oracle exceeds extent")
            result[displacement:displacement+n] = source
    elif shift or immediate:
        if shift:
            kind, source_bits, dest_bits = shift.groups()
            source = words(folder/"slot-1-before.bin", int(source_bits)//8)
            counts = words(folder/"slot-2-before.bin", 4)
        else:
            kind = immediate.group(1)
            source = words(folder/"slot-1-before.bin", 4 if kind == "shr" else 2)
            dest_bits = "16"
            counts = [1 if kind == "shr" else 16]*n
        width = int(dest_bits)//8
        mask = (1 << int(dest_bits))-1
        require(len(source) == n and len(counts) == n, "shift oracle extent")
        result = [((value << (count&127)) if kind == "shl" else (value >> (count&127))) & mask
                  for value, count in zip(source, counts)]
    else:
        raise ValueError(f"unrecognized regression: {name}")
    return struct.pack("<"+("H" if width == 2 else "I")*len(result), *result)


def main():
    index = json.loads((ROOT/"artifact-sha256.json").read_text())
    actual = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob("*")
              if p.is_file() and p.name != "artifact-sha256.json" and "__pycache__" not in p.parts}
    require(actual == index["files"], "artifact hash index")
    receipt = json.loads((ROOT/"receipt.json").read_text())
    build = json.loads((ROOT/"mesa-build-identity.json").read_text())
    outputs = {}
    expected_names = {f"{kind}_i{bits}_offset{offset}_disp{disp}"
                      for kind in ("load", "store") for bits in (16, 32)
                      for offset, disp in ((1, 0), (0, 2), (1, 2))}
    expected_names |= {f"{kind}_i{source}_to_i{dest}" for kind in ("shl", "shr")
                       for source in (16, 32) for dest in (16, 32)}
    expected_names |= {"shr_immediate_to_i16", "shl_immediate_to_i16"}
    for route, executor in (("metal", "G13 machine code through Metal"), ("native", "direct IOGPU G13")):
        root = ROOT/route
        audited = audit(root, expected_executor=executor)
        require(audited["launches"] == 22, "regression launch count")
        full_audit = json.loads(gzip.decompress((root/"full-audit.json.gz").read_bytes()))
        full_summary = json.loads((root/"full-summary.json").read_text())
        require(full_audit["status"] == "PASS" and full_audit["launches"] == 134 and
                full_summary["status"] == "PASS" and len(full_summary["checks"]) == 134, "full suite receipt")
        require(sha(root/"full-summary.json") == full_audit["files"]["summary.json"], "full summary identity")
        require(sha(root/"full-audit.json.gz") == receipt["runs"][route]["full_audit_gzip_sha256"], "full audit identity")
        for p in root.rglob("*"):
            if p.is_file() and str(p.relative_to(root)) in full_audit["files"] and p.name != "summary.json":
                require(sha(p) == full_audit["files"][str(p.relative_to(root))], "retained original artifact")
        outputs[route] = {}
        for path in sorted(root.glob("*/launch.json")):
            launch = json.loads(path.read_text())
            name = launch["program"]["name"]
            result = expected_output(path.parent, launch)
            require(result == (path.parent/"expected.bin").read_bytes() ==
                    (path.parent/"slot-0-after.bin").read_bytes(), f"integer oracle: {route}/{name}")
            outputs[route][name] = (launch["program"], result)
        require(set(outputs[route]) == expected_names, "regression coverage")
        provenance = json.loads((root/"summary.json").read_text())["compiler_provenance"]
        require(len(provenance["programs"]) == 22, "compiler receipt coverage")
        for entry in provenance["programs"].values():
            identity = json.loads((root/entry["receipt"]).read_text())
            require(identity["compiler_sha256"] == build["compiler_sha256"], "helper cohort identity")
            require(identity == {k: v for k, v in entry.items() if k != "receipt"}, "plan compiler identity")
    require(outputs["metal"] == outputs["native"], "cross-transport code/results")
    require((ROOT/"metal/source-sha256.json").read_bytes() ==
            (ROOT/"native/source-sha256.json").read_bytes(), "source cohort identity")
    platform = json.loads((ROOT/"native/platform.json").read_text())
    require(platform["metal_api_calls"] == 0 and platform["agx_metal_absent"] is True, "native transport")
    regression = json.loads((ROOT/"compiler-regression.json").read_text())
    previous = json.loads((ROOT.parent/"macos-mesa-v2/compiler-regression.json").read_text())
    require(regression["status"] == "PASS" and regression["authored_codegen_blocked"] is True and
            regression["independently_mesa_compiled"] == 43 and
            regression["default_programs"] == previous["default_programs"] and
            regression["mesa_programs"] == previous["mesa_programs"], "model compiler regression")
    require("Ran 89 tests" in (ROOT/"unit-checks.log").read_text() and
            (ROOT/"unit-checks.log").read_text().rstrip().endswith("OK"), "software checks")
    print(f"PASS: {len(actual)} hashes, 44 GPU regression oracles, two 134-check receipts, 43 unchanged model programs")


if __name__ == "__main__":
    main()
