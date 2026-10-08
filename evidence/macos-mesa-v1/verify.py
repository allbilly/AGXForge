#!/usr/bin/env python3
"""Recheck compact Mesa code identities, ABI admission and retained GPU outputs."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from agxforge.g13.abi import Binding, G13Program
from agxforge.g13.mesa import from_binary


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    actual = {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob("*")
              if p.is_file() and p.name != "artifact-sha256.json"}
    if actual != read(ROOT / "artifact-sha256.json"):
        raise ValueError("compact files changed or are missing")
    receipt = read(ROOT / "receipt.json")
    summary = read(ROOT / "summary.json")
    if receipt["status"] != "PASS" or summary["status"] != "PASS" or summary["full_model_verified"]:
        raise ValueError("incorrect verification scope")
    build = read(ROOT / "mesa-build-identity.json")
    if not build["archive_verified"] or not build["compiler_sources_verified"]:
        raise ValueError("unverified compiler sources")
    sources = read(ROOT / "source-sha256.json")
    if build["adapter_sha256"] != sources["tools/mesa_agx/bridge.c"]:
        raise ValueError("adapter and run sources differ")
    outputs = 0
    for row in summary["checks"]:
        folder = (ROOT / row["compiler_identity"]).parent.parent
        if row != read(folder / "check.json") or row["status"] != "PASS":
            raise ValueError("incomplete check")
        launch = read(folder / "launch/launch.json")
        if launch["status"] != "COMPLETED" or not launch["fence_completed"] or not launch["canaries_intact"]:
            raise ValueError("failed GPU launch")
        descriptor = launch["program"]
        if descriptor["origin"] != "Mesa 26.2.4 NIR -> AGX" or launch["executor"] != "G13 machine code through Metal":
            raise ValueError("compiler/executor scope changed")
        code = (folder / "compiler/shader.bin").read_bytes()
        if code != (folder / "launch/shader.bin").read_bytes():
            raise ValueError("submitted bytes differ from compiler output")
        identity = read(folder / "compiler/compiler-identity.json")
        if identity["compiler_sha256"] != build["compiler_sha256"] or identity["build_identity_sha256"] != digest(ROOT / "mesa-build-identity.json"):
            raise ValueError("compiler provenance changed")
        if identity["protocol_sha256"] != digest(folder / "compiler/input.ir"):
            raise ValueError("lowered IR changed")
        if identity["descriptor"] != descriptor:
            raise ValueError("compiler and launch descriptors differ")
        fields = {key: descriptor[key] for key in G13Program.__dataclass_fields__ if key != "code"}
        fields["bindings"] = tuple(Binding(**b) for b in fields["bindings"])
        for key in ("builtins", "reserved_register_halfs"): fields[key] = tuple(fields[key])
        program = G13Program(code=code, **fields)
        if program.code_hash != identity["code_sha256"] or program.code_hash != launch["shader.bin"]["sha256"]:
            raise ValueError("shader identity differs")
        from_binary(program, code, read(folder / "compiler/metadata.json"))
        data = (folder / "launch/slot-0-after.bin").read_bytes()
        expected = (folder / "expected.bin").read_bytes()
        values = [x[0] for x in struct.iter_unpack("<f", data)]
        references = [x[0] for x in struct.iter_unpack("<d" if row["expected_dtype"] == "float64" else "<f", expected)]
        if len(values) != len(references) or len(data) != program.bindings[0].min_bytes:
            raise ValueError("output extent changed")
        if row["atol"] == row["rtol"] == 0:
            if data != expected: raise ValueError("exact output differs")
        else:
            for value, reference in zip(values, references):
                if not math.isfinite(value) or not math.isfinite(reference) or abs(value-reference) > row["atol"] + row["rtol"]*abs(reference):
                    raise ValueError("GPU output exceeds numerical tolerance")
        maximum = max((abs(a-b) for a, b in zip(values, references)), default=0)
        if maximum != row["max_abs_error"]: raise ValueError("reported error differs from retained tensors")
        outputs += len(values)
    if len(summary["checks"]) != receipt["gpu_checks"] or receipt["gpu_checks"] != 30:
        raise ValueError("coverage changed")
    print(f"PASS: {len(actual)} artifact hashes, 30 Mesa shader/launch identities, {outputs} GPU output values")


if __name__ == "__main__": verify()
