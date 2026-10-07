#!/usr/bin/env python3
"""Audit completed Asahi or M1 macOS evidence and optionally write its hash manifest.

This reads recorded outputs, not the GPU. It never turns missing hardware work
or COMPLETED_UNVERIFIED into a hardware correctness pass.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, manifest=None, *, expected_executor=None):
    summary = json.loads((root/"summary.json").read_text())
    if summary["status"] != "PASS": raise ValueError("bundle does not record a correctness pass")
    if "verified" in summary and not summary["verified"]: raise ValueError("model was not verified")
    if expected_executor is not None:
        platform = json.loads((root/"platform.json").read_text())
        if summary.get("executor") != expected_executor or platform.get("executor") != expected_executor:
            raise ValueError("bundle executor differs from requested backend")
    checks = summary["checks"]
    if not checks or any(c["status"] != "PASS" for c in checks): raise ValueError("missing/failed checks")
    launches = list(root.rglob("launch.json"))
    expected_launches = summary.get("dispatches", len(checks))
    if len(launches) != expected_launches: raise ValueError("launch count disagrees with receipt")
    for path in launches:
        launch = json.loads(path.read_text()); p=launch["program"]
        if launch["status"] != "COMPLETED" or not launch.get("fence_completed") or not launch.get("canaries_intact"):
            raise ValueError(f"incomplete/unguarded launch: {path}")
        executor = launch.get("executor")
        if expected_executor is not None and executor != expected_executor:
            raise ValueError(f"launch executor differs from requested backend: {path}")
        apple_baseline = executor == "Apple-compiled scalar Metal"
        if not apple_baseline and sha(path.parent/"shader.bin") != p["code_sha256"]: raise ValueError(f"shader hash: {path}")
        if p["logical_threads"] != launch["logical_threads"]: raise ValueError(f"extent mismatch: {path}")
        state_names = ("metal-archive.bin",) if executor == "G13 machine code through Metal" else ("usc.bin","cdm.bin","drm-command.bin")
        if apple_baseline:
            if launch.get("program_role") != "logical graph descriptor; G13 bytes are not executed":
                raise ValueError("Apple baseline must distinguish MSL execution from G13 descriptors")
            state_names = ("shader.metal", "metal-library.bin", "metal-archive.bin")
        if executor == "direct IOGPU G13":
            state_names = ("usc.bin", "cdm.bin", "iogpu-submit.bin", "iogpu-pages.bin", "uniforms.bin")
            completion = bytes.fromhex(launch.get("completion", ""))
            tokens = launch.get("notification_tokens", [])
            if len(completion) != 80 or len(tokens) != 2 or not all(tokens) or tokens[0] == tokens[1]:
                raise ValueError("missing native completion tokens/receipt")
            for index, token in enumerate(tokens):
                cookie, start, end, status, reserved = struct.unpack_from("<5Q", completion, index*40)
                if cookie != token or not start or end < start or status or reserved:
                    raise ValueError("failed native completion notification")
            buffers = {b["binding"]: b["gpu_address"] for b in launch["buffers"]}
            pointers = [buffers[b["name"]] for b in sorted(p["bindings"], key=lambda b: b["slot"])]
            if (path.parent/"uniforms.bin").read_bytes() != struct.pack("<"+"Q"*len(pointers), *pointers):
                raise ValueError("native uniform pointers disagree with bindings")
        for name in state_names:
            if sha(path.parent/name) != launch[name]["sha256"]: raise ValueError(f"state hash: {path}")
        if state_names == ("metal-archive.bin",):
            from agxforge.g13.metal import code_range
            archive = (path.parent/"metal-archive.bin").read_bytes()
            offset, capacity = code_range(archive)
            code = (path.parent/"shader.bin").read_bytes()
            if (offset != launch["code_offset"] or capacity != launch["code_capacity"] or
                    archive[offset:offset+len(code)] != code): raise ValueError("authored shader placement")
    tensor_checks=0
    if "verified" in summary:
        import numpy as np
        bounds = json.loads((root/"bounds.json").read_text())
        for position in checks:
            folder=root/f"token-{position['position']:03d}"
            tensors=json.loads((folder/"checks.json").read_text())
            if len(tensors)!=position["tensors"] or not position["top_token_matches"]:
                raise ValueError("incomplete tensor/argmax check")
            for row in tensors:
                if row["status"]!="PASS": raise ValueError("failed tensor check")
                name=row["tensor"]
                category=name if name in ("embedding","logits") else next(
                    (key for key in ("norm","rope","cache","scores","probs","attention","residual","activation") if key in name), "projection")
                atol,rtol=bounds[category]
                if [row["atol"],row["rtol"]] != [atol,rtol]: raise ValueError("per-tensor bounds changed")
                actual_path=folder/(name+".f32")
                if sha(actual_path)!=row["actual_sha256"]: raise ValueError("tensor hash changed")
                actual=np.frombuffer(actual_path.read_bytes(),np.float32)
                expected=np.frombuffer((folder/(name+".f64")).read_bytes(),np.float64)
                if len(actual)!=len(expected) or len(actual)!=row["elements"]: raise ValueError("tensor extent")
                finite=np.isfinite(expected)
                if not (np.all(np.isfinite(actual[finite])) and
                        np.all(np.abs(actual[finite].astype(np.float64)-expected[finite]) <= atol+rtol*np.abs(expected[finite])) and
                        np.array_equal(actual[~finite],expected[~finite])): raise ValueError("tensor outside declared bounds")
                if name=="embedding" and actual.tobytes()!=expected.astype(np.float32).tobytes(): raise ValueError("embedding bits")
                if name=="logits":
                    argmax=int(np.argmax(actual))
                    if argmax!=int(np.argmax(expected)) or argmax!=position["next_token"]: raise ValueError("argmax")
                tensor_checks+=1
    else:
        paths = list(root.rglob("check.json"))
        if (len(paths) != len(checks) or
                {path.parent for path in paths} != {path.parent for path in launches}):
            raise ValueError("every scalar launch requires exactly one result check")
        rows = [json.loads(path.read_text()) for path in paths]
        canonical = lambda row: json.dumps(row, sort_keys=True)
        if Counter(map(canonical, rows)) != Counter(map(canonical, checks)):
            raise ValueError("scalar result checks disagree with receipt")
        for path, row in zip(paths, rows):
            if row["status"]!="PASS": raise ValueError("failed scalar check")
            for name in ("slot-0-after.bin", "expected.bin"):
                if not (path.parent/name).is_file():
                    raise ValueError(f"missing scalar output artifact: {path.parent/name}")
            actual=(path.parent/"slot-0-after.bin").read_bytes()
            expected=(path.parent/"expected.bin").read_bytes()
            if not actual or not expected: raise ValueError("empty scalar output artifact")
            atol,rtol=row.get("atol",0),row.get("rtol",0)
            if atol or rtol or row.get("nan_policy")=="classification":
                import numpy as np
                a=np.frombuffer(actual,np.float32)
                dtype=np.dtype(row.get("expected_dtype","float32"))
                e=np.frombuffer(expected,dtype)
                if row.get("nan_policy")=="classification":
                    nan=np.isnan(e)
                    if not (np.all(np.isnan(a[nan])) and a[~nan].tobytes()==e[~nan].tobytes()): raise ValueError("NaN/bit contract")
                else:
                    finite=np.isfinite(e)
                    if not (np.all(np.isfinite(a[finite])) and
                            np.all(np.abs(a[finite].astype(np.float64)-e[finite])<=atol+rtol*np.abs(e[finite])) and
                            np.array_equal(a[~finite],e[~finite])): raise ValueError("scalar numerical bound")
            elif row.get("expected_dtype") and row.get("actual_dtype") != row["expected_dtype"]:
                import numpy as np
                a=np.frombuffer(actual,np.dtype(row["actual_dtype"]))
                if a.astype(row["expected_dtype"]).tobytes()!=expected: raise ValueError("exact scalar conversion")
            elif actual!=expected: raise ValueError("scalar output bits")
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*"))
           if p.is_file() and p!=manifest}
    result=dict(schema_version=1,status="PASS",scope="offline audit of recorded native evidence",
                launches=len(launches),tensor_checks=tensor_checks,files=files)
    if manifest:
        if manifest.exists(): raise ValueError("manifest already exists; choose a new destination")
        manifest.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--manifest",type=Path)
    args=parser.parse_args()
    result=audit(args.directory,args.manifest)
    print(f"PASS: audited {result['launches']} recorded launches and {result['tensor_checks']} model tensors; {len(result['files'])} artifact hashes")
