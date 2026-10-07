#!/usr/bin/env python3
"""Prepare a local base-M1 driver support cache without submitting GPU work."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.runtime.macos import ROOT

DIRECTORY = ROOT / "build/macos"


def main():
    build = subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()
    if build != "26A434": raise RuntimeError("support profile requires macOS build 26A434")
    collector = DIRECTORY / "libagxforge_support.dylib"
    if "--collect" not in sys.argv:
        env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(collector))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--collect"], env=env, check=True)
        return
    from agxforge.g13.smoke import program
    from agxforge.runtime.macos import Executor
    lib = C.CDLL(str(collector))
    lib.agxforge_write_support.argtypes = [C.c_char_p]
    lib.agxforge_write_support.restype = C.c_int
    temporary = DIRECTORY / "iogpu-support.bin.tmp"
    with Executor() as gpu:
        gpu.prepare(program("store", 32))
        if lib.agxforge_write_support(os.fsencode(temporary)):
            raise RuntimeError("measured driver code heap not found; support was not collected")
    data = temporary.read_bytes()
    if len(data) != 0x5000 or not any(data[64:0x2840]):
        temporary.unlink()
        raise RuntimeError("missing driver helper code")
    target = DIRECTORY / "iogpu-support.bin"
    temporary.replace(target)
    receipt = dict(schema_version=1, build=build, gpu="Apple M1", bytes=len(data),
                   sha256=hashlib.sha256(data).hexdigest(), code_start=0x5000,
                   template_sha256=hashlib.sha256((ROOT/"agxforge/runtime/iogpu-26A434.json").read_bytes()).hexdigest(),
                   preparation="local Metal pipeline creation; GPU submissions refused by interposer")
    (DIRECTORY/"iogpu-support.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("PASS: local IOGPU support prepared; no GPU commands submitted")


if __name__ == "__main__": main()
