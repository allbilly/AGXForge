"""A compiled decode kernel as the bundle the executors load, and its buffers.

A BUNDLE is a directory: the program (program.bin), its image (scan.o, scan.lib.metallib, scan.arc.metallib), its
manifest (manifest.json: the ABI, the grid, the instruction table, every hash), its layout (decodeop.json) and its
three input buffers (a.f16, b.f16, c.f32). `author` writes one and refuses to overwrite; the image is
agxforge.g17.scanlink's, and the archive is checked to contain the object it was authored from.

The buffers start with the carrier's tiles (agxforge.kernels.emit._carrier): carrier_tiles draws them,
carrier_reference is their product, _with_carrier places them.

Moved from tools/g17decodeops.py; generic_manifest is gemm_generic's branch of
tools/g17tensorcommonruntime.manifest_for, which now calls it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from agxforge.kernels import numerics
from agxforge.kernels.emit import CARRIER

F32 = np.float32
GENERIC_NAME = "tensor_gemm_generic_runtime_demo"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generic_manifest(program, s):
    """The manifest of a gemm_generic program whose transport is `s` (generic_view's dict): the image contract
    with the grid, the buffer shapes and the ABI, the image hashes left zero for `author` to fill."""
    from agxforge.g17 import runtime

    abi = program.abi_plain(program.abi())
    contract = program.contract()
    width = max([s["N"]] + [st[0] for st in s["stages"]])         # the C buffer's width
    # P3 split-K: the launch has split_k more threadgroups (each writes an M x N partial), and the C
    # buffer stacks them: (split_k*M) x N. The reduce folds them; the partials are what this dispatch
    # writes and what generic_reference returns.
    tensor = dict(M=s["M"], N=width, K=s["K"], lda=s["K"], ldb=s["N"], ldc=width,
                  a_type=s["a"], b_type=s["b"], c_type="int" if s["a"] == "int8" else "float",
                  simdgroups=s["simdgroups"],
                  grid=[32 * s["simdgroups"] * s["threadgroups"] * s["grid_n"] * s["split_k"], 1, 1],
                  threadgroup=[32 * s["simdgroups"], 1, 1], grid_n=s["grid_n"],
                  split_k=s["split_k"],
                  composition="attention" if s.get("attention") else "gemm_generic",
                  epilogue=tuple(s["epilogue"]) or None,
                  stages=tuple(tuple(st) for st in s["stages"]) or None,
                  accumulate=True if s["accumulate"] else None, saturate=True if s["saturate"] else None,
                  transA=True if s.get("transA") else None, transB=True if s.get("transB") else None)
    shape = {"rows": s["M"] * s["split_k"], "columns": width}
    tensor["grid"] = tuple(tensor["grid"])
    tensor["threadgroup"] = tuple(tensor["threadgroup"])
    return runtime.ImageContract(
        format=runtime.manifest_format(runtime.TENSOR_KIND), kind=runtime.TENSOR_KIND,
        name=contract.name, shape=runtime.Shape(**shape), tensor=runtime.TensorSpec(**tensor),
        abi=runtime.compiler_abi_from_plain(abi), code_size=len(program.code),
        instructions=tuple(runtime.Instruction(offset=i.offset, length=i.length, opcode=i.opcode)
                           for i in contract.instructions),
        sha256={"archive": "0" * 64, "library": "0" * 64,
                "object": "0" * 64, "code": sha(program.code)},
        field_ledger={"metadata": "compiler and measured tensor class"})

def carrier_tiles(lay, seed=4242):
    """The carrier's A tile (16 G x 16 halves) and B tile (16 x 16), drawn."""
    rng = np.random.default_rng(seed)
    ca = rng.uniform(-2, 2, (CARRIER * lay["groups"], CARRIER)).astype(np.float16)
    cb = rng.uniform(-2, 2, (CARRIER, CARRIER)).astype(np.float16)
    return ca, cb


def carrier_reference(lay, ca, cb):
    """Threadgroup t's rows [16 t, 16 t + 16) of A times B: one MMA over the whole tile."""
    return numerics.gemm(ca.astype(F32), cb.astype(F32))


def _place(buf, off, arr):
    raw = np.ascontiguousarray(arr).tobytes()
    buf[off:off + len(raw)] = raw


def _buffers(lay):
    return bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"]), bytearray(lay["c_bytes"])


def _with_carrier(lay, bufs):
    a, b, c = bufs
    ca, cb = carrier_tiles(lay)
    _place(a, 0, ca)
    _place(b, 0, cb)
    return ca, cb


def generic_view(lay):
    """What manifest_for reads for gemm_generic's transport. A layout may carry its own `view` (a K-loop
    GEMM whose buffers are large enough; tools/g17qmv.py), else the default below."""
    if "view" in lay:
        return dict(lay["view"])
    return dict(M=lay["M"], N=lay["N"], K=lay["K"], a="half", b="half", simdgroups=1, threadgroups=lay["groups"],
                grid_n=1, split_k=1,                   # Set C's column and K grids (25.134): neither is used here
                epilogue=[], stages=[], accumulate=False, saturate=False)


def author(bundle: Path, lay, program, a, b, c, extra=None):
    """Write a bundle the common worker loads: the image, its manifest, the three inputs."""
    from agxforge.g17 import scanlink
    if program.name != GENERIC_NAME:
        # manifest_for describes a program by its NAME, and a name it does not know falls through to the 17 x 19
        # demo's tensor spec: the manifest would describe a different program. Every decode kernel is gemm_generic.
        raise ValueError("author: %s is not a gemm_generic program, so no manifest describes it" % program.name)
    bundle = Path(bundle)
    if bundle.exists():
        raise ValueError("refusing to overwrite an existing bundle: %s" % bundle)
    if (len(a), len(b), len(c)) != (lay["a_bytes"], lay["b_bytes"], lay["c_bytes"]):
        raise ValueError("inputs do not match the transport")
    image = scanlink.author(program)
    manifest = generic_manifest(program, generic_view(lay)).model_copy(update={
        "sha256": {"archive": sha(image.archive), "library": sha(image.library),
                   "object": sha(image.object), "code": sha(program.code)},
        "field_ledger": image.field_ledger})
    expected = [(bd.index, bd.offset, bd.written) for bd in manifest.abi.bindings]
    if scanlink.verify_contract(image.archive, image.library, expected) != image.object:
        raise ValueError("authored archive does not contain its delivered object")
    bundle.mkdir(parents=True)
    (bundle / "decodeop.json").write_text(json.dumps(dict(layout=lay, **(extra or {})), indent=1, sort_keys=True) + "\n")
    for name, data in (("a.f16", a), ("b.f16", b), ("c.f32", c)):
        (bundle / name).write_bytes(data)
    (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    for name, data in (("scan.arc.metallib", image.archive), ("scan.lib.metallib", image.library),
                       ("scan.o", image.object), ("program.bin", program.code)):
        (bundle / name).write_bytes(data)
    return {"program_sha256": sha(program.code), "code_bytes": len(program.code),
            "manifest_sha256": sha((bundle / "manifest.json").read_bytes())}
