"""The deliver index: the contract between the deliverer and the graph assembler.

An index (DIR/index.json) is ONE flat list of entries. The assembler selects an entry by (kind, bits, role, variant,
cap) and reads its bundle (a path relative to the index), its launch (threadgroups, threads_per_group, base), its
slot_map and the layout offsets its kind's SCHEMA names; recipe (the builder and the full layout) is what a
compile-only check rebuilds the program from, and program_sha256 what it must match.

    COMMON, SCHEMA, validate    the fields every entry carries, and each kind's
    deliver                     copy a verified bundle to its content-addressed home, bundles/<name>-<sha256[:16]>,
                                and return its entry: the same sources give the same bytes and the same path
    jsonable                    a layout as the entry records it

Moved from tools/g17deliver.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil

# THE INDEX CONTRACT with the graph (Piece A's g17q4graph selects by (kind, bits, role, variant, cap)): every entry
# carries COMMON, plus its kind's layout offsets and the fields the earlier hand-built index files carried.
COMMON = ("kind", "bits", "role", "variant", "cap", "bundle", "name", "threadgroups", "threads_per_group", "base",
          "slot_map", "program_sha256", "recipe")


SCHEMA = {
    "qmv": COMMON + ("op", "x_dtype", "x_offset", "X", "W", "S", "B", "OUT", "RES", "out_dtype"),
    "lm_head": COMMON + ("op", "x_dtype", "x_offset", "X", "W", "S", "B", "OUT", "out_dtype"),
    # the multi-vector qmv (MM 25.144.3): the qmv fields plus the batch and the vector-major strides
    "qmv_batch": COMMON + ("op", "x_dtype", "x_offset", "X", "W", "S", "B", "OUT", "RES", "out_dtype", "batch", "x_stride",
                           "out_stride", "h_stride", "res_stride"),
    "norm": COMMON + ("op", "in_dtype", "out_dtype", "X", "G", "OUT"),
    "attn": COMMON + ("ATTN", "attn_bytes", "region3_bytes", "COST", "SINT", "rope_bytes", "KOFF", "VOFF"),
    "gen_argmax": COMMON + ("GEN", "LOG", "R_X16", "region_bytes", "PAIRS"),
    "gen_step": COMMON + ("GEN", "LOG", "R_X16", "region_bytes", "PAIRS"),
    # MM 25.205: the speculative verify step's 16-row argmax pass 1 (the driver reduces each row's G pairs)
    "argmax_rows": COMMON + ("V", "G", "C", "pairs_bytes"),
    "qmvw": COMMON + ("N", "K", "nb", "W", "S", "B", "out_layout"),
    # M2 (MM 25.144.2): the prefill append and attention, sharing decode's cache (KOFF/VOFF) and rope region (COST/SINT)
    "prefill_attn": COMMON + ("mmax", "threadgroups_per_row", "QKVROW", "Q16", "PATTN", "KOFF", "VOFF", "COST", "SINT",
                              "rope_bytes", "prefill_region3_bytes", "region3_bytes"),
    # prefill elementwise over M rows (g17rows, MM 25.144.3)
    "swiglu_rows": COMMON + ("rows", "N", "unroll", "GATE", "UP", "ACT"),
    # MM 25.172: the batched decode on the tensor units. qsm: slot 1 (physical) the q4 weight block (W, S, B byte
    # offsets inside it), slot 2 the batch's x fp16 [16][K] rows, slot 3 y fp32 [16][N] or the sk partials [sk][16][N];
    # psum: slot 1 the partials, slot 2 the residual input at IN, slot 3 the output at OUT
    "qsm": COMMON + ("N", "K", "sk", "W", "S", "B", "out_layout"),
    "psum": COMMON + ("N", "sk", "mode", "rows", "pstride", "IN", "OUT", "out_layout"),
    "residual_rows": COMMON + ("rows", "N", "unroll", "H", "H16"),
    "fold_rows": COMMON + ("rows", "N", "unroll", "P0", "P1", "X"),
    # M1 (MM 25.144.1): the quantized prefill GEMM as two dispatches. qmm_dequant writes W16[K][N] fp16 into its C
    # buffer (slot 3); qmm's B buffer (slot 2) IS that W16, its A (slot 1) is x fp16 [M][K] rows, its C (slot 3)
    # y fp32 [M][N] rows (ldc = N), or for split_k 2 the two K-half partials stacked [2M][N], folded p0 + p1.
    "qmm_dequant": COMMON + ("N", "K", "W", "S", "B", "OUT", "out_layout"),
    "qmm": COMMON + ("M", "N", "K", "split_k", "A_layout", "B_layout", "out_layout"),
    # MM 25.144.12: w3's qmm with the SwiGLU in its tail (g17swigluqmm): slot 1 x fp16 [M][K], slot 2 w3's W16 [K][N],
    # slot 3 one region: U fp32 [M][N] at U_OFF (scratch), w1's gate G fp32 [M][N] at G_OFF, act fp16 [M][N] at ACT_OFF
    "qmm_swiglu": COMMON + ("M", "N", "K", "G_OFF", "U_OFF", "ACT_OFF", "out_layout"),
    # the tensor-unit route (MM 25.144.2): one program per bucket (M, p0 block)
    "prefill_mma": COMMON + ("M", "p0", "NB", "QKVROW", "Q16", "QT", "VB", "SCR", "PATTN", "KOFF", "VOFF", "COST", "SINT",
                             "rope_bytes", "mma_region3_bytes", "region3_bytes", "preconditions", "contract_notes"),
}


def validate(entry):
    """The entry's missing contract fields (empty when it is complete)."""
    return [k for k in SCHEMA[entry["kind"]] if k not in entry]


def jsonable(lay):
    return json.loads(json.dumps(lay))


def deliver(job, out, root):
    """Copy a verified bundle to its content-addressed home and return its index entry."""
    sha = hashlib.sha256(job["prog"].code).hexdigest()
    name = job.get("deliver_as", job["name"])
    dest = root / "bundles" / ("%s-%s" % (name, sha[:16]))
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(job["dir"], dest)
    e = dict(job["entry"], bundle=str(dest.relative_to(out)), name=name, program_sha256=sha, sha256=sha,
             code_bytes=len(job["prog"].code))
    e.setdefault("bits", None)
    e.setdefault("cap", None)
    # how the bundle was checked: bit-exact against its reference unless its builder declares the numerical check it
    # passed instead (the hardware-exp2 attention and the fast SwiGLU are enclosure-checked: their hardware exp2 is not
    # reproducible on the CPU)
    e["verified"] = job.get("verified", "hardware, bit-exact over a 0x7f sentinel (tools/g17deliver.py)")
    return e
