"""Which form of each kernel a model's graph uses, and its layout: the catalogue the deliverer builds from.

For each role of a decoder layer the catalogue gives the kernel's layout (where every input and output sits in the
three buffers) for a variant named in the model config, so `qmv.build_qmv2(qmv_layout(4, "qkv", {"ks": 2, ...})[0])`
is the q4 fused qkv projection in split-K form:

    qmv_layout        the four projections (qkv, wo + residual, w1 + w3 + SwiGLU, w2 + residual), by variant: split-K
                      (ks), hi16 scale loads (lean), loop-carried addresses (ptr), the position chains (chain)
    head_layout       the vocabulary projection, optionally with the argmax's pass 1 fused (argmax) or batched
    norm_layout       the wide RMSNorm, fp32 or fp16 out, seeded or once-rounded rsqrt
    attn_layout       the fused decode attention; attn_variant names a flag string's variant
    qmv_batch_layout, norm_batch_layout, argmax_rows_layout    the batched and multi-row forms

ARCH is the model's shapes (ARCHS: InternLM2.5-1.8B, Qwen3-0.6B, Qwen3-8B); tools/g17deliver.set_arch selects one.
KSPLIT is the split-K per projection, HEAD_VARIANTS the head forms built per width, BASE the v2 projection's fixed form.

Moved from tools/g17deliver.py, which builds each kernel with its verification inputs, dispatches it once on the GPU
against its reference, and delivers it into the index (agxforge.inference.index).
"""
from __future__ import annotations

from agxforge.kernels import attention, norm, qmv

# the v2 fp32-x qmv (MM 25.139.7 / 25.141.9): xvec, wpt 2 vector loads, hoisted constants
BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, wpt=2, vload=True, hoist_consts=True)


# THE MODEL'S SHAPES (MM 25.182): d_model, the attention output width (heads x head_dim, wo's K), the fused qkv width, the
# FFN width, the vocabulary (the argmax's padded width and lanes), and the norm eps. InternLM2.5-1.8B is the default and
# every bundle it builds is unchanged; `build --arch qwen3` selects Qwen3-0.6B's (tools/g17qwen3.py). The attention
# (16 q heads, 8 KV heads of 128) is the same in both, so only the single-sequence projections, norms, head and
# generation step read this table.
ARCHS = {"internlm2": dict(d=2048, hd=2048, qkv=4096, ffn=8192, vocab=92544, vocab_pad=92544, per_lane=12, eps=1e-5),
         "qwen3": dict(d=1024, hd=2048, qkv=4096, ffn=3072, ffn_prefill=4096, vocab=151936, vocab_pad=152064, per_lane=24, eps=1e-6,
                       ksplit={4: dict(w2_res2=2)}),     # K 3,072 is 6 trips: split-K 4 does not divide them
         # Qwen3-8B: 32 query heads against 8 KV heads (GQA 4), the attention output 32 x 128 = 4,096 (wo's K)
         "qwen3_8b": dict(d=4096, hd=4096, qkv=6144, ffn=12288, ffn_prefill=16384, vocab=151936, vocab_pad=152064,
                          per_lane=24, eps=1e-6, heads=32, kv_heads=8,
                          qsm_sk_occ={"qkv": 1, "wo": 8, "w1": 1, "w3": 1, "w2": 8, "head": 1},
                          # the verify step's qsm head reads the decode head's weights in place, and the driver takes the
                          # argmax on the host over the true vocabulary: N = 151,936 (the decode head's W / S / B)
                          qsm_head_vocab=True)}


ARCH = dict(ARCHS["internlm2"], name="internlm2")


# split-K per projection and width (MM 25.141.15): rows 1, S simdgroups per threadgroup
KSPLIT = {4: dict(qkv=2, wo_res1=2, w2_res2=4, ffn=2), 8: dict(qkv=4, wo_res1=4, w2_res2=4, ffn=2)}


_KSPLIT0 = {k: dict(v) for k, v in KSPLIT.items()}


# the unsplit v2 rows per simdgroup (the q4/q8 v2 packages)
V2_ROWS = dict(qkv=4, wo_res1=4, w2_res2=4, ffn=2)


# lm_head variants per width: {} the delivered form; split-K + lean (MM 25.144.7). q8's "x32" head reads fp32 x from
# the out32 final norm, which the fp32-x lean/ptr path needs.
# ks 2 + lean without ptr (q8 x32) is the batched head's one-sequence twin: a batched graph's single-sequence check
# runs it (g17modelbuild.single_of drops "batch").
HEAD_VARIANTS = {4: ({}, dict(ks=2, lean=True), dict(ks=2, lean=True, ptr=True), dict(ks=2, lean=True, ptr=True, argmax=True)),
                 8: ({}, dict(ks=4, lean=True), dict(ks=2, lean=True, x32=True), dict(ks=4, lean=True, ptr=True, x32=True),
                     dict(ks=2, lean=True, ptr=True, x32=True), dict(ks=2, lean=True, ptr=True, x32=True, argmax=True),
                     dict(ks=2, lean=True, ptr=True, x32=True, w4=True))}


# ------------------------------------------------------------------------------------------------ layouts
def qmv_layout(bits, role, variant):
    """The layout of one projection: variant {} = the v2 form, {"ks": S} = split-K, {"ks": S, "lean": True} = split-K
    with hi16 scale loads (the form the graph was validated with), {"ks": S, "lean": True, "ptr": True} = that plus
    loop-carried addresses. Returns (layout, flags)."""
    ks = variant.get("ks")
    rows = 1 if ks else V2_ROWS[role]
    extra = dict(sgs=ks, ksplit=True, coop=True) if ks else {}
    flags = []
    if variant.get("lean"):
        extra["hi16_scales"] = True
        flags.append("hi16_scales")
    if role == "ffn":
        lay = dict(qmv.qmv_swiglu_layout(ARCH["ffn"], ARCH["d"], bits, rows), **BASE, act32=True, **extra)
    elif role == "qkv":
        lay = dict(qmv.case(ARCH["qkv"], ARCH["d"], bits, rows, nocarrier=True)[0], **BASE, **extra)
    elif role == "wo_res1":
        lay = qmv.with_residual(dict(qmv.case(ARCH["d"], ARCH["hd"], bits, rows, nocarrier=True)[0], **BASE, **extra), "add16")
    elif role == "w2_res2":
        lay = qmv.with_residual(dict(qmv.case(ARCH["d"], ARCH["ffn"], bits, rows, nocarrier=True)[0],
                                   **dict(BASE, wpt=4 if bits == 8 else 2), **extra), "add32_to16")
    else:
        raise ValueError("unknown projection role %r" % role)
    if variant.get("nodeq"):
        lay = dict(lay, ablate_nodeq=True)         # MM 25.200: the timing ablation (loads kept, no dequant arithmetic)
        flags.append("ablate_nodeq")
    if variant.get("ptr"):
        lay = dict(lay, ptr_addr=True)             # loop-carried addresses (MM 25.141.18); refused outside its shape
        flags.append("ptr_addr")
    if variant.get("chain"):
        # the r1 loop without its per-row extras (MM 25.144.4): position chains against raw x, fused epilogue, op428
        # pool masks, and16 read in place after the load's first waited consumer. Its own fp32 order.
        ch = dict(chains=True, epi_fma=True, and16_direct=True, pool_masks=True)
        lay = dict(lay, interleave=False, hoist_consts=False)      # the timed form (55 -> 51 registers)
        lay = dict(lay, **ch)
        flags += sorted(ch)
    return lay, flags


def head_layout(bits, variant=None):
    """The lm_head (92,544 x 2048). variant {} = the delivered form: q4 the xvec fp32-x form (wpt 2, rows 4), q8 the
    fp16-x (x16) form, wpt 2. {"ks": S, ...} = split-K, rows 1, S simdgroups (the cooperative class), with "lean"
    (hi16 scale loads) and "ptr" (loop-carried addresses, fp32 x only). "x32" gives q8 an fp32-x (xvec) head, which
    reads the out32 final norm (MM 25.144.7). Returns (layout, variant)."""
    variant = dict(variant or {})
    ks = variant.get("ks")
    rows = 1 if ks else 4
    fp32_x = bits == 4 or variant.get("x32")
    if fp32_x:
        lay = dict(qmv.case(ARCH["vocab"], ARCH["d"], bits, rows, nocarrier=True)[0], **dict(BASE, wpt=4 if variant.get("w4") else 2))
    else:
        lay = dict(qmv.case(92544, 2048, 8, rows, nocarrier=True)[0], interleave=True, lean=True, coalesced=True, a16=True,
                   wpt=2, x16=True)
    if not variant:
        return lay, ({"xvec": True} if bits == 4 else {"x16": True})
    if ks:
        lay.update(sgs=ks, ksplit=True, coop=True)
    if variant.get("lean"):
        lay["hi16_scales"] = True
    if variant.get("ptr"):
        if not fp32_x:
            raise ValueError("lm_head ptr: loop-carried addresses need the fp32-x (xvec) head")
        lay["ptr_addr"] = True
    if variant.get("argmax"):
        lay = qmv.with_argmax_chunks(lay)          # the head writes g17gen's pass-1 pairs itself (MM 25.144.7)
    if variant.get("batch"):
        # THE BATCHED HEAD (MM 25.144.7): one dispatch, each weight word read once for nb sequences (M3's multi-vector
        # qmv, build_qmv2_batch). Vector-major: x_b fp32 at X + 4 K b (the batched final norm's rows), logits_b at
        # OUT + 4 V b (where the batched argmax reads them). fp32 x only, and no ptr_addr (outside the batch scope).
        if not fp32_x or variant.get("ptr") or variant.get("argmax"):
            raise ValueError("lm_head batch: the fp32-x split-K lean head without ptr / argmax")
        if variant.get("pass"):
            lay["batch_pass"] = variant["pass"]
        lay = qmv.with_batch(lay, variant["batch"])
    return lay, variant


def norm_layout(in_dtype, out32, seed):
    lay = norm.rmsnorm_loop_layout(ARCH["d"], in_dtype, groups=32, unroll=16 if in_dtype == "half" else 8, hoist=True)
    if out32:
        lay["out32"] = True
    lay["rs_seed" if seed else "rs_once"] = True
    return lay


def attn_layout(cap, flags=""):
    """The delivered wide butterfly attention; `flags` ("keyblock=2+tgsplit=4", ...) applies the M5 long-context forms
    (g17attn.with_keyblock / with_nsum / with_tgsplit, MM 25.144.5) in order. "" is the base form."""
    lay = attention.attn_rope_layout(cap=cap, heads=ARCH.get("heads", 16), kv_heads=ARCH.get("kv_heads", 8))
    lay = attention.with_attn32(attention.with_bfly_merge(attention.with_wide(attention.with_fused_merge(
        attention.with_rope_tables(lay)))))
    for f in (x for x in flags.split("+") if x):
        k, _, v = f.partition("=")
        lay = getattr(attention, "with_" + k)(lay, *([int(v)] if v else []))
    return lay


def attn_variant(flags):
    """The index entry's variant dict for an attention flag string."""
    v = dict(widebf=True, attn32=True)
    for f in (x for x in flags.split("+") if x):
        k, _, n = f.partition("=")
        v[k] = int(n) if n else True
    return v


def qmv_batch_layout(bits, role, nb, pv=None, dq=False, wide=False, accsplit=1):
    """The multi-vector qmv (MM 25.144.3): the lean split-K form of the role with with_batch(nb). Constants are hoisted
    unless the batched program then runs out of registers (nb = 8 for the FFN and q4 w2), when they stay in the loop.
    pv: vectors per pass (batch_pass), the weight stream read nb / pv times with only pv vectors live at once."""
    lay, flags = qmv_layout(bits, role, dict(ks=KSPLIT[bits][role], lean=True))
    if wide:
        # MLX qmv_wide (MM 25.144.9): 8 K-lanes per row, 4 rows per simdgroup, 2 simdgroups; same X/W/S/B/OUT and
        # vector-major strides as the split-K form, so it drops into the graph. Plain qmv only for now.
        if role != "qkv":
            raise ValueError("qmv wide: qkv only for now (no residual/swiglu wide kernel yet)")
        lay = qmv.with_batch(dict(lay), nb) if nb > 1 else dict(lay, batch=1)
        lay = dict(lay, wide=True, klanes=8, rows_per_sg=4, wide_sgs=1, wide_pass=min(nb, pv or 4))
        rows_tg = lay["rows_per_sg"] * lay["wide_sgs"]
        lay["groups"] = (lay["Nout"] // rows_tg) * (nb // lay["wide_pass"])
        return lay, flags + ["wide"], qmv.build_qmv2(lay)
    if pv:
        lay = dict(lay, batch_pass=pv)
        flags = flags + ["batch_pass"]
    if dq:
        # dequantize once per weight, reused by every vector of the pass as a plain fp32 dot (its own order)
        lay = dict(lay, dequant_once=True)
        flags = flags + ["dequant_once"]
    if accsplit > 1:
        # split each (vector, row) accumulator into `accsplit` independent partials: 1/accsplit the critical chain
        lay = dict(lay, acc_split=accsplit)
        flags = flags + ["acc_split%d" % accsplit]
    for hoist in (True, False):
        # nb 1 (dq only): the single-vector dequantize-once kernel, the single layout
        cand = qmv.with_batch(dict(lay, hoist_consts=hoist), nb) if nb > 1 else dict(lay, hoist_consts=hoist)
        try:
            prog = qmv.build_qmv2(cand)
        except Exception:
            if not hoist:
                raise
            continue
        return cand, flags + ["batch"] + ([] if hoist else ["no_hoist"]), prog


def norm_batch_layout(in_dtype, out32, seed, nb):
    """The wide RMSNorm over nb rows in ONE dispatch (MM 25.144.3): nb threadgroups of 1024, row b's x at X + b d
    elements and its out at OUT + b d elements (vector-major); the gain is shared."""
    lay = dict(norm_layout(in_dtype, out32, seed), batch=nb)
    d = lay["d"]
    xs, os_ = (2 if in_dtype == "half" else 4), (4 if out32 else 2)
    lay["a_bytes"] = max(lay["a_bytes"], qmv._align(lay["X"] + xs * d * nb))
    lay["c_bytes"] = max(lay["c_bytes"], qmv._align(lay["OUT"] + os_ * d * nb))
    return lay


def argmax_rows_layout(V, rows=16, per_lane=4):
    """MM 25.205: g17gen's argmax pass 1 over `rows` contiguous logit rows (the speculative verify step's 16), its chunk
    C = 32 per_lane dividing the vocabulary (Qwen3's 151,936 = 1,187 x 128). Pass 2's 256-pair cap does not apply: the
    driver reduces each row's G pairs. Indices are global (below 2^23, exact floats); row r's are r V + v."""
    C = 32 * per_lane
    if V % C or rows * V >= (1 << 23):
        raise ValueError("argmax rows: V a multiple of %d, rows x V below 2^23" % C)
    G = V // C
    return dict(op="argmax", V=V, per_lane=per_lane, C=C, G=G, PAIRS=0, TOK=0, rows=rows, batched=True,
                pairs_bytes=8 * G * rows, logits_bytes=4 * V * rows)
