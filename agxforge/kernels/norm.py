"""RMSNorm for decode: one row, one wide threadgroup (MM 25.141.4).

    rmsnorm_loop_layout     the buffer layout (the carrier's tiles, then the row, the gain and the output)
    build_rmsnorm_wide      the kernel: thread t owns elements t, t + tpg, ...; its squares summed in order, the
                            lanes combined by the measured row and column butterflies, the simdgroup partials
                            through threadgroup memory, then mean, eps, the correctly rounded rsqrt and the scale
    rmsnorm_wide_reference  its value, in the same order (numerics); rmsnorm_wide_rows_reference, many rows at once

Moved from tools/g17decodeops.py.
"""
from __future__ import annotations

import numpy as np

from agxforge.kernels import numerics
from agxforge.kernels.emit import CARRIER, _align, _c, _carrier_bytes, _cf, _transport, emit_constants, emit_rn

F32 = np.float32

def rmsnorm_loop_layout(d, in_dtype, groups=1, unroll=8, hoist=False):
    """rmsnorm_layout for build_rmsnorm_loop: `groups` threadgroups each REDUNDANTLY form the whole
    sum of squares (the reference's order, lane l over i = 0..d/32-1, then the butterflies) and each
    scales and stores its own d/groups slice. `unroll` elements per lane per loop trip."""
    if d % 32:
        raise ValueError("rmsnorm: the row must fill 32 lanes")
    if in_dtype not in ("half", "float"):
        raise ValueError("rmsnorm: the input row is half or float")
    per = d // 32                                   # elements per lane in the sum pass
    if groups < 1 or per % groups:
        raise ValueError("rmsnorm_loop: %d elements per lane do not split over %d threadgroups" % (per, groups))
    slice_ = per // groups                          # elements per lane in the scale pass
    if unroll < 1 or per % unroll or slice_ % unroll and unroll % slice_:
        raise ValueError("rmsnorm_loop: unroll %d does not tile %d / %d elements per lane" % (unroll, per, slice_))
    # each buffer: the carrier's tile, then the row. The carrier's A tile is 16 G rows of 16 halves, its B
    # tile one 16 x 16 (so the weight row sits at 512 whatever G), its D tile 16 G rows of 16 floats
    X, G, OUT = _align(2 * CARRIER * CARRIER * groups), _align(2 * CARRIER * CARRIER), _carrier_bytes(groups)
    t = _transport(groups, X + (2 if in_dtype == "half" else 4) * d, G + 2 * d, OUT + 2 * d, N=128)
    return dict(op="rmsnorm_loop", d=d, in_dtype=in_dtype, groups=groups, unroll=unroll, X=X, G=G, OUT=OUT,
                **({"hoist": True} if hoist else {}), **t)


def build_rmsnorm_wide(lay, eps, tpg=1024):
    """RMSNorm as ONE threadgroup of `tpg` threads (a multiple of 32, d = 2 tpg or a multiple of tpg): thread t owns
    elements t, t + tpg, ...; its squares are summed in that order (the first a product, then fadd of each next
    product); the simdgroup's lanes combine with the row then column butterflies; lane 0 of simdgroup s stores the
    partial at threadgroup word s; after one barrier EVERY simdgroup reads the tpg/32 partials (lane l reads word l,
    lanes past tpg/32 read word 0 and are not summed... they read zero) and butterflies them again, so every lane
    holds the same total without a second barrier; then mean, eps, the corrected rsqrt and the scale, as
    build_rmsnorm. Offsets as build_rmsnorm_loop (x at X, gain at G, the fp16 row out at OUT), no carrier; the
    bindings are the cooperative three-binding class's (the harness binds [c, a, b] at 0, 1, 2)."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    d = lay["d"]
    if tpg % 32 or d % tpg or tpg > 1024:
        raise ValueError("rmsnorm_wide: tpg a multiple of 32, at most 1024, dividing d")
    per, nsg = d // tpg, tpg // 32
    # THE COOPERATIVE THREE-BINDING CLASS (MM 25.140.3): slot 0 written (the row out), slot 1 x, slot 2 the gain;
    # system registers 156 and 164 only, so the lane is t & 31 rather than its own register
    c = ir.Buffer("C", 0, elem=ir.F32); a = ir.Buffer("A", 1, elem=ir.F16); bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    fn.declare_threadgroup(33 if lay.get("rs_once") else 32, size=(tpg, 1, 1))
    b = ir.Builder(fn, fn.block("entry"))
    xunit = 2 if lay["in_dtype"] == "half" else 4
    t0 = b.builtin("thread_position_in_threadgroup", name="t0")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")          # 0; read so the class's registers match
    t = b.add(t0, b.mul(tg, _c(b, 0, "tgz"), name="tg0"), name="t")
    lane = getattr(b, "and")(t, ir.Imm(31), name="lane")
    sg = b.shr(t, _c(b, 5, "five"), name="sg")
    # BATCHED (lay["batch"], MM 25.144.3): threadgroup b normalizes row b - its x and its out are offset b d elements
    # (vector-major, the multi-vector qmv's layout); the gain is shared
    tr = b.add(t, b.mul(tg, _c(b, d, "rowd"), name="rowoff"), name="tr") if lay.get("batch", 1) > 1 else t

    def load_v(i, tag):
        idx = b.add(tr, _c(b, i * tpg + lay["X"] // xunit, tag + "_o"), name=tag + "_i")
        if lay["in_dtype"] == "half":
            return b.f16_to_f32(b.load(a, idx, width="half", name=tag + "_h"), name=tag)
        return b.load(a, idx, type=ir.I32, name=tag)
    vs = [load_v(i, "v%d" % i) for i in range(per)]
    acc = None
    for i, v in enumerate(vs):
        sq = b.fmul(v, v, type=ir.F32, name="sq%d" % i)
        acc = sq if acc is None else b.fadd(acc, sq, type=ir.F32, name="acc%d" % i)
    s = TR.emit_butterfly(b, acc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    s = TR.emit_butterfly(b, s, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    # lane 0 of each simdgroup publishes its partial
    pub, joined = fn.block("publish"), fn.block("published")
    b.br_cond(b.cmp(lane, 1, "lt", name="lane0"), pub, joined)
    b.at(pub)
    b.store_tg(b.fadd(s, _cf(b, F32(0.0), "pz"), type=ir.F32, name="part"), sg)
    b.br(joined)
    b.at(joined)
    b.barrier("threadgroup")
    # every simdgroup reads all partials: lane l < nsg reads word l, the rest contribute exact zeros
    word = b.csel(lane, _c(b, nsg - 1, "nsgm1"), _c(b, 0, "w0"), lane, rel="gt", name="word")
    p = b.load_tg(word, type=ir.I32, name="p")
    p = b.fadd(p, _cf(b, F32(0.0), "pz2"), type=ir.I32, name="pv")
    p = b.csel(lane, _c(b, nsg - 1, "nsgm1b"), _cf(b, F32(0.0), "zero"), p, rel="gt", name="pm")
    p = b.fadd(p, _cf(b, F32(0.0), "pz3"), type=ir.F32, name="pf")
    tot = TR.emit_butterfly(b, p, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    tot = TR.emit_butterfly(b, tot, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    mean = b.fmul(tot, _cf(b, F32(1.0 / d), "inv_d"), name="mean")
    if lay.get("rs_seed"):
        # THE HARDWARE SEED ITSELF (op3850, MM 25.141.16): an exact function, modelled by g17decodestep.rsqrt_seed,
        # so every thread forms r in one instruction with no region and no second barrier
        r = b.rsqrt(b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), type=ir.I32, name="rs_seed")
    elif lay.get("rs_once"):
        # THE CORRECTED RSQRT ONCE, by simdgroup 0 (about 110 of the program's 196 IR ops): it publishes r at word
        # 32, and after a second barrier every thread reads it. The other simdgroups SKIP the region (cc's skip
        # branch, 25.141.2) rather than walking it masked. Same arithmetic, so the same value.
        fn.skip_regions = True
        rsb, rsj = fn.block("rs_one"), fn.block("rs_done")
        b.br_cond(b.cmp(sg, 1, "lt", name="sg0"), rsb, rsj)
        b.at(rsb)
        K = emit_constants(b)
        r0 = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
        b.store_tg(b.fadd(r0, _cf(b, F32(0.0), "rz"), type=ir.F32, name="rpub"), _c(b, 32, "w32"))
        b.br(rsj)
        b.at(rsj)
        b.barrier("threadgroup")
        r = b.fadd(b.load_tg(_c(b, 32, "w32r"), type=ir.I32, name="r_ld"), _cf(b, F32(0.0), "rz2"), type=ir.I32, name="r")
    else:
        K = emit_constants(b)
        r = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
    # the scale pass as a COUNTED LOOP over the thread's elements (the class is witnessed above 31 instructions
    # only with a back edge, MM 25.140.4); element i is reloaded, so nothing is carried but the index
    hdr, post = fn.block("scale_loop"), fn.block("scale_done")
    e0 = t
    j0 = _c(b, 0, "j0")
    b.br(hdr)
    b.at(hdr)
    j = b.phi(j0, name="j")
    e = b.phi(e0, name="e")
    batched = lay.get("batch", 1) > 1
    er = b.add(e, b.mul(tg, _c(b, d, "rowd2"), name="rowoff2"), name="er") if batched else e
    xi = b.add(er, _c(b, lay["X"] // xunit, "xs_o"), name="xs_i")
    if lay["in_dtype"] == "half":
        v = b.f16_to_f32(b.load(a, xi, width="half", name="xs_h"), name="xs")
    else:
        v = b.fadd(b.load(a, xi, type=ir.I32, name="xs_l"), _cf(b, F32(0.0), "xs_z"), type=ir.I32, name="xs")
    g = b.f16_to_f32(b.load(bb, b.add(e, _c(b, lay["G"] // 2, "g_o"), name="g_i"), width="half", name="g_h"), name="g")
    y = b.fmul(b.fmul(v, r, name="vr"), g, name="y")
    if lay.get("out32"):
        # fp32 of the fp16-rounded row (exact widening): the next qmv reads it as xvec fp32 x
        b.store_at(c, b.add(er, _c(b, lay["OUT"] // 4, "o_o"), name="o_i"),
                   b.f16_to_f32(b.f32_to_f16_rte(y, name="yh"), name="yw"))
    else:
        b.store_at(c, b.add(er, _c(b, lay["OUT"] // 2, "o_o"), name="o_i"), b.f32_to_f16_rte(y, name="yh"), width="half")
    jn = b.add(j, ir.Imm(1), name="j_next")
    en = b.add(e, _c(b, tpg, "tpg"), name="e_next")
    ir.Builder.phi_latch(j, jn)
    ir.Builder.phi_latch(e, en)
    b.br_cond(b.cmp(jn, per, "lt", name="more"), hdr, post)
    b.at(post)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def rmsnorm_wide_rows_reference(V, g, spec, tpg=1024, seed=False):
    """rmsnorm_wide_reference of every row of V [R, d] at once: the same operations per row (the lane butterflies as
    tensorreduce.butterfly_array). A row whose butterfly values are not all finite is recomputed by
    rmsnorm_wide_reference itself, so overflow behaves as it does there. test_g17simspeed checks bit-identity."""
    from agxforge.g17 import tensorreduce as TR
    V = np.asarray(V, F32)
    R, d = V.shape
    per, nsg = d // tpg, tpg // 32
    sq = numerics.fmul(V, V).reshape(R, per, tpg)
    acc = sq[:, 0].copy()
    for i in range(1, per):
        acc = numerics.fadd(acc, sq[:, i])
    lanes = TR.butterfly_array(TR.butterfly_array(acc.reshape(R, nsg, 32), TR.ROW_BUTTERFLY_MASKS, "sum"),
                               TR.COLUMN_BUTTERFLY_MASKS, "sum")
    pl = np.zeros((R, 32), F32)
    pl[:, :nsg] = lanes[:, :, 0]
    tot = TR.butterfly_array(TR.butterfly_array(pl, TR.ROW_BUTTERFLY_MASKS, "sum"), TR.COLUMN_BUTTERFLY_MASKS, "sum")[:, 0]
    ok = np.isfinite(acc).all(1) & np.isfinite(lanes).all((1, 2)) & np.isfinite(tot)
    out = np.zeros(V.shape, F32)
    if ok.any():
        r = (numerics.rsqrt_seed if seed else numerics.rsqrt)(numerics.fadd(numerics.fmul(tot[ok].astype(F32), F32(1.0 / d)), F32(spec.norm_eps)))
        out[ok] = numerics.narrow(numerics.fmul(numerics.fmul(V[ok], r[:, None]), np.asarray(g, F32)[None, :]), spec.storage)
    for i in np.nonzero(~ok)[0]:
        out[i] = rmsnorm_wide_reference(V[i], g, spec, tpg=tpg, seed=seed)
    return out


def rmsnorm_wide_reference(v, g, spec, tpg=1024, seed=False):
    """build_rmsnorm_wide's value: per-thread ordered squares, the lane butterflies per simdgroup, then the
    butterflies over the partials (zeros past tpg/32), mean, eps, rsqrt, scale."""
    from agxforge.g17 import tensorreduce as TR
    v = np.asarray(v, F32); d = v.size; per, nsg = d // tpg, tpg // 32
    sq = numerics.fmul(v, v).reshape(per, tpg)
    acc = sq[0].copy()
    for i in range(1, per):
        acc = numerics.fadd(acc, sq[i])
    parts = []
    for s in range(nsg):
        lanes = TR.butterfly([float(x) for x in acc[32 * s:32 * s + 32]], TR.ROW_BUTTERFLY_MASKS, "sum")
        lanes = TR.butterfly(list(lanes), TR.COLUMN_BUTTERFLY_MASKS, "sum")
        parts.append(F32(lanes[0]))
    pl = [float(parts[l]) if l < nsg else 0.0 for l in range(32)]
    tot = TR.butterfly(pl, TR.ROW_BUTTERFLY_MASKS, "sum")
    tot = TR.butterfly(list(tot), TR.COLUMN_BUTTERFLY_MASKS, "sum")
    r = (numerics.rsqrt_seed if seed else numerics.rsqrt)(numerics.fadd(numerics.fmul(F32(tot[0]), F32(1.0 / d)), F32(spec.norm_eps)))
    return numerics.narrow(numerics.fmul(numerics.fmul(v, r), np.asarray(g, F32)), spec.storage)
