"""Inspectable scalar kernels for the first G13 model route.

All arithmetic, reductions, projections, RoPE, softmax and argmax are GPU
programs. No tensor instructions, hidden qmm route, atomics, spills or barriers.
IR construction and target compilation are separate responsibilities.
"""
from dataclasses import dataclass
import math
import struct
from . import ir
from . import cc

CAPABILITIES = frozenset({"scalar", "integer", "bounded_loops", "fp16", "bf16_storage", "approx_math"})


@dataclass
class Kernel:
    function: ir.Function
    threads: int
    extents: dict
    requirements: frozenset = frozenset({"scalar"})

    def required_capabilities(self):
        required = set(self.requirements)
        kinds = {o.kind for block in self.function.blocks for o in block.ops}
        if kinds & {"add", "sub", "mul", "madd", "and", "or", "xor", "shl", "shr", "csel", "icmp"}:
            required.add("integer")
        if kinds & {"phi", "br_cond"}: required.add("bounded_loops")
        if kinds & {"rcp", "rsqrt", "exp2", "log2", "sin_turns"}: required.add("approx_math")
        if kinds & {"f16_to_f32", "f32_to_f16_rte"} or any(b.elem == ir.F16 for b in self.function.buffers):
            required.add("fp16")
        if any(b.elem == "bfloat" for b in self.function.buffers): required.add("bf16_storage")
        return frozenset(required)

    def compile(self, capabilities=CAPABILITIES):
        missing = self.required_capabilities() - capabilities
        if missing: raise cc.Unsupported(f"{self.function.name} requires {sorted(missing)}")
        return cc.compile_function(self.function, threads=self.threads, extents=self.extents)


def setup(name, sizes, threads, dtypes=None, requirements=()):
    dtypes = dtypes or {}
    buffers = {n: ir.Buffer(n, i, dtypes.get(n, ir.F32)) for i, n in enumerate(sizes)}
    fn = ir.Function(name, buffers.values())
    b = ir.Builder(fn, fn.block("entry"))
    t = b.builtin("thread_position_in_grid")
    return Kernel(fn, threads, sizes, frozenset({"scalar", *requirements})), b, t, buffers


def cf(b, number):
    return b.const(struct.unpack("<I", struct.pack("<f", number))[0])


def load_float(b, buf, index):
    if buf.elem == "bfloat":
        return b.shl(b.load(buf, index, width="half"), ir.Imm(16), type=ir.I32)
    if buf.elem == ir.F16:
        return b.f16_to_f32(b.load(buf, index, width="half"))
    return b.load(buf, index)


def loop(b, bound, seeds, body, tag="loop", cap=None):
    """Counted do-while; body receives (counter, carried values) and returns next values."""
    zero = b.const(0)
    header = b.fn.block(tag); after = b.fn.block(tag + "_done")
    b.br(header); b.at(header)
    index = b.phi(zero)
    carried = [b.phi(s) for s in seeds]
    updated = body(index, carried)
    inc = b.add(index, ir.Imm(1)); b.phi_latch(index, inc)
    for phi, value in zip(carried, updated): b.phi_latch(phi, value)
    pred = b.cmp(inc, bound, cap=cap) if isinstance(bound, ir.Value) else b.cmp(inc, bound)
    b.br_cond(pred, header, after); b.at(after)
    return carried


def vector(operation, n):
    spec, b, t, p = setup(operation, {"out": n * 4, "a": n * 4, "b": n * 4}, n,
                            requirements=("approx_math",) if operation == "swiglu" else ())
    a, x = b.load(p["a"], t), b.load(p["b"], t)
    if operation == "copy": value = a
    elif operation == "add": value = b.fadd(a, x)
    elif operation == "mul": value = b.fmul(a, x)
    elif operation == "sub": value = b.fsub(a, x)
    elif operation == "fma": value = b.fma(a, x, cf(b, -1))
    elif operation == "swiglu":
        e = b.exp2(b.fmul(a, cf(b, -math.log2(math.e))))
        value = b.fmul(b.fmul(a, b.rcp(b.fadd(cf(b, 1), e))), x)
    else: raise ValueError(operation)
    b.store_at(p["out"], t, value); b.ret()
    return spec


def embedding(vocab, d, dtype="bfloat"):
    spec, b, t, p = setup("embedding", {"out": d*4, "weights": vocab*d*ir.ELEM_BYTES[dtype], "params": 8}, d,
                            {"weights": dtype, "params": ir.I32}, ("integer", "bf16_storage"))
    token = b.load(p["params"], ir.Imm(0))
    index = b.add(b.mul(token, b.const(d)), t)
    b.store_at(p["out"], t, load_float(b, p["weights"], index)); b.ret()
    return spec


def gemv(rows, cols, dtype="bfloat", bias=False):
    sizes = {"out": rows*4, "x": cols*4, "weights": rows*cols*ir.ELEM_BYTES[dtype]}
    if bias: sizes["bias"] = rows * ir.ELEM_BYTES[dtype]
    spec, b, t, p = setup(f"gemv_{rows}_{cols}_{dtype}_{bias}", sizes, rows,
                            {"weights": dtype, "bias": dtype}, ("integer", "bounded_loops"))
    base = b.mul(t, b.const(cols)); zero = cf(b, 0)
    def body(i, carried):
        w = load_float(b, p["weights"], b.add(base, i))
        x = b.load(p["x"], i)
        return [b.fma(w, x, carried[0])]
    total, = loop(b, cols, [zero], body)
    if bias: total = b.fadd(total, load_float(b, p["bias"], t))
    b.store_at(p["out"], t, total); b.ret()
    return spec


def qmv4(rows, cols):
    if cols % 8: raise ValueError("q4 rows require a multiple of eight columns")
    sizes = {"out": rows*4, "x": cols*4, "weights": rows*cols//2, "scale": rows*4, "offset": rows*4}
    spec, b, t, p = setup("qmv4", sizes, rows, {"weights": ir.I32}, ("integer", "bounded_loops"))
    base = b.mul(t, b.const(cols//8)); zero = cf(b, 0)
    scale, offset = b.load(p["scale"], t), b.load(p["offset"], t)
    def body(i, carried):
        word = b.load(p["weights"], b.add(base, b.shr(i, ir.Imm(3))))
        shift = b.mul(getattr(b, "and")(i, ir.Imm(7)), ir.Imm(4))
        q = getattr(b, "and")(b.shr(word, shift), ir.Imm(15))
        w = b.fadd(b.fmul(b.u32_to_f32(q), scale), offset)
        return [b.fma(w, b.load(p["x"], i), carried[0])]
    total, = loop(b, cols, [zero], body)
    b.store_at(p["out"], t, total); b.ret()
    return spec


def reduce_rows(rows, cols, operation="sum"):
    spec, b, t, p = setup(f"reduce_{operation}_{rows}_{cols}", {"out": rows*4, "x": rows*cols*4}, rows,
                            requirements=("bounded_loops", "integer"))
    base = b.mul(t, b.const(cols)); seed = cf(b, -math.inf if operation == "max" else 0)
    def body(i, carried):
        x = b.load(p["x"], b.add(base, i))
        if operation == "sum": v = b.fadd(carried[0], x)
        elif operation == "squares": v = b.fma(x, x, carried[0])
        elif operation == "max": v = b._def("fmax", [carried[0], x])
        else: raise ValueError(operation)
        return [v]
    total, = loop(b, cols, [seed], body)
    b.store_at(p["out"], t, total); b.ret()
    return spec


def norm_scale(d, eps):
    spec, b, t, p = setup("rms_scale", {"out": 4, "sum": 4}, 1, requirements=("approx_math",))
    x = b.load(p["sum"], ir.Imm(0))
    value = b.rsqrt(b.fadd(b.fmul(x, cf(b, 1/d)), cf(b, eps)))
    b.store_at(p["out"], t, value); b.ret()
    return spec


def norm_apply(d, dtype="bfloat"):
    spec, b, t, p = setup("rms_apply", {"out": d*4, "x": d*4, "gain": d*ir.ELEM_BYTES[dtype], "scale": 4}, d,
                            {"gain": dtype}, ("bf16_storage", "integer"))
    x, g, s = b.load(p["x"], t), load_float(b, p["gain"], t), b.load(p["scale"], ir.Imm(0))
    b.store_at(p["out"], t, b.fmul(b.fmul(x, s), g)); b.ret()
    return spec


def rope(heads, hd):
    if hd & (hd-1) or hd < 2: raise ValueError("RoPE head dimension must be a power of two")
    n, half = heads*hd, hd//2
    spec, b, t, p = setup("rope", {"out": n*4, "x": n*4, "invfreq": half*4, "params": 8}, n,
                            {"params": ir.I32}, ("integer", "approx_math"))
    comp = getattr(b, "and")(t, b.const(hd-1))
    base = getattr(b, "and")(t, b.const((~(hd-1)) & 0xffffffff))
    j = getattr(b, "and")(comp, b.const(half-1))
    first = b.load(p["x"], b.add(base, j))
    second = b.load(p["x"], b.add(b.add(base, j), b.const(half)))
    pos = b.u32_to_f32(b.load(p["params"], ir.Imm(1)))
    turns = b.fmul(b.fmul(pos, b.load(p["invfreq"], j)), cf(b, 1/(2*math.pi)))
    sin = b._def("sin_turns", [turns]); cos = b._def("sin_turns", [b.fadd(turns, cf(b, 0.25))])
    lo = b.fsub(b.fmul(first, cos), b.fmul(second, sin))
    hi = b.fadd(b.fmul(second, cos), b.fmul(first, sin))
    value = b.csel(comp, b.const(half-1), hi, lo, rel="gt")
    b.store_at(p["out"], t, value); b.ret()
    return spec


def kv_write(width, capacity):
    spec, b, t, p = setup("kv_write", {"cache": capacity*width*4, "x": width*4, "params": 8}, width,
                            {"params": ir.I32}, ("integer",))
    pos = b.load(p["params"], ir.Imm(1))
    index = b.add(b.mul(pos, b.const(width)), t)
    b.store_at(p["cache"], index, b.load(p["x"], t)); b.ret()
    return spec


def scores(heads, kv_heads, hd, capacity):
    if capacity & (capacity-1): raise ValueError("attention capacity must be a power of two")
    spec, b, t, p = setup("attention_scores", {"out": heads*capacity*4, "q": heads*hd*4,
        "k": capacity*kv_heads*hd*4, "headmap": heads*4, "params": 8}, heads*capacity,
        {"headmap": ir.I32, "params": ir.I32}, ("integer", "bounded_loops"))
    head = b.shr(t, b.const(capacity.bit_length()-1)); token = getattr(b, "and")(t, b.const(capacity-1))
    kvhead = b.load(p["headmap"], head)
    qb = b.mul(head, b.const(hd))
    kb = b.mul(b.add(b.mul(token, b.const(kv_heads)), kvhead), b.const(hd))
    def body(i, carried):
        return [b.fma(b.load(p["q"], b.add(qb, i)), b.load(p["k"], b.add(kb, i)), carried[0])]
    total, = loop(b, hd, [cf(b, 0)], body)
    scaled = b.fmul(total, cf(b, 1/math.sqrt(hd)))
    pos = b.load(p["params"], ir.Imm(1))
    masked = b.csel(token, pos, cf(b, -math.inf), scaled, rel="gt")
    b.store_at(p["out"], t, masked); b.ret()
    return spec


def softmax_element(heads, capacity, stage):
    n = heads*capacity
    spec, b, t, p = setup("softmax_" + stage, {"out": n*4, "x": n*4, "row": heads*4}, n,
                            requirements=("integer", "approx_math"))
    head = b.shr(t, b.const(capacity.bit_length()-1)); x = b.load(p["x"], t); row = b.load(p["row"], head)
    if stage == "exp": value = b.exp2(b.fmul(b.fsub(x, row), cf(b, math.log2(math.e))))
    elif stage == "normalize": value = b.fmul(x, b.rcp(row))
    else: raise ValueError(stage)
    b.store_at(p["out"], t, value); b.ret()
    return spec


def attend(heads, kv_heads, hd, capacity):
    spec, b, t, p = setup("attention_values", {"out": heads*hd*4, "probs": heads*capacity*4,
        "v": capacity*kv_heads*hd*4, "headmap": heads*4}, heads*hd,
        {"headmap": ir.I32}, ("integer", "bounded_loops"))
    head = b.shr(t, b.const(hd.bit_length()-1)); comp = getattr(b, "and")(t, b.const(hd-1))
    kh = b.load(p["headmap"], head); pb = b.mul(head, b.const(capacity))
    vb = b.add(b.mul(kh, b.const(hd)), comp)
    def body(i, carried):
        prob = b.load(p["probs"], b.add(pb, i))
        value = b.load(p["v"], b.add(b.mul(i, b.const(kv_heads*hd)), vb))
        return [b.fma(prob, value, carried[0])]
    total, = loop(b, capacity, [cf(b, 0)], body)
    b.store_at(p["out"], t, total); b.ret()
    return spec


def argmax(n):
    spec, b, t, p = setup("argmax", {"out": 4, "x": n*4}, 1, {"out": ir.I32}, ("integer", "bounded_loops"))
    zero = b.const(0)
    def body(i, carried):
        best, index = carried
        value = b.load(p["x"], i)
        newindex = b._def("fcsel", [value, best, i, index], rel="gt")
        newbest = b._def("fmax", [value, best])
        return [newbest, newindex]
    _, index = loop(b, n, [cf(b, -math.inf), zero], body)
    b.store_at(p["out"], t, index); b.ret()
    return spec
