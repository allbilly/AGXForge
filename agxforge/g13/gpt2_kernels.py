"""Scalar G13 additions for GPT-2; checkpoint tensors stay in their original layout."""
import math
from . import ir
from .kernels import setup, cf, loop


def position_embedding(positions, d):
    spec, b, t, p = setup("position_embedding", {"out": d*4, "weights": positions*d*4, "params": 8}, d,
                          {"params": ir.I32})
    pos = b.load(p["params"], ir.Imm(1))
    index = b.add(b.mul(pos, b.const(d)), t)
    b.store_at(p["out"], t, b.load(p["weights"], index)); b.ret()
    return spec


def conv1d(inputs, outputs):
    """HF Conv1D: x @ weights[inputs, outputs] + bias, without CPU transposition."""
    spec, b, t, p = setup(f"conv1d_{inputs}_{outputs}",
        {"out": outputs*4, "x": inputs*4, "weights": inputs*outputs*4, "bias": outputs*4}, outputs)
    stride = b.const(outputs)
    def body(i, carried):
        index = b.add(b.mul(i, stride), t)
        return [b.fma(b.load(p["weights"], index), b.load(p["x"], i), carried[0])]
    total, = loop(b, inputs, [cf(b, 0)], body)
    b.store_at(p["out"], t, b.fadd(total, b.load(p["bias"], t))); b.ret()
    return spec


def split_qkv(d):
    spec, b, t, p = setup("split_qkv", {"q": d*4, "k": d*4, "v": d*4, "packed": 3*d*4}, d)
    for role, offset in (("q", 0), ("k", d), ("v", 2*d)):
        index = t if not offset else b.add(t, b.const(offset))
        b.store_at(p[role], t, b.load(p["packed"], index))
    b.ret()
    return spec


def layer_norm_stats(d, eps):
    """Two-pass population variance: sum is supplied by the existing reduction kernel."""
    spec, b, t, p = setup("layer_norm_stats", {"out": 8, "x": d*4, "sum": 4}, 1)
    mean = b.fmul(b.load(p["sum"], ir.Imm(0)), cf(b, 1/d))
    def body(i, carried):
        centered = b.fsub(b.load(p["x"], i), mean)
        return [b.fma(centered, centered, carried[0])]
    variance_sum, = loop(b, d, [cf(b, 0)], body)
    scale = b.rsqrt(b.fadd(b.fmul(variance_sum, cf(b, 1/d)), cf(b, eps)))
    b.store_at(p["out"], ir.Imm(0), mean)
    b.store_at(p["out"], ir.Imm(1), scale); b.ret()
    return spec


def layer_norm_apply(d):
    spec, b, t, p = setup("layer_norm_apply",
        {"out": d*4, "x": d*4, "gain": d*4, "bias": d*4, "stats": 8}, d)
    centered = b.fsub(b.load(p["x"], t), b.load(p["stats"], ir.Imm(0)))
    normalized = b.fmul(centered, b.load(p["stats"], ir.Imm(1)))
    value = b.fadd(b.fmul(normalized, b.load(p["gain"], t)), b.load(p["bias"], t))
    b.store_at(p["out"], t, value); b.ret()
    return spec


def gelu(n):
    """gelu_new: x * sigmoid(2*sqrt(2/pi)*(x + 0.044715*x**3))."""
    spec, b, t, p = setup("gelu_new", {"out": n*4, "x": n*4}, n)
    x = b.load(p["x"], t)
    cubic = b.fmul(b.fmul(x, x), x)
    inner = b.fma(cubic, cf(b, .044715), x)
    exponential = b.exp2(b.fmul(inner, cf(b, -2*math.sqrt(2/math.pi)*math.log2(math.e))))
    value = b.fmul(x, b.rcp(b.fadd(cf(b, 1), exponential)))
    b.store_at(p["out"], t, value); b.ret()
    return spec
