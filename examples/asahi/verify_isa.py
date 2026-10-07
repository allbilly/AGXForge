#!/usr/bin/env python3
"""Native C1/C2/C3 numerical checks, including reuse and dynamic bounded loops."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from agxforge.g13 import kernels as k, ir
from agxforge.runtime.asahi import Executor, source_identity


def binary(op, n):
    spec, b, t, p = k.setup(op, {"out": n*4, "a": n*4, "b": n*4}, n)
    a, x = b.load(p["a"], t), b.load(p["b"], t)
    value = b._def(op, [a, x])
    b.store_at(p["out"], t, value); b.ret()
    return spec


def unary(op, n):
    spec, b, t, p = k.setup(op, {"out": n*4, "x": n*4}, n)
    x = b.load(p["x"], t)
    v = b._def(op, [x], type=ir.I16 if op == "f32_to_f16_rte" else ir.I32)
    if op == "f32_to_f16_rte": v = b.f16_to_f32(v)
    b.store_at(p["out"], t, v); b.ret()
    return spec


def bounded(n):
    spec, b, t, p = k.setup("dynamic_loop", {"out": n*4, "bound": n*4}, n)
    bound = b.load(p["bound"], t)
    result, = k.loop(b, bound, [b.const(0)], lambda i, values: [b.add(values[0], ir.Imm(1))], cap=7)
    b.store_at(p["out"], t, result); b.ret()
    return spec


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    (output/"source-sha256.json").write_text(json.dumps(identity, indent=2)+"\n")
    report = dict(status="RUNNING", scope="compiler ISA semantics and persistent buffer reuse", checks=[])
    try:
        with Executor(va_slot=4) as gpu:
            (output/"platform.json").write_text(json.dumps(gpu.platform(), indent=2)+"\n")
            def check(spec, data, expected, *, atol=0, rtol=0, classify_nan=False, reused=None):
                p = spec.compile()
                bufs = reused or [gpu.buffer(v.min_bytes, v.access, data.get(v.name)) for v in p.bindings]
                if reused:
                    for binding, buf in zip(p.bindings, bufs):
                        if binding.name in data: buf.write(data[binding.name])
                path = output/f"{len(report['checks']):03d}-{p.name}"
                launch = gpu.dispatch(p, bufs, spec.threads, evidence=path)
                expected = np.asarray(expected).reshape(-1)
                raw = bufs[0].read(); actual = np.frombuffer(raw, expected.dtype)
                if classify_nan:
                    nan = np.isnan(expected)
                    passed = np.all(np.isnan(actual[nan])) and actual[~nan].tobytes() == expected[~nan].tobytes()
                elif atol or rtol:
                    passed = np.all(np.isfinite(actual)) and np.all(np.abs(actual.astype(np.float64)-expected) <= atol+rtol*np.abs(expected))
                else: passed = raw == expected.tobytes()
                # Read-only inputs must be preserved as well as their surrounding canaries.
                passed = passed and all(buf.read() == data[binding.name] for binding, buf in zip(p.bindings, bufs)
                                        if binding.name in data and binding.access == "read")
                (path/"expected.bin").write_bytes(expected.tobytes())
                row = dict(name=p.name, status="PASS" if passed else "WRONG_OUTPUT", atol=atol, rtol=rtol,
                           nan_policy="classification" if classify_nan else "bits", reused_buffers=reused is not None,
                           fence_completed=launch["fence_completed"], canaries_intact=launch["canaries_intact"])
                (path/"check.json").write_text(json.dumps(row, indent=2)+"\n"); report["checks"].append(row)
                if not passed: gpu.poisoned=True; raise RuntimeError(f"{p.name}: numerical contract failed")
                return bufs
            n=33
            a = np.resize(np.array([0,1,0xffffffff,0x80000000,0x7fffffff,0x12345678,0xfedcba98], np.uint32), n)
            b = np.resize(np.array([1,2,17,31,0xffffffff], np.uint32), n)
            for op, expected in (("add", a+b), ("sub", a-b), ("mul", a*b),
                                 ("and", a&b), ("or", a|b), ("xor", a^b)):
                check(binary(op,n), dict(a=a.tobytes(), b=b.tobytes()), expected)
            shifts=np.resize(np.array([0,1,4,7,16,31], np.uint32),n)
            for op, expected in (("shl",a<<shifts),("shr",a>>shifts)):
                check(binary(op,n),dict(a=a.tobytes(),b=shifts.tobytes()),expected)
            for op, source in (("u32_to_f32",a),("i32_to_f32",a.view(np.int32))):
                check(unary(op,n),dict(x=source.tobytes()),source.astype(np.float32))
            values=np.resize(np.array([0.,.5,1.5,127.9,65535.75,16777216.,-0.],np.float32),n)
            check(unary("f32_to_u32",n),dict(x=values.tobytes()),np.trunc(values).astype(np.uint32))
            signed=values.copy(); signed[1::2]*=-1
            check(unary("f32_to_i32",n),dict(x=signed.tobytes()),np.trunc(signed).astype(np.int32))
            check(unary("bitcast",n),dict(x=a.tobytes()),a)
            halves=np.resize(np.array([0.,-0.,1.,-1.,1+2**-11,1+3*2**-11,65504.,2**-14,math.inf],np.float32),n)
            check(unary("f32_to_f16_rte",n),dict(x=halves.tobytes()),halves.astype(np.float16).astype(np.float32))
            bits=np.resize(np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f800000,0xff800000,0x7fc01234],np.uint32),n)
            check(k.vector("copy",n),dict(a=bits.tobytes(),b=bytes(n*4)),bits)
            # Explicit hardware FTZ policy, with signed zero, normal minima, infinity and NaN.
            floating=bits.view(np.float32); one=np.ones(n,np.float32)
            expected=floating.copy(); subnormal=(bits&0x7fffffff)<0x00800000
            expected.view(np.uint32)[subnormal]=bits[subnormal]&0x80000000
            check(k.vector("mul",n),dict(a=bits.tobytes(),b=one.tobytes()),expected,classify_nan=True)
            fa=np.resize(np.array([1+2**-12,1+2**-23,1.,0.,-2.,math.inf,math.nan],np.float32),n)
            fb=np.resize(np.array([1-2**-12,1-2**-23,2.,1.,3.,2.,1.],np.float32),n)
            fused=np.array([math.fma(float(x),float(y),-1.) for x,y in zip(fa,fb)],np.float32)
            check(k.vector("fma",n),dict(a=fa.tobytes(),b=fb.tobytes()),fused,classify_nan=True)
            x=np.resize(np.array([.125,.5,1.,2.,16.,64.],np.float32),n)
            for op, expected in (("rcp",1/x.astype(np.float64)),("rsqrt",1/np.sqrt(x.astype(np.float64))),
                                 ("log2",np.log2(x.astype(np.float64)))):
                check(unary(op,n),dict(x=x.tobytes()),expected.astype(np.float32),atol=2e-5,rtol=2e-4)
            x=np.linspace(-10,10,n,dtype=np.float32)
            check(unary("exp2",n),dict(x=x.tobytes()),np.exp2(x.astype(np.float64)).astype(np.float32),atol=2e-5,rtol=2e-4)
            bounds=np.resize(np.array([0,1,2,6,7,8,0xffffffff],np.uint32),n)
            check(bounded(n),dict(bound=bounds.tobytes()),np.clip(bounds,1,7))
            bufs=None
            for seed in range(4):
                x=np.arange(n,dtype=np.float32)+seed*10; y=np.arange(n,dtype=np.float32)/2-seed
                bufs=check(k.vector("sub",n),dict(a=x.tobytes(),b=y.tobytes()),x-y,reused=bufs)
        if identity != source_identity(): raise RuntimeError("sources changed during run")
        report.update(status="PASS",teardown="clean")
    except BaseException as error: report.update(status="FAIL",error=str(error)); raise
    finally: (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); result=run(args.output)
    print(f"PASS: {len(result['checks'])} native ISA and reuse checks")
