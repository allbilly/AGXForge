#!/usr/bin/env python3
"""Run existing AGXForge scalar IR through Mesa NIR -> AGX -> macOS Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from agxforge.g13 import kernels as k, gpt2_kernels as g, mesa
from agxforge.runtime.macos import Executor, source_identity
from examples.asahi.verify_kernels import compare


def review_checks(check, rng):
    """Independent integer/offset oracles for the Mesa review regressions."""
    from agxforge.g13 import ir
    n = 65
    for element, dtype in ((ir.I16, np.uint16), (ir.I32, np.uint32)):
        size = np.dtype(dtype).itemsize
        values = (np.arange(n+3, dtype=np.uint32)*7919+12345).astype(dtype)
        for offset, disp in ((1, 0), (0, 2), (1, 2)):
            displacement = offset + disp
            width = "half" if size == 2 else "word"
            spec, b, t, p = k.setup(f"load_{element}_offset{offset}_disp{disp}",
                {"out": n*size, "x": (n+3)*size}, n, {"out": element, "x": element})
            loaded = b.load(p["x"], t, width=width, offset=offset, disp=disp)
            b.store_at(p["out"], t, loaded, width=width); b.ret()
            check(spec, {"x": values.tobytes()}, values[displacement:displacement+n], dtype=dtype)
            spec, b, t, p = k.setup(f"store_{element}_offset{offset}_disp{disp}",
                {"out": (n+3)*size, "x": n*size}, n, {"out": element, "x": element})
            b.store_at(p["out"], t, b.load(p["x"], t, width=width), width=width)
            b.b.ops[-1].attrs.update(offset=offset, disp=disp); b.ret()
            expected = np.full(n+3, 0xa55a, dtype)
            initial = expected.tobytes()
            expected[displacement:displacement+n] = values[:n]
            check(spec, {"out": initial, "x": values[:n].tobytes()}, expected, dtype=dtype)
    # Cross all source/destination widths with count boundaries and a lane tail.
    counts = [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 0xffffffff]
    words = [0, 1, 0x8000, 0xffff, 0x10000, 0x80000000, 0xffffffff]
    pairs = [(word, count) for word in words for count in counts]
    pairs += pairs[:17]  # 129 elements includes a masked final SIMD group.
    shifts = np.array([count for _, count in pairs], np.uint32)
    for source, source_dtype in ((ir.I16, np.uint16), (ir.I32, np.uint32)):
        source_size = np.dtype(source_dtype).itemsize
        values = np.array([word & ((1 << (source_size*8))-1) for word, _ in pairs], source_dtype)
        for dest, dest_dtype in ((ir.I16, np.uint16), (ir.I32, np.uint32)):
            dest_size = np.dtype(dest_dtype).itemsize
            for operation in ("shl", "shr"):
                spec, b, t, p = k.setup(f"{operation}_{source}_to_{dest}",
                    {"out": len(pairs)*dest_size, "x": values.nbytes, "counts": shifts.nbytes}, len(pairs),
                    {"out": dest, "x": source, "counts": ir.I32})
                value = b.load(p["x"], t, width="half" if source_size == 2 else "word")
                result = getattr(b, operation)(value, b.load(p["counts"], t), type=dest)
                b.store_at(p["out"], t, result, width="half" if dest_size == 2 else "word"); b.ret()
                mask = (1 << (dest_size*8))-1
                expected = np.array([((int(v) << (int(c)&127)) if operation == "shl" else
                                      (int(v) >> (int(c)&127))) & mask
                                     for v, c in zip(values, shifts)], dest_dtype)
                check(spec, {"x": values.tobytes(), "counts": shifts.tobytes()}, expected, dtype=dest_dtype)
    # Immediate counts also exercise constant folding of these two failures.
    for operation, source, word, count in (("shr", ir.I32, 0x10000, 1), ("shl", ir.I16, 1, 16)):
        source_dtype = np.uint32 if source == ir.I32 else np.uint16
        values = np.full(n, word, source_dtype)
        spec, b, t, p = k.setup(f"{operation}_immediate_to_i16", {"out": n*2, "x": values.nbytes}, n,
                               {"out": ir.I16, "x": source})
        value = b.load(p["x"], t, width="word" if source == ir.I32 else "half")
        result = getattr(b, operation)(value, ir.Imm(count), type=ir.I16)
        b.store_at(p["out"], t, result, width="half"); b.ret()
        expected = ((word >> count) if operation == "shr" else (word << count)) & 0xffff
        check(spec, {"x": values.tobytes()}, np.full(n, expected, np.uint16), dtype=np.uint16)


def extra_checks(check, rng):
    """Widths, selection/NaN semantics and builtins that full models don't exercise."""
    from agxforge.g13 import ir
    n = 65
    for which, expected in (
        ("thread_position_in_grid", np.arange(n, dtype=np.uint32)),
        ("thread_position_in_threadgroup", np.arange(n, dtype=np.uint32) % 32),
        ("threadgroup_position_in_grid", np.arange(n, dtype=np.uint32) // 32),
        ("thread_index_in_simdgroup", np.arange(n, dtype=np.uint32) % 32),
        ("simdgroup_index_in_threadgroup", np.zeros(n, np.uint32))):
        spec, b, t, p = k.setup("builtin_"+which, {"out": n*4}, n, {"out": ir.I32})
        b.store_at(p["out"], t, b.builtin(which)); b.ret()
        check(spec, {}, expected, dtype=np.uint32)
    left = np.array([0, 1, 2, 0xffffffff, 5, 7, 11], np.uint32)
    right = np.array([0, 2, 1, 0, 5, 8, 10], np.uint32)
    floating = np.array([0., -0., -1., np.nan, 2., np.inf, -np.inf], np.float32)
    other = np.array([-0., 0., 1., 2., np.nan, np.inf, np.inf], np.float32)
    for rel, compare_fn in (("eq", np.equal), ("lt", np.less), ("gt", np.greater)):
        for fp in (False, True):
            a, x = (floating, other) if fp else (left, right)
            spec, b, t, p = k.setup(f"{'fcsel' if fp else 'csel'}_{rel}",
                {"out": 28, "a": 28, "b": 28}, 7, {"out": ir.I32})
            predicate = compare_fn(a, x)
            value = b._def("fcsel" if fp else "csel",
                [b.load(p["a"], t), b.load(p["b"], t), b.const(3), b.const(9)], rel=rel)
            b.store_at(p["out"], t, value); b.ret()
            check(spec, {"a": a.tobytes(), "b": x.tobytes()}, np.where(predicate, 3, 9).astype(np.uint32), dtype=np.uint32)
    for op, relation in (("fmax", np.greater), ("fmin", np.less)):
        spec, b, t, p = k.setup(op+"_bits", {"out": 28, "a": 28, "b": 28}, 7, {"out": ir.I32})
        value = b._def(op, [b.load(p["a"], t), b.load(p["b"], t)])
        b.store_at(p["out"], t, value); b.ret()
        expected = np.where(relation(floating, other), floating.view(np.uint32), other.view(np.uint32))
        check(spec, {"a": floating.tobytes(), "b": other.tobytes()}, expected, dtype=np.uint32)
    half = np.array([0., -0., 1.25, -2.5, 65504., 2**-14, 1+2**-11], np.float32)
    spec, b, t, p = k.setup("half_narrow", {"out": 14, "x": 28}, 7, {"out": ir.F16})
    b.store_at(p["out"], t, b.f32_to_f16_rte(b.load(p["x"], t)), width="half"); b.ret()
    check(spec, {"x": half.tobytes()}, half.astype(np.float16).view(np.uint16), dtype=np.uint16)
    words = np.array([0, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1], np.uint16)
    spec, b, t, p = k.setup("bf16_transport_bits", {"out": 28, "x": 14}, 7, {"out": ir.I32, "x": "bfloat"})
    b.store_at(p["out"], t, k.load_float(b, p["x"], t)); b.ret()
    check(spec, {"x": words.tobytes()}, words.astype(np.uint32) << 16, dtype=np.uint32)
    conversions = (
        ("u32_to_f32", np.array([0, 1, 16777215, 16777216, 16777217, 0xffffffff], np.uint32), np.float32),
        ("i32_to_f32", np.array([0, 1, -1, 16777217, -16777217, -2147483648, 2147483647], np.int32), np.float32),
        ("f32_to_u32", np.array([0, 1.9, 2.9, 16777216, .5, 4294967040], np.float32), np.uint32),
        ("f32_to_i32", np.array([0, -0., -1.9, 1.9, -16777216, -2147483648, 2147483520], np.float32), np.int32))
    for operation, values, dtype in conversions:
        spec, b, t, p = k.setup(operation, {"out": values.nbytes, "x": values.nbytes}, len(values),
                               {"out": ir.F32 if dtype == np.float32 else ir.I32})
        b.store_at(p["out"], t, getattr(b, operation)(b.load(p["x"], t))); b.ret()
        check(spec, {"x": values.tobytes()}, values.astype(dtype), dtype=dtype)
    review_checks(check, rng)


def cases():
    rng = np.random.default_rng(213)
    for n in (31, 32, 33, 65, 129):
        a = (rng.integers(-100, 100, n) / 8).astype(np.float32)
        b = (rng.integers(-100, 100, n) / 16).astype(np.float32)
        for op, expected in (("copy", a), ("add", a+b), ("sub", a-b), ("mul", a*b)):
            yield k.vector(op, n), dict(a=a.tobytes(), b=b.tobytes()), expected, 0, 0
    for inputs, outputs in ((31, 33), (65, 31), (33, 65), (768, 2304)):
        weights = rng.normal(0, .1, (inputs, outputs)).astype(np.float32)
        x = rng.normal(0, .2, inputs).astype(np.float32)
        bias = rng.normal(0, .1, outputs).astype(np.float32)
        expected = x.astype(np.float64) @ weights.astype(np.float64) + bias
        yield g.conv1d(inputs, outputs), dict(x=x.tobytes(), weights=weights.tobytes(), bias=bias.tobytes()), expected, 3e-6, 1e-5
    x = rng.normal(0, .2, (3, 65)).astype(np.float32)
    for op, expected in (("sum", x.astype(np.float64).sum(1)),
                         ("squares", (x.astype(np.float64)**2).sum(1)), ("max", x.max(1))):
        yield k.reduce_rows(3, 65, op), dict(x=x.tobytes()), expected, 2e-6, 1e-5
    for n in (31, 65, 768):
        x = np.linspace(-100, 100, n, dtype=np.float32)
        xf = x.astype(np.float64)
        expected = .5*xf*(1+np.tanh(np.sqrt(2/np.pi)*(xf+.044715*xf**3)))
        yield g.gelu(n), dict(x=x.tobytes()), expected, 2e-5, 2e-4


def run(output, compiler):
    output.mkdir(parents=True, exist_ok=False)
    sources = source_identity()
    for p in [ROOT / "tools/build_mesa_agx.py", *sorted((ROOT / "tools/mesa_agx").glob("*"))]:
        if p.is_file(): sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (output / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    report = dict(schema_version=1, status="RUNNING", scope="AGXForge IR -> Mesa NIR -> AGX -> macOS Metal",
                  checks=[], full_model_verified=False)
    try:
        prepared = []
        for i, (spec, data, expected, atol, rtol) in enumerate(cases(), 1):
            path = output / f"{i:03d}-{spec.function.name}-{spec.threads}"
            program = mesa.compile(spec, path / "compiler", compiler=compiler)
            own = spec.compile()
            row = dict(name=program.name, threads=spec.threads, status="COMPILED",
                       mesa_register_halfs=program.register_halfs, g13_register_halfs=own.register_halfs,
                       mesa_code_bytes=len(program.code), g13_code_bytes=len(own.code), atol=atol, rtol=rtol,
                       compiler_identity=str((path / "compiler/compiler-identity.json").relative_to(output)))
            report["checks"].append(row)
            prepared.append((program, data, expected, row, path))
        with Executor() as gpu:
            (output / "platform.json").write_text(json.dumps(gpu.platform(), indent=2) + "\n")
            # Every program passes decoding and carrier packaging before any
            # GPU submission, including the full GPT-2 projection shape.
            for program, *_ in prepared: gpu.prepare(program)
            for program, data, expected, row, path in prepared:
                buffers = [gpu.buffer(b.min_bytes, b.access, data.get(b.name)) for b in program.bindings]
                launch = gpu.dispatch(program, buffers, program.logical_threads, evidence=path / "launch")
                actual = np.frombuffer(buffers[0].read(), dtype=np.float32)
                expected = np.asarray(expected).reshape(-1)
                passed, error = compare(actual, expected, row["atol"], row["rtol"])
                (path / "expected.bin").write_bytes(expected.tobytes())
                row.update(status="PASS" if passed else "WRONG_OUTPUT", max_abs_error=error,
                           expected_dtype=str(expected.dtype), canaries_intact=launch["canaries_intact"])
                (path / "check.json").write_text(json.dumps(row, indent=2) + "\n")
                if not passed:
                    gpu.poisoned = True
                    raise RuntimeError(f"{program.name}: output outside declared tolerance")
        current = source_identity()
        for key in sources.keys() - current.keys():
            current[key] = hashlib.sha256((ROOT / key).read_bytes()).hexdigest()
        if sources != current: raise RuntimeError("sources changed during verification")
        report.update(status="PASS", teardown="clean")
    except BaseException as error:
        report.update(status="FAIL", error=str(error)); raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=mesa.DEFAULT_COMPILER)
    args = parser.parse_args()
    result = run(args.output, args.compiler)
    print(f"PASS: {len(result['checks'])} Mesa-compiled kernel checks on macOS Metal")
