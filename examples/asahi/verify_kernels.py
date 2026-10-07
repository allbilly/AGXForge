#!/usr/bin/env python3
"""Hardware suite for compiler-generated kernels, with operation-specific bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from agxforge.g13 import kernels as k
from agxforge.runtime.asahi import Executor, source_identity


def bf16(values): return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)
def widen(values): return (values.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def compare(actual, expected, atol=0, rtol=0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if atol == 0 and rtol == 0:
        passed = actual.astype(expected.dtype).tobytes() == expected.tobytes()
        return bool(passed), 0.0 if passed else None
    finite = np.isfinite(expected)
    error = np.abs(actual[finite].astype(np.float64) - expected[finite])
    passed = np.all(np.isfinite(actual[finite])) and np.all(error <= atol + rtol*np.abs(expected[finite]))
    passed = passed and np.array_equal(actual[~finite], expected[~finite])
    return bool(passed), float(np.max(error, initial=0))


def run(output, *, executor_factory=Executor, identity_factory=source_identity, extra_checks=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    sources = identity_factory()
    report = dict(schema_version=1, status="RUNNING", scope="AGXForge IR -> G13 -> GPU", checks=[])
    (output / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    rng = np.random.default_rng(213)
    gpu = None
    try:
        with executor_factory() as gpu:
            (output / "platform.json").write_text(json.dumps(gpu.platform(), indent=2) + "\n")
            def check(spec, data, expected, atol=0, rtol=0, dtype=np.float32):
                p = spec.compile()
                row = dict(name=p.name, logical_threads=spec.threads, atol=atol, rtol=rtol, status="NOT_RUN")
                report["checks"].append(row)
                path = output / f"{len(report['checks']):03d}-{p.name}"
                bufs = [gpu.buffer(binding.min_bytes, binding.access, data.get(binding.name)) for binding in p.bindings]
                launch = gpu.dispatch(p, bufs, spec.threads, evidence=path)
                actual = np.frombuffer(bufs[0].read(), dtype=dtype)
                expected = np.asarray(expected).reshape(-1)
                passed, err = compare(actual, expected, atol, rtol)
                (path / "expected.bin").write_bytes(expected.tobytes())
                row.update(status="PASS" if passed else "WRONG_OUTPUT", max_abs_error=err,
                           expected_dtype=str(expected.dtype), actual_dtype=str(np.dtype(dtype)),
                           fence_completed=True, canaries_intact=launch["canaries_intact"])
                (path / "check.json").write_text(json.dumps(row, indent=2) + "\n")
                if not passed:
                    gpu.poisoned = True
                    raise RuntimeError(f"{p.name}: output outside declared bound; stopped")
                return actual
            for n in (31, 32, 33, 63, 64, 65, 127, 128, 129):
                a = (rng.integers(-100, 100, n) / 8).astype(np.float32)
                b = (rng.integers(-100, 100, n) / 16).astype(np.float32)
                for op, expected in (("copy", a), ("add", a+b), ("sub", a-b), ("mul", a*b)):
                    check(k.vector(op, n), {"a": a.tobytes(), "b": b.tobytes()}, expected)
            for dtype in ("bfloat", "f16", "f32"):
                w = rng.normal(0, .1, (33, 65)).astype(np.float32)
                storage = bf16(w) if dtype == "bfloat" else w.astype(np.float16) if dtype == "f16" else w
                wr = widen(storage) if dtype == "bfloat" else storage.astype(np.float64)
                x = rng.normal(0, .2, 65).astype(np.float32)
                check(k.gemv(33, 65, dtype), {"x": x.tobytes(), "weights": storage.tobytes()}, wr @ x, 2e-6, 1e-5)
            weights = rng.integers(0, 16, (33, 16), dtype=np.uint32)
            packed = np.zeros((33, 2), np.uint32)
            for j in range(8): packed |= weights[:, j::8] << (j*4)
            x = rng.normal(0, .2, 16).astype(np.float32)
            scales = np.full(33, .125, np.float32); offsets = np.full(33, -.5, np.float32)
            check(k.qmv4(33, 16), dict(weights=packed.tobytes(), x=x.tobytes(), scale=scales.tobytes(), offset=offsets.tobytes()),
                  (weights.astype(np.float64)*scales[:, None]+offsets[:, None]) @ x, 2e-6, 1e-5)
            x = rng.normal(0, .2, (3, 65)).astype(np.float32)
            for op, expected in (("sum", x.astype(np.float64).sum(1)), ("squares", (x.astype(np.float64)**2).sum(1)), ("max", x.max(1))):
                check(k.reduce_rows(3, 65, op), dict(x=x.tobytes()), expected, 2e-6, 1e-5)
            d = 65; x = rng.normal(0, .2, d).astype(np.float32); gain = bf16(rng.normal(1, .05, d))
            sums = np.array([(x.astype(np.float64)**2).sum()], np.float32)
            scale = check(k.norm_scale(d, 1e-6), dict(sum=sums.tobytes()), np.array([1/np.sqrt(float(sums[0])/d+1e-6)]), 1e-5, 2e-4)
            check(k.norm_apply(d), dict(x=x.tobytes(), gain=gain.tobytes(), scale=scale.tobytes()),
                  x.astype(np.float64)*widen(gain)*scale[0], 2e-6, 1e-5)
            vocab, d, token = 7, 65, 3; w = bf16(rng.normal(0, .1, (vocab, d)))
            check(k.embedding(vocab, d), dict(weights=w.tobytes(), params=np.array([token, 0], np.uint32).tobytes()), widen(w)[token])
            heads, kvh, hd, cap = 14, 2, 64, 32
            inv = (1e6**(-np.arange(hd//2, dtype=np.float64)*2/hd)).astype(np.float32)
            x = rng.normal(0, .2, (heads, hd)).astype(np.float32)
            for pos in (0, 1, 3, 31):
                angles = pos*inv.astype(np.float64)
                lo, hi = x[:, :hd//2], x[:, hd//2:]
                expected = np.concatenate([lo*np.cos(angles)-hi*np.sin(angles), hi*np.cos(angles)+lo*np.sin(angles)], axis=1)
                check(k.rope(heads, hd), dict(x=x.tobytes(), invfreq=inv.tobytes(), params=np.array([0, pos], np.uint32).tobytes()), expected, 2e-5, 2e-4)
            pos = 3; params = np.array([0, pos], np.uint32).tobytes()
            cache = np.zeros((cap, kvh, hd), np.float32); v = rng.normal(0, .2, (kvh, hd)).astype(np.float32)
            cache[pos] = v
            check(k.kv_write(kvh*hd, cap), dict(cache=bytes(cache.nbytes), x=v.tobytes(), params=params), cache)
            keys = rng.normal(0, .2, (cap, kvh, hd)).astype(np.float32)
            values = rng.normal(0, .2, (cap, kvh, hd)).astype(np.float32)
            mapping = np.repeat(np.arange(kvh, dtype=np.uint32), heads//kvh)
            expected = np.einsum('hd,thd->ht', x.astype(np.float64), keys[:, mapping].astype(np.float64))/np.sqrt(hd)
            expected[:, pos+1:] = -np.inf
            check(k.scores(heads, kvh, hd, cap), dict(q=x.tobytes(), k=keys.tobytes(), headmap=mapping.tobytes(), params=params), expected, 2e-6, 1e-5)
            scores = expected.astype(np.float32); maximum = scores.max(1)
            er = np.exp(scores.astype(np.float64)-maximum[:, None])
            exps = check(k.softmax_element(heads, cap, "exp"), dict(x=scores.tobytes(), row=maximum.tobytes()), er, 2e-5, 2e-4)
            denom = exps.reshape(heads, cap).astype(np.float64).sum(1).astype(np.float32)
            probs = check(k.softmax_element(heads, cap, "normalize"), dict(x=exps.tobytes(), row=denom.tobytes()),
                          exps.reshape(heads, cap).astype(np.float64)/denom[:, None], 2e-5, 2e-4).reshape(heads, cap)
            expected = np.einsum('ht,thd->hd', probs.astype(np.float64), values[:, mapping].astype(np.float64))
            check(k.attend(heads, kvh, hd, cap), dict(probs=probs.tobytes(), v=values.tobytes(), headmap=mapping.tobytes()), expected, 2e-6, 1e-5)
            logits = rng.normal(0, .2, 129).astype(np.float32); logits[63] = 9; logits[65] = 9
            check(k.argmax(129), dict(x=logits.tobytes()), np.array([63], np.uint32), dtype=np.uint32)
            a = rng.normal(0, .2, 65).astype(np.float32); up = rng.normal(0, .2, 65).astype(np.float32)
            check(k.vector("swiglu", 65), dict(a=a.tobytes(), b=up.tobytes()), a.astype(np.float64)/(1+np.exp(-a.astype(np.float64)))*up, 2e-5, 2e-4)
            if extra_checks: extra_checks(check, rng)
        if sources != identity_factory(): raise RuntimeError("sources changed during verification")
        report["status"] = "PASS"; report["teardown"] = "clean"
    except BaseException as error:
        report.update(status="FAIL", error=str(error)); raise
    finally:
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(f"PASS: {len(result['checks'])} compiler-generated native kernel checks; {args.output}")
