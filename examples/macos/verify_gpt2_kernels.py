#!/usr/bin/env python3
"""Check the retained kernels plus GPT-2 additions on base M1, including lane tails."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from agxforge.g13 import kernels as k, gpt2_kernels as g
from examples.asahi.verify_kernels import run
from examples.macos.backend import select_backend, source_identity


def extra_checks(check, rng):
    for inputs, outputs in ((31, 33), (65, 31), (33, 65)):
        weights = rng.normal(0, .1, (inputs, outputs)).astype(np.float32)
        x = rng.normal(0, .2, inputs).astype(np.float32)
        bias = rng.normal(0, .1, outputs).astype(np.float32)
        check(g.conv1d(inputs, outputs), dict(x=x.tobytes(), weights=weights.tobytes(), bias=bias.tobytes()),
              x.astype(np.float64) @ weights.astype(np.float64)+bias, 2e-6, 1e-5)
    for d in (31, 65, 768):
        weights = rng.normal(0, .1, (8, d)).astype(np.float32)
        for pos in (0, 7):
            check(g.position_embedding(8, d), dict(weights=weights.tobytes(), params=np.array([0, pos], np.uint32).tobytes()), weights[pos])
        packed = rng.normal(0, .2, 3*d).astype(np.float32)
        # The primary output is checked here; model traces independently check all three split outputs.
        check(g.split_qkv(d), dict(packed=packed.tobytes()), packed[:d])
        for x in (rng.normal(0, .2, d).astype(np.float32), np.full(d, 3.25, np.float32)):
            expected_mean = x.astype(np.float64).mean()
            expected_scale = 1/np.sqrt(np.mean((x.astype(np.float64)-expected_mean)**2)+1e-5)
            sums = check(k.reduce_rows(1, d), dict(x=x.tobytes()), [x.astype(np.float64).sum()], 2e-6, 1e-5)
            stats = check(g.layer_norm_stats(d, 1e-5), dict(x=x.tobytes(), sum=sums.tobytes()),
                          [expected_mean, expected_scale], 1e-5, 2e-4)
            gain = rng.normal(1, .05, d).astype(np.float32); bias = rng.normal(0, .1, d).astype(np.float32)
            expected = (x.astype(np.float64)-float(stats[0]))*float(stats[1])*gain+bias
            check(g.layer_norm_apply(d), dict(x=x.tobytes(), gain=gain.tobytes(), bias=bias.tobytes(), stats=stats.tobytes()),
                  expected, 2e-6, 1e-5)
        for x in (rng.normal(0, 2, d).astype(np.float32), np.linspace(-100, 100, d, dtype=np.float32)):
            xf = x.astype(np.float64)
            expected = .5*xf*(1+np.tanh(np.sqrt(2/np.pi)*(xf+.044715*xf**3)))
            check(g.gelu(d), dict(x=x.tobytes()), expected, 2e-5, 2e-4)


if __name__ == "__main__":
    _, Executor, name, argv = select_backend()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.output, executor_factory=Executor, identity_factory=source_identity, extra_checks=extra_checks)
    print(f"PASS: {len(result['checks'])} compiled kernel checks; {name}")
