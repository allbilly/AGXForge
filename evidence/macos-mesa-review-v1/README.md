# Mesa review fixes on M1 macOS

The Mesa adapter now supplies a 32-bit width for displaced address additions
and implements G13 shift semantics before narrowing the result. Sources are
zero-extended, counts use their low seven bits, and effective counts 32..127
produce zero. The fix covers both shift directions and 16/32-bit source and
destination combinations.

The rebuilt Mesa 26.2.4 helper passed **134 kernel checks through Metal and
134 through direct IOGPU**, including **22 new regression cases per route**:

- Six displaced loads and six displaced stores, covering 16/32-bit elements,
  `offset`, `disp`, their sum, masked lane tails and untouched output elements.
- Eight dynamic shift cases, covering both directions, all source/destination
  width combinations, 129 lanes and counts around 16, 32, 128 and 256.
- Two immediate shift cases, including `0x10000 >> 1` into I16, which now returns
  `0x8000`, and an I16 left shift by 16, which returns zero.

The machine is the same base M1 MacBook Air on macOS 27.0.1 build 26A434 used
in the [earlier full model evidence](../macos-mesa-v2/README.md). Both new runs
passed on their first attempts. All 89 software checks pass. The independently
compiled 43 GPT-2/Qwen Mesa program descriptors and shader hashes match the
earlier model proof exactly, as do all 43 default G13 descriptors. Full models
were not rerun for this fix; the retained model comparison establishes that
their executed code and launch contracts did not change.

This bundle keeps the 44 regression launch receipts, GPU inputs and outputs,
execution images, compiler outputs, source identities, build identity and full
134-check audit manifests. Each run's `summary.json` explicitly describes the
22-case subset; `full-summary.json` preserves the original full-suite receipt.
Build identities are deduplicated with relative symlinks. The full local bundles
are named in [receipt.json](receipt.json).

Verify the artifact hashes, completion/state/code provenance and recomputed
integer outputs without NumPy or GPU access:

```sh
python3 -S evidence/macos-mesa-review-v1/verify.py
```

Rerun the entire expanded suite with the rebuilt helper:

```sh
.venv-macos/bin/python tools/build_mesa_agx.py
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/verify_gpt2_kernels.py \
  --compiler mesa --output results/macos-mesa-review-metal
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/verify_gpt2_kernels.py \
  --compiler mesa --backend iogpu --output results/macos-mesa-review-native
```

Direct IOGPU retains its existing experimental scope and historical intermittent
driver faults. Other GPU generations and OS builds were not tested.

INTENT: Mesa narrows shift inputs before execution and omits displacement-add width; the review expects G13-equivalent shifts and valid displaced loads; the macOS README promises scalar integer operations and validated G13 programs.

TWINS: searched displacement additions without a width and shifts that narrow before execution - found one other site: the left-shift path in tools/mesa_agx/bridge.c; both shift paths and their count semantics are fixed.
