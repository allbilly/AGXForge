# Mesa / Asahi userspace compilation on M1 macOS

AGXForge's scalar IR now lowers independently to Mesa NIR. Unmodified Mesa
**26.2.4** supplies AGX optimization, instruction selection, register allocation,
scheduling and encoding. AGXForge validates the resulting native bytes and keeps
its existing macOS packaging and execution. No Linux DRM driver is used. The
adapter is selected with `--compiler mesa`; G13 remains the default.

The measured machine is a **base Apple M1 MacBook Air**, seven GPU cores, 8 GB,
running **macOS 27.0.1 build 26A434**, with 16 KiB pages. All six runs below use
the same runtime/compiler source identity and verified Mesa helper build.

| Verification | Metal | Direct IOGPU |
|---|---:|---:|
| Kernel checks, including tails, model projections, FP16, BF16 and comparisons | 112 PASS | 112 PASS |
| GPT-2 124M FP32 positions / dispatches / independent FP64 tensor checks | 12 / 3,264 / 2,508 PASS | 12 / 3,264 / 2,508 PASS |
| Qwen2.5-0.5B BF16 positions / dispatches / independent FP64 tensor checks | 6 / 3,780 / 2,754 PASS | 6 / 3,780 / 2,754 PASS |

GPT-2 generated ` the capital of the French Republic, and` from
`The capital of France is`. Qwen generated ` for a good job` from fixed token
IDs `9707,11,12890`. GPU argmax matched the independently computed FP64 graph at
every position. All 5,262 model tensors were bit-identical between transports;
no tensor operation used a CPU fallback. Every recorded launch completed with
intact guards. Direct dispatch processes reported zero Metal calls and no
loaded AGXMetal driver. Their support cache was prepared locally through Metal
in a separate process that refused GPU submissions.

[The receipt](receipt.json) records each complete local bundle and its full
audit hash. [The compiler regression](compiler-regression.json) confirms that
all 43 default GPT-2/Qwen program descriptors and shader hashes remained
unchanged, and that all 43 Mesa kernels compiled with authored G13 code
generation blocked. The captured experiment script is included with its
original local paths. [The unit log](unit-checks.log) records 87 passing
hardware-free checks. The independently admitted Mesa route does not inherit
the authored compiler's instruction-selection or register-allocation limits.

Mesa was built from the official SHA-256-pinned release archive. The retained
[build identity](mesa-build-identity.json) lists the actual helper hash and 1,674
source hashes, checked against that archive. Only two Meson integration edits
and the new bridge overlay differ; Mesa's compiler implementation is unchanged.
See [the build and run instructions](../../examples/macos/README.md#optional-mesa--asahi-compiler)
and [official Mesa Asahi documentation](https://docs.mesa3d.org/drivers/asahi.html).

The compact bundle deduplicates 127 compiler outputs and preserves their exact
input protocol, NIR, metadata, native bytes and provenance receipts. It retains
all 14,312 launch receipts, all model tensor-check reports and all six complete
audit manifests. Every scalar GPU output, its reference and execution images
are retained. Larger binary images and JSON collections use lossless gzip.

Run the compact verifier without NumPy or GPU access:

```sh
python3 -S evidence/macos-mesa-v2/verify.py
```

It checks the compact hash index, compiler/build identities, native byte/ABI
admission, all retained launch completions and state hashes, scalar numerical
results and Metal shader placement, model coverage and cross-transport tensor
hashes. Full model weights and tensor arrays are omitted. Recomputing all model
bounds and checking every model execution image requires the complete local
bundles named in the receipt:

```sh
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python tools/asahi_evidence.py \
  results/macos-mesa-gpt2-metal-v2
```

These are correctness measurements. **Mesa throughput was not benchmarked.**
The adapter supports the existing bounded scalar model kernels, selected FP16
operations/storage and BF16 reads. BF16 stores, tensor instructions, shader
preambles, promoted constants, scratch/shared memory, more than eight buffers
and more than 80 register halfwords are rejected. Other GPU generations, model
variants and OS builds remain unverified; this does not replace the G17 backend.
Direct IOGPU remains experimental: [earlier driver faults](../macos-gpt2-v1/README.md)
remain unresolved. All six runs here passed on their first attempts; no automatic
retries or fault recovery were added. The [original 30-check Mesa proof](../macos-mesa-v1/README.md)
is retained with its earlier, narrower scope.

INTENT: code used AGXForge's G13 encoder on macOS; the task expects Mesa/Asahi code generation; the macOS README defines native G13 execution with independent validation.
