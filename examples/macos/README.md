# Base M1 / G13G on macOS

Both routes run AGXForge-authored G13 machine code, including the BF16
Qwen2.5-0.5B-Instruct and FP32 GPT-2 graphs. Qwen shares the compiler, model planner and independent
FP64 references with [the Asahi route](../asahi/README.md). The measured platform
is a base M1 MacBook Air on macOS **27.0.1, build 26A434**, with 16 KiB pages.
Other M1-family GPUs are rejected. Other macOS releases have not been validated.

Metal passed 81 handwritten smoke checks, 26 ISA/reuse checks, 57 compiled
kernel checks and 2,754 intermediate tensor checks over six Qwen positions.
The 3,780 model dispatches produced ` for a good job` from fixed input tokens
`9707,11,12890`, with GPU argmax matching the independent FP64 graph at every
position. These are correctness measurements, not a throughput comparison.
See [the receipts](../../evidence/macos-m1-v2/receipt.json).

A second six-position run at allocation slot 7 passed with all 2,754 GPU tensors
bit-identical to slot 0. A one-position run at slot 11 blocked CPU-reference
entry points and reproduced all 459 already verified GPU tensors exactly.

Direct IOGPU passes the same 81 smoke, 26 ISA and 57 kernel checks, including
Q4, and all six Qwen positions with 2,754 independent tensor checks. Its
dispatch process performs no Metal API calls and loads no AGXMetal driver.
One setup step creates Metal pipelines in a separate process to collect the
driver support code locally; GPU submissions are refused during that step.
The direct route is experimental and restricted to the exact measured OS build.

## Build and run

Use Xcode with the Metal compiler installed (`xcrun metal` and `xcrun metallib`),
and Python 3.12 or newer. The handwritten smoke suite and hardware-free G13
checks use the standard library. Numerical suites and model execution use the
existing shared dependencies in `requirements-asahi.txt`:

```sh
make macos-tools
make macos-test
python3 -m venv .venv-macos
.venv-macos/bin/python -m pip install -r requirements-asahi.txt
python3 examples/macos/probe.py
make macos-support
python3 examples/macos/probe.py --backend iogpu
.venv-macos/bin/python examples/macos/smoke.py --output results/macos-metal-smoke
.venv-macos/bin/python examples/macos/verify_isa.py --output results/macos-metal-isa
.venv-macos/bin/python examples/macos/verify_kernels.py --output results/macos-metal-kernels
```

Every suite requires a **new** output directory. `--backend metal` is the default;
`--backend iogpu` selects the direct transport. Both probes exercise three
resource lifecycles without submitting GPU commands. The convenience Make
targets accept `PYTHON=.venv-macos/bin/python` and `MACOS_BACKEND=iogpu`.
Run `make macos-support` once before selecting IOGPU. The support cache is
validated against its manifest and the measured template on every new executor.

Download the pinned public checkpoint and run verified inference:

```sh
.venv-macos/bin/python examples/asahi/download_qwen.py --output models/macos/qwen2.5-0.5b
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/qwen.py \
  --checkpoint models/macos/qwen2.5-0.5b --verify --token-ids 9707,11,12890 \
  --generate 4 --output results/macos-metal-model
.venv-macos/bin/python tools/asahi_evidence.py results/macos-metal-model
# Direct execution uses the same pinned checkpoint and validation:
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/qwen.py \
  --backend iogpu --checkpoint models/macos/qwen2.5-0.5b --verify \
  --token-ids 9707,11,12890 --generate 4 --output results/macos-iogpu-model
.venv-macos/bin/python tools/asahi_evidence.py results/macos-iogpu-model
```

The default checkpoint path remains `models/asahi/qwen2.5-0.5b`, allowing both
routes to share an existing download. Omitting `--verify` disables the CPU
reference and records `COMPLETED_UNVERIFIED`; completion alone is not a
correctness pass. All inference tensor operations and argmax execute on the GPU.
Tokenization, uploads, static constants and scheduling run on the CPU.

## GPT-2 and compiler comparison

`gpt2.py` runs the standard 124M GPT-2 checkpoint with learned position
embeddings, LayerNorm, packed Q/K/V Conv1D projections and `gelu_new`.
Weights remain in their original FP32 checkpoint layout; the GPU performs
projections, Q/K/V splitting, attention, normalization, activation and argmax.
The pinned checkpoint is `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. No framework model importer is involved.

```sh
.venv-macos/bin/python examples/macos/download_gpt2.py
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/verify_gpt2_kernels.py \
  --output results/macos-gpt2-kernels
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/gpt2.py \
  --verify --prompt 'The capital of France is' --generate 8 \
  --output results/macos-gpt2-metal
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/gpt2.py \
  --backend iogpu --verify --prompt 'The capital of France is' --generate 8 \
  --output results/macos-gpt2-iogpu
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/gpt2_apple.py \
  --verify --prompt 'The capital of France is' --generate 8 \
  --output results/macos-gpt2-apple
OPENBLAS_NUM_THREADS=1 .venv-macos/bin/python examples/macos/benchmark_gpt2.py \
  --metal results/macos-gpt2-metal \
  --apple results/macos-gpt2-apple --rounds 5 --output results/macos-gpt2-comparison
```

All three graphs passed 2,508 independent FP64 tensor checks over 12 positions
and generated ` the capital of the French Republic, and`. The G13 Metal and
native tensors were bit-identical. The expanded G13 kernel suite passed 93
checks on both routes, including non-square Conv1D layouts, lane tails, zero
variance and saturated GELU inputs. Other GPT-2 checkpoint sizes are unverified;
the measured native runner reserves a 640 MiB tensor arena for this checkpoint.

The Apple baseline uses separately authored scalar MSL for the same graph,
compiled by `xcrun metal`, with the same buffers, 32-lane groups, scheduling,
GPU waits and guard checks. It executes Apple's code, and its evidence explicitly
distinguishes that from the G13 program descriptors. Its stable sigmoid GELU
form avoids a NaN observed in the default fast-math lowering of `tanh` for a
finite activation. The shared auditor checks the MSL source, library and archive
hashes and the independently verified tensors.

The benchmark requires audited model bundles with matching checkpoint, source
identity, context capacity, token history and input prompt, including the same
prefill/decode boundary. The requested backend must match the executor in the
summary, platform and every launch record. Each worker prepares pipelines,
warms one complete sequence, and times a second sequence. Workers run in separate
processes, with the order rotated across five rounds. Reported decode rates cover
seven positions following five prompt positions. Uploads, compilation, CPU
reference work and evidence writes are excluded; host scheduling, GPU waits,
canary checks and GPU argmax reads are included. This is a scalar graph comparison,
not an optimized MLX, PyTorch/MPS or batched prefill benchmark.
Add `--iogpu results/macos-gpt2-iogpu` to include the native transport when its
verified bundle is available. The measured compiler comparison was 8.585 tokens/s
for Apple and 5.709 for AGXForge through Metal, a 1.504x ratio of medians.
The earlier native comparison was a separate measurement cohort.

Direct IOGPU has also produced intermittent channel errors (`status 0xe`) in
GPT-2 and Qwen validation. Those runs remain FAIL; fresh contexts have completed
full validation and timing sequences. The cause is unresolved, and an executor
always stops after an error. See the [GPT-2 evidence](../../evidence/macos-gpt2-v1/README.md)
for the final measurements and retained failure records.
After an error, existing buffers also reject CPU reads, writes and guard checks;
closing the executor is still permitted.

## Metal implementation and boundaries

`agxforge/runtime/macos.m` supplies device, buffer, archive, pipeline and command
buffer operations. A locally compiled eight-buffer carrier reserves enough
code and registers for the selected G13 programs. `agxforge/g13/metal.py` locates
the unique native `_agc.main` symbol with bounded Mach-O traversal and replaces
only its instruction range. The archive is loaded with
`MTLPipelineOptionFailOnBinaryArchiveMiss`: an archive miss fails instead of
compiling the carrier source as a dispatch fallback. Distinct shader hashes use
distinct archive/library paths to avoid pipeline cache aliasing. The carrier
itself is never dispatched.

The admitted profile has eight pointer bindings, at most 80 register halfwords,
32-lane SIMD groups, scalar code, no spills or shared memory, and the builtin
contract in `G13Program`. The compiler and ABI enforce immutable code, bindings,
extents, alignment, register/uniform declarations and supported instructions.
Buffer access declarations are checked on the host. Metal chooses GPU virtual
addresses; `--va-slot` varies allocation placement without claiming control over
that allocator. Exact code, pipeline placement, bindings, geometry, completion
and 64-byte buffer canaries are recorded for every dispatch.

## Direct IOGPU implementation and limits

`agxforge/runtime/iogpu.c` opens the base-M1 `AGXAcceleratorG13G` user client,
creates resources through IOKit selectors and submits with `IOConnectTrap4`.
The Python wrapper restores a measured launch envelope with fresh CPU/GPU
addresses and authors shader bytes, uniform pointers and USC/CDM packets per
launch. This is a measured envelope, not a general macOS launch-layout builder.
The template was captured from this project's successful Metal store launch;
all code-heap bytes and userspace Metal object pointers were removed. No Apple
shader, framework or model weight is redistributed in the template.

The native profile admits eight bindings, 80 register halfwords and 32 lanes
per workgroup. It reserves the captured carrier's larger USC/CDM resource counts.
The local support cache occupies bytes 0..0x4fff of the code heap; immutable
compiled entries start at 0x5000. The remaining 44 KiB pool rejects exhaustion.
Each launch invalidates the USC cache and receives fresh command/encoder IDs.
A separately allocated, guarded tensor arena has one residency entry, preserving
the small captured views. The model command reserves a 1,152 MiB arena.
GPU addresses, command pages, the 64-byte trap record and two native completion
notifications are saved. A completion requires two distinct matching cookies,
ordered timestamps, zero error/status fields and intact canaries. The C parser
checks ring bounds, permits Apple's exact-end wraparound, and uses a bounded
monotonic wait.

The helper links IOKit, with no Metal linkage or API calls. On this macOS build,
the system framework dependency closure can incidentally load Metal.framework;
this route therefore makes no claim that Metal.framework is absent from the
process. Loading AGXMetal is refused. Do not mix both executors in one process
after opening Metal; use separate commands.

The earlier Q4 fault came from erasing driver helper code that the envelope
still invokes. Retaining local support code and separating authored entries
resolved it. Model bring-up also exposed an exact-end ring rejection, duplicate
residency accounting when growing the captured arena, and reused trace IDs.
The earlier failures remain recorded as failures in the evidence directories.
Larger layouts, additional OS builds and fault recovery are unverified.

Both executors serialize work with advisory GPU locks and stop after any failed
submission, timeout, canary or numerical check. A stopped executor cannot prepare,
allocate or submit more work. Closing resources is not proof of GPU recovery.

## Evidence and provenance

The [compact evidence index](../../evidence/macos-m1-v2/README.md) retains passing
summaries, per-run source identities, platform details and the failing native
receipt. Full local bundles contain code, archives or native packets, outputs,
expected values and model tensors. The shared offline auditor checks shader/state
hashes, embedded Metal code placement, native completion cookies and pointers,
output errors and per-tensor bounds. A failed or unverified bundle cannot pass.

Source references and attribution are in
[sources.json](../../experimental/macos-launch/sources.json) and
[THIRD-PARTY-NOTICES.md](../../experimental/macos-launch/THIRD-PARTY-NOTICES.md).
The retained G17 code and hardware scope remain separate; this work does not
validate M5 execution or extend the G13 compiler's instruction subset.
