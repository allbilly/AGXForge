# GPT-2 on base M1: correctness and compiler comparison

The standard 124M GPT-2 checkpoint runs through AGXForge's G13 compiler on this
base M1 / macOS 27.0.1 build 26A434. Metal, direct IOGPU and the separately
Apple-compiled scalar baseline each completed 12 token positions, 3,264 GPU
dispatches and 2,508 independently computed FP64 tensor checks. They generated
` the capital of the French Republic, and` from `The capital of France is`.
The G13 Metal and successful native tensors were bit-identical; Apple results
passed the same bounds and top-token checks, with different FP32 rounding.

The [receipt](receipt.json), [checkpoint identity](checkpoint.json) and
[run instructions](../../examples/macos/README.md#gpt-2-and-compiler-comparison)
record the exact scope. The G13 kernel suite passed 93 checks on each transport,
and [66 hardware-free checks](unit-checks.log) passed. Qwen's Metal regression
passed four positions and 1,836 tensor checks. Its native regression exposed the
same intermittent driver error observed with GPT-2.

The final [compiler comparison](compiler-benchmark/summary.json) uses identical
weights, scalar graph, buffers, 32-lane groups and token history. Each backend
ran five measured sequences, each preceded by a full warmup in a separate process.
Five prompt positions precede seven measured decode positions at capacity 32.
CPU reference work, compilation, uploads and evidence capture are outside the
timed intervals; GPU waits, scheduling, canaries and GPU argmax reads are timed.

| Compiler and transport | Median decode tokens/s | Range across five rounds | Median prompt time |
|---|---:|---:|---:|
| AGXForge G13 through Metal | 5.709 | 4.412–7.720 | 1.141 s |
| Apple-compiled scalar MSL through Metal | 8.585 | 5.407–11.211 | 0.673 s |

The ratio of median rates is **1.504×** for Apple over AGXForge. Apple was faster
in all five paired rounds; the median paired ratio is 1.452×. The range shows
substantial run-to-run variation on this shared Mac. This is a short-context
scalar graph comparison, not an optimized MLX, PyTorch/MPS or batched prefill result.

An [earlier transport comparison](earlier-native-transport-benchmark/summary.json)
completed five rounds at **5.986 tokens/s** for direct IOGPU and **4.242 tokens/s**
for G13 through Metal, a ratio of 1.411×. This is a separate cohort: these numbers
must not be combined into a simultaneous three-way compiler comparison.

Direct IOGPU remains experimental. Later full validation produced intermittent
channel errors with notification `status 0xe`, including on unchanged Qwen
kernels. Fresh contexts have also completed full GPT-2 validation and all five
earlier timed rounds. The cause is unresolved. A full CDM barrier experiment
also failed and was reverted; the current native packets retain the previous
barrier. No automatic retries, ignored errors or fault recovery were added.
All six [failed experiments](failures/) retain FAIL, including the first Apple
baseline's NaN in fast-math `tanh`. Its GELU was changed to the equivalent stable
sigmoid form and revalidated before timing.

The compact folders preserve summaries, platforms, source identities, plans,+bounds, per-round timing and failing packets. Complete local bundles remain
under `results/macos-gpt2/`; the receipt records full-bundle manifest hashes.
The compiler cohort's model and benchmark sources match exactly. The successful
native model uses identical G13 program descriptors and inference implementation;
its captured source identity predates changes to the benchmark script only.
These compact files omit full weights and tensor arrays and cannot independently
repeat the full numerical audit without those local bundles.

Run `python3 evidence/macos-gpt2-v1/verify.py` to check this compact index, the
worker records, median calculations, cohort separation and failure statuses.
Run `tools/asahi_evidence.py` on the full model bundles to check tensor bounds,
argmax, execution images, launch notifications and state hashes.

The implementation adds an FP32 GPT-2 adapter, position embeddings, stable
two-pass LayerNorm, packed Q/K/V splitting, original-layout Conv1D projections
and GELU. It reuses attention, reductions, KV storage and GPU argmax. The shared
validation harness accepts model factories while retaining Qwen defaults. The
Apple baseline executes `xcrun metal` output and records it distinctly from G13
descriptors. The pinned checkpoint and architecture references are in
[sources.json](sources.json). Other GPT-2 checkpoint sizes and other macOS builds
are unverified. Downloaded Mesa investigation files were removed after their
source URLs and hashes were recorded.

INTENT: code has no GPT-2 runner; the task expects GPT-2 inference; the macOS README requires authored G13 execution with independent FP64 validation.

INTENT: Metal tanh yields NaN for a finite GPT-2 activation; the task expects finite GELU; GPT-2's reference defines the tanh approximation.

TWINS: searched GPU GELU tanh calls and parsed all 431 repository Python files - found 0 other sites: none.
