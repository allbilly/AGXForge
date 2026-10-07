# AGXForge

**A native compiler and runtime for the Apple M5 GPU (G17) and the tensor units in its cores, built on a machine
model recovered and checked by measurement.** AGXForge compiles its own IR to G17 machine code: instruction selection,
register allocation, encoder, and the object, metadata and container the Metal path loads. The same code runs two ways:
**through Metal**, and **below Metal**, with AGXForge's runtime writing the launch state and submitting through
Apple's private IOGPU interface. All measurements are from **one M5 Pro (H17s, 48 GB)**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/execution-paths-dark.svg">
  <img alt="The two execution paths. AGXForge's compiler takes model kernels built in Python, or Metal source via Apple's front end and AIR, and produces native G17 machine code. Through Metal, Metal creates the pipelines and command buffers; this path carries every performance result. Below Metal, AGXForge's runtime writes the resources, launch state and Submit records for MiniLM and Qwen with no Metal in the process. Both paths go through Apple's IOGPU framework, kernel driver and firmware." src="docs/figures/execution-paths.svg">
</picture>

> **Status.** This is a curated, archival release of a research checkout, published as an artifact and not
> maintained. Issues are open for questions and corrections; code contributions are not being solicited.
> [RELEASE.md](RELEASE.md) says what was kept, what was left out and why.

| | |
|---|---|
| **Read** | [the technical reference](docs/g17-technical-reference.pdf) (PDF, 191 pages; [LaTeX source](docs/tex/)), [the demonstrations](docs/demonstrations.md), [the evidence record](docs/g17-tensorops-machine-model.md) |
| **Run** | [four example workflows](examples/README.md): compile a tensor product, check a tensor rule against hardware evidence, inspect the decode study, inspect native inference |
| **Inspect** | [the implementation map](#the-implementation), [what this release contains](RELEASE.md), [provenance of every file](PROVENANCE.md), [RELEASE-MANIFEST.json](RELEASE-MANIFEST.json) |

## Below Metal: a complete model, with no Metal in the process

Qwen2.5-0.5B-Instruct, asked *"Write a short greeting."*, answered **"Hello! How can I assist you today?"**
([receipt](evidence/g17-native-qwen-guarded-generation.json)). Every model operation ran as AGXForge-compiled code
launched by AGXForge's runtime. MiniLM-L6 ranks *"A puppy runs across a green
field."* (cosine 0.615) above *"The spacecraft entered orbit around Mars."* (0.171) for the query *"A dog is running
through the grass."* ([receipt](evidence/g17-native-encoder-guarded-retrieval.json)).

- Correctness is checked against FP64 reference logits, independently of the generated text: 37 of 37 Qwen logit
  vectors lie within `0.05 + 0.003·|reference|` of the original checkpoint evaluated in FP64, and agree on the top
  token, with the native tokens teacher-forced ([independent check](evidence/g17-native-qwen-generation-independent-check.json));
  MiniLM's embeddings and ranking are checked the same way. Bounds were fixed before dispatch.
- Apple still supplies the IOGPU framework, the kernel driver and the firmware, and compilation uses Apple's G17
  decoder. The CPU tokenizes and picks tokens; activations stay on the GPU.
- **Scope:** matrix operands travel in half precision with FP32 accumulation, and an auxiliary half-precision control
  fails (kept as a failure). Native execution is slower than matched Metal and about 7-15x slower than Transformers
  on MPS. Measured on macOS 26.6.2 (build 25G83) only.

## Through Metal: a decode design that beats mlx-lm at 1,792 tokens

InternLM2.5-1.8B-chat, 4-bit weights, a 1,792-token context, and one fixed 128-token history (mlx-lm's own greedy
sequence) decoded by three implementations, all through Metal, four alternating repetitions:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/decode-throughput-dark.svg">
  <img alt="Median decode throughput: mlx-lm 162.2 tokens per second; the AGXForge design with AGXForge's compiler 184.5 (1.14x); the same design with Apple's compiler 213.4 (1.32x)." src="docs/figures/decode-throughput.svg">
</picture>

| Implementation (through Metal) | Median tokens/s | Per repetition / mlx-lm |
|---|---|---|
| mlx-lm `generate_step` | 162.2 | 1.00 |
| AGXForge design, AGXForge compiler | **184.5** | 1.13-1.15 |
| AGXForge design, Apple's compiler | **213.4** | 1.31-1.33 |

The measured implementation advantage survives Apple compilation, and Apple's compiler improves it further. The
design is a parallel attention reduction, fused projections and residuals, seven dispatches per layer against
mlx-lm's sixteen, and one command buffer per token. Its Metal twins use the same threads, work partition and operation
order, compiled by Apple. They produce outputs bit-identical to the AGXForge-compiled arm's, and run faster. So
AGXForge's instruction control is not needed for this advantage, and the study does not isolate how much each fusion,
reduction or dispatch choice contributes. Scope: one model at one context; the shared history makes the
arms comparable but is not free generation (the argmax differs from mlx-lm at 3 of 128 positions, all ties or
one-bf16-step ranks); at a 196-token context AGXForge's median is slightly below mlx-lm's (0.91-1.02x per repetition
in free generation, MM 25.211). [Workflow 3](examples/README.md#3-the-matched-decode-study) re-derives these numbers from the
[receipt](evidence/g17-matched-study-v1/decode-shared-history.json)'s per-repetition rows; it does not rerun the study,
and exact reproduction is incomplete (the original 1,792-token prompt's ids were not kept).

## A tensor finding: one register, two meanings

A tensor MMA leaves its 16 x 16 result spread over 32 lanes, eight registers each. The next MMA, fed those registers
as its B operand, reads them with a different coordinate map: in lane 0, slot 4 holds D element (1, 0) but is read
as B element (8, 0). Apple-compiled `simdgroup_matrix` code packs D canonically, so a register-fed B arrives
row-rotated and must be corrected; AGXForge packs the producer so that D row 8 sits where B row 8 is read, and the
chained product needs no shuffle.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/feed-packing-dark.svg">
  <img alt="Lane 0's eight slots: canonical D coordinates, the coordinates the next MMA reads as B, and the D row each packing puts there. Canonical packing gives B row 8 the contents of D row 1; AGXForge's packing gives it D row 8." src="docs/figures/feed-packing.svg">
</picture>

The preregistered reference for this compiler's chained program applied the rotation;
the recorded GPU output matches the unrotated product on all 1,024 elements, and the rotated reading misses every
one, by up to 289.8 ([workflow 2](examples/README.md#2-a-tensor-rule-against-hardware-evidence) recomputes both on
the CPU from the retained run). Scope: one 32 x 32 stage and one SIMD group. The result is about layout; it says
nothing about timing.

In saturating int8 accumulation, the int8 MMA clips per 16-product issue when one encoding bit is set ([the
demonstrations, number 3](docs/demonstrations.md#3-what-the-tensor-unit-does-measured)); AGXForge's
compiler emits it, while the public Metal matrix interfaces examined declare no saturating operation. Apple's
cooperative-tensor chaining carries a silent hazard: a consumer narrower than its producer reads zeros or
wrong elements while the compatibility check returns true. Neither is explained by the packing above. On int8 GEMM
performance this release makes no claim of an advantage: Apple's public `matmul2d` is also exact at the measured
shapes and its tuned medians, from a later session on a later OS, are lower than AGXForge's historical ones; no paired
timing exists ([MM 25.212](docs/g17-tensorops-machine-model.md#25212-apples-public-int8-gemm-mpp-matmul2d-is-exact-at-25161s-four-shapes-and-its-tuned-medians-396-1556-us-are-below-this-projects-historical-int8-medians-not-a-paired-study-2026-10-04)).

## The implementation

| Part | Where | Notes |
|---|---|---|
| IR and builder | [`agxforge/g17/ir.py`](agxforge/g17/ir.py) | typed SSA; tensor operations are first-class |
| Compiler | [`agxforge/g17/cc.py`](agxforge/g17/cc.py), [`tlower.py`](agxforge/g17/tlower.py), [`indexgen.py`](agxforge/g17/indexgen.py), [`mmaenc.py`](agxforge/g17/mmaenc.py), [`encode.py`](agxforge/g17/encode.py) | selection, allocation, scoreboard waits, tensor lowering, encoders |
| Executor contract | [`agxforge/g17/abi.py`](agxforge/g17/abi.py) | what an executor must provide; `program.abi()` |
| Image author (Metal path) | [`agxforge/g17/scanlink.py`](agxforge/g17/scanlink.py), [`mdgen.py`](agxforge/g17/mdgen.py), [`authorobj.py`](agxforge/g17/authorobj.py), [`machobj.py`](agxforge/g17/machobj.py) | object, `__GPU_METADATA`, library and binary archive |
| Decode and prefill kernels | [`agxforge/kernels/`](agxforge/kernels/__init__.py): [`qmv.py`](agxforge/kernels/qmv.py), [`attention.py`](agxforge/kernels/attention.py), [`norm.py`](agxforge/kernels/norm.py), [`argmax.py`](agxforge/kernels/argmax.py), [`prefill_mma.py`](agxforge/kernels/prefill_mma.py), [`numerics.py`](agxforge/kernels/numerics.py) | each kernel's layout, IR builder and numerical reference (bit-exact, or an enclosure for the hardware-exp2 forms); the fp32 arithmetic the references share |
| Model graph (Metal path) | [`agxforge/inference/`](agxforge/inference/__init__.py): [`catalog.py`](agxforge/inference/catalog.py), [`assemble.py`](agxforge/inference/assemble.py), [`prefill.py`](agxforge/inference/prefill.py), [`execute.py`](agxforge/inference/execute.py) | the route from a model config to tokens: kernel forms, graph assembly from a deliver index, execution |
| Metal executor | [`tools/g17decodegen.m`](tools/g17decodegen.m), [`tools/g17twinrun.m`](tools/g17twinrun.m), [`tools/g17bundlerun.m`](tools/g17bundlerun.m) | one command buffer per token; the matched-study harness; the bundle runner |
| Native runtime (below Metal) | [`spike/agxsub/g17pure3_preflight.c`](spike/agxsub/g17pure3_preflight.c), [`tools/g17inferencesession.py`](tools/g17inferencesession.py), [`agxforge/g17/runtime.py`](agxforge/g17/runtime.py) | resources, launch state, IOGPU Submit records; the measured launch layouts of build 25G83 |
| Model pipeline | [`tools/g17inferencelower.py`](tools/g17inferencelower.py), [`g17inferenceprepare.py`](tools/g17inferenceprepare.py), [`g17inferencebundle.py`](tools/g17inferencebundle.py), [`g17nativegenerate.py`](tools/g17nativegenerate.py) | import, lowering, planning, bundling, generation |
| Decoder interface | [`agxforge/g17/model.py`](agxforge/g17/model.py), [`tools/agx3dis.c`](tools/agx3dis.c) | Apple's G17 decoder, loaded from `GPUCompiler.framework` |
| CPU emulator | [`tools/g17emu.py`](tools/g17emu.py) | runs a program from its bytes; its framing uses Apple's decoder, so it runs only on macOS |
| ISA tables | [`isa/`](isa/) | the recovered forms, operand maps and contracts the compiler reads |

## Requirements and Apple dependencies

- macOS on Apple silicon with Xcode (`xcrun metal`, `clang`). The compiler's release checks decode every program
  with Apple's G17 decoder from `GPUCompiler.framework`, so even compiling needs macOS.
- Some tensor-lowering paths read retained Apple-compiled witness objects as encoding templates
  (`agxforge/g17/tensorlower.py`, from `results/g17-tensor-*-witness-v1/` and `results/g17-tensor-stride-prediction-v1/`).
  These are included and identified file by file in [PROVENANCE.md](PROVENANCE.md). Compilation also uses Apple's
  decoder to validate emitted instructions.
- Python 3 (tested with 3.14.7) and `pip install -r requirements.txt` (NumPy, Pydantic, PyYAML, ml_dtypes). Workflows that reproduce
  measurements need more ([examples/README.md](examples/README.md)).

Running anything on the GPU needs an M5-family GPU (G17). The Metal path uses Metal and Apple's front end; the
native path uses Apple's private IOGPU framework, the AGX kernel driver and the firmware, and its launch layouts
were measured on **macOS 26.6.2 (25G83)** only. The native path has **not been re-run on any other OS build**: on
macOS 27.0.1 (26A434) it was not dispatched. A passive capture of Metal's own submission there differs from the
25G83 record in 25 of the 44 structures the runtime writes ([MM 25.213](docs/g17-tensorops-machine-model.md)). The Metal path's decode
route was re-run end to end on 26A434 for this release ([workflow 3](examples/README.md#3-the-matched-decode-study)). `python3 tools/g17platform.py --check` compares
your machine with the measured platform and says which claims a difference bears on (timing, power and occupancy
most; encodings and layouts least). [docs/execution-validation.md](docs/execution-validation.md)
describes the dispatch rules; every dispatching tool takes a machine-wide GPU lock.

```
make platform            # how this machine differs from the measured one (macOS 26.6.2, M5 Pro)
make native-tools        # the decoder wrapper and the native harnesses
make examples            # workflows 1-4: compile, simulate, and two receipt checks - no GPU dispatch
make test                # the retained tests - no GPU dispatch
```

**How it was made.** [Claude Code](https://claude.com/product/claude-code), Anthropic's coding agent, was used throughout the
project to help reverse-engineer the M5 GPU and its instruction set, and to document them.

**Licence.** This project's own code and writing are licensed under the [MIT License](LICENSE). Third-party material
retains its applicable terms: the model metadata keeps its upstream Apache-2.0 licence
([THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)). Files produced by Apple's tools or derived from Apple's decoder
metadata are listed with their origin in [PROVENANCE.md](PROVENANCE.md).

The evidence record links to research files this release does not carry; those links lead to
[docs/OMITTED.md](docs/OMITTED.md), which names each one and says why.
