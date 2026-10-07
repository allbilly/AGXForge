# Base M1 / G13G on Asahi

This fork adds a scalar G13 compiler and a native Asahi DRM executor. It runs
Qwen2.5-0.5B-Instruct with BF16 weights, FP32 activations, tokenwise prefill and
GPU argmax. All model tensor operations run in AGXForge-generated shaders.
The optional FP64 CPU graph supplies independent expectations for validation;
its outputs never feed the GPU graph. File I/O, static constants, tokenization
and scheduling run on the host.

The measured machine is a base M1 MacBook Air, G13G B1, on kernel `7.1.13+`,
with 16 KiB pages. See [the hardware receipt](../../evidence/asahi-m1-v1/receipt.json)
for exact source hashes, platform identity, commands and result counts.
The [macOS route](../macos/README.md) separately supports Metal carriers and
an experimental direct IOGPU subset. M1 Pro/Max/Ultra, G13 matrix acceleration,
shared memory, SIMD cooperation, atomics, spilling and performance tuning
remain future work. G17 source and decoder validation are preserved; M5 GPU
execution has not been revalidated here. Apple's firmware remains required.

## Build and run

Use an already working Asahi installation with render-node access, a C compiler,
Python 3.10 or newer (3.12+ for the pinned NumPy/model dependencies) and the **matching installed** `drm/asahi_drm.h` / `drm.h`
headers. The helper has compile-time layout assertions and rejects GPU profiles
outside base M1 G13G A0/B1. B1 is the hardware-tested revision; A0 is admitted
by the same source profile but has not been run. Other kernel UAPI revisions
need their own review and hardware validation. Do not change the working kernel
or firmware as part of these commands.

From the repository root:

```sh
make asahi-tools
make g13-test
make asahi-probe
python3 examples/asahi/01_store.py --output results/asahi-smoke/store
python3 examples/asahi/02_add.py --output results/asahi-smoke/add
python3 examples/asahi/03_mul.py --output results/asahi-smoke/mul
```

The last three are handwritten shaders, independent of the IR compiler. The
full nine-operation suite is `make asahi-smoke`. Use a new output directory
when repeating a command: evidence is never overwritten. If `results/asahi-smoke`
already exists, run `smoke.py --output <new-directory>` instead of that make target.
The probe performs 100 VM/queue/buffer resource cycles without dispatching.
Handwritten tests cover store, IDs, copy, ADD, MUL, SUB and fused FMA with exact
bits, changing seeds, fresh allocations, varied GPU addresses and canaries.

Install model/test dependencies in an isolated environment:

```sh
python3 -m venv .venv-asahi
.venv-asahi/bin/python -m pip install -r requirements-asahi.txt
make asahi-isa PYTHON=.venv-asahi/bin/python
make asahi-kernels PYTHON=.venv-asahi/bin/python
.venv-asahi/bin/python examples/asahi/download_qwen.py
OPENBLAS_NUM_THREADS=1 .venv-asahi/bin/python examples/asahi/qwen.py \
  --verify --token-ids 9707,11,12890 --generate 4 \
  --output results/asahi-model
```

The downloader pins checkpoint revision
`7ae557604adf67be50417f59c2c2f167def9a775`; weights need about 1 GB, and each
six-position validation bundle needs about 536 MiB. The checkpoint and full run
bundles are ignored by Git. A free-running text command uses `--prompt` instead
of `--token-ids`. Omit `--verify` to run without the diagnostic CPU tensor graph;
that run reports `COMPLETED_UNVERIFIED`, even when generated text looks plausible.
This runner uses raw prompt tokens and greedy decoding, with no chat template
or EOS early stop. KV capacity defaults to 32; it must be a power of two within
the declared model limit. `--va-slot 0..15` changes shader/data virtual addresses
for relocation checks.

## Boundaries and contract

`agxforge/g13/ir.py` reuses the existing scalar SSA objects and adds narrow G13
conveniences. `cc.py` has its own selector and conservative allocation; the G17
compiler is never patched. Registers are represented by 16-bit halfword counts,
with overlapping 16/32-bit views. The compiler reserves `r0` for execution masks,
`r1` for scratch and `r2` for the tail bound, then allocates distinct registers.
It inserts load/store waits and emits no discard hints or spills.

The admitted CFG is a sequence of straight-line blocks and individually bounded
do-while loops with a zero-seeded counter advancing by one. Runtime bounds have
a static cap and at least one iteration, including a runtime bound of zero.
Nested loops and arbitrary conditional CFGs are rejected. Masked tails round
the physical dispatch to complete 32-lane groups while retaining the fixed
logical extent. One-dimensional launches and the explicitly listed builtins
are supported. This is a selected scalar subset, not the complete G13 ISA.

`G13Program` freezes code, dense logical buffer bindings, read/write directions,
minimum extents, alignment, register/uniform requirements, builtin use, numerical
policy and launch extent. The executor checks these before allocation/dispatch.
Buffer extents are author declarations: the compiler does not prove every
dynamic memory index. Model token/position bounds and static shapes are checked
by the model planner; canaries and output checks are separate hardware evidence.

The model planner builds and compiles all 22 selected programs before GPU/model
allocation. The kernel registry checks requirements derived from every selected
IR body as well as explicit declarations. Its projections are scalar GEMV;
there is no `qmm` route. A separate Q4 kernel is checked on synthetic matrices;
the accepted model checkpoint is BF16 and does not use that kernel.

The C helper discovers the node by DRM identity, creates a VM and low-priority
queue, allocates/maps/binds BOs, constructs USC and CDM packets and submits one
compute command through Asahi DRM. Pointers are relocated for each context.
Shader addresses are relative to the queue's 4 GiB USC VA window, which does not
allocate 4 GiB of physical memory. The native helper links only libc; no Mesa,
Vulkan, OpenGL, Metal or OpenCL library supplies model dispatch.

One synchronous dispatch uses a fresh DRM syncobj and a bounded monotonic wait.
Instruction waits, CDM visibility and CPU fence completion remain separate.
Weights, KV caches, temporaries and shader objects persist across token positions.
Execution is serialized by advisory GPU locks. Every error stops the sequence.
After a completion timeout, the helper closes the DRM file without explicitly
unbinding possibly active resources; this is not proof of GPU recovery. Preserve
the evidence and establish a known-good health test before starting another suite.
The current stack advertises soft faults, so neither successful submission nor
absence of a reported GPU fault substitutes for checked output.

## Numerical checks and evidence

`verify_isa.py` checks integer wraparound/subtraction/bitwise/shifts, numeric
conversion, bitcasts, FP16 rounding, signed zero, subnormals, infinities, NaNs,
cancellation-sensitive FMA, approximate operations, dynamic capped loops and
changing inputs in reused buffers. Copies preserve bits; arithmetic flushes
subnormals to signed zero on this profile; NaN arithmetic checks classification.
Finite conversions are checked within representable ranges. Invalid/overflowing
float-to-integer behavior is outside the admitted numerical test contract.

`verify_kernels.py` covers boundary sizes 31/32/33 through 127/128/129, BF16/F16/F32
GEMV, Q4 unpacking, reductions, RMSNorm, embedding, RoPE (including position 31),
KV writes, causal scores, stable softmax, attention, SwiGLU and argmax ties.
Bounds are operation-specific and fixed before dispatch. Model validation checks
459 intermediate tensors per position, including the full initialized KV cache,
masked scores, residuals and logits. Logits must satisfy
`abs(actual-reference) <= 0.05 + 0.003*abs(reference)` and GPU argmax must agree.
This independent FP64 graph follows the Qwen2 architecture; it is not a claim of
bit identity with BF16 Transformers execution or validation of all contexts.

Every dispatch bundle saves shader bytes/disassembly, USC/CDM/DRM commands,
logical bindings, relocated GPU addresses, geometry, code identity, completion
and canary status. Small tests also save before/after buffers and expected bytes.
Model bundles save every observed GPU tensor and its FP64 expectation, checkpoint
hashes, per-tensor checks and per-position receipts. Resident weights are identified
by checkpoint hashes instead of copied into every dispatch bundle. Source hashes
are checked again at the end, so an edited implementation cannot silently inherit
a passing result. Compact tracked evidence is indexed separately from full local
bundles, hardware-free checks and unavailable macOS/G17 hardware checks.

Source baselines and attribution are in
[sources.json](../../experimental/asahi-launch/sources.json) and
[THIRD-PARTY-NOTICES.md](../../experimental/asahi-launch/THIRD-PARTY-NOTICES.md).
The assembler/decoder uses pinned [applegpu](https://github.com/dougallj/applegpu)
tables. Its emulator is partial and is not the hardware oracle. Mesa's
[compute dispatch implementation](https://github.com/mirror/mesa/blob/e24dc5bd1e7fe6101bdc866fb16a15a8fcae1aae/src/asahi/vulkan/hk_cmd_dispatch.c)
and packet definitions are source references; they are not a runtime dependency.
The checkpoint's [configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json)
and Hugging Face's [Qwen2 implementation](https://github.com/huggingface/transformers/blob/ff704ffd47d800e31b31f2f81a5a2952fb0fbf71/src/transformers/models/qwen2/modeling_qwen2.py)
provide the architecture reference.

Recheck an existing bundle without dispatching:

```sh
.venv-asahi/bin/python tools/asahi_evidence.py results/asahi-model
```

The auditor recalculates output errors, checks code/state hashes and verifies
recorded completion/canary status. `--manifest <new-file>` writes a hash index
of all artifacts. This is an offline evidence audit, not a new hardware run.
