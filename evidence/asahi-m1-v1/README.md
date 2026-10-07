# Native M1 Asahi validation

This is the compact receipt for the declared G13G B1 scalar path. The complete
local artifacts remain in the ignored `results/asahi-*` directories. Each
bundle's artifact manifest is identified by SHA-256 in [receipt.json](receipt.json).
The compact receipt is hardware evidence; hardware-free tests and unavailable
platform checks are recorded separately.

| Check | Result |
|---|---|
| Resource probe | 100 VM/queue/BO lifecycle cycles, no dispatch |
| Hardware-free G13 checks | 13 tests passed on Linux |
| Handwritten execution | 81 native dispatches passed |
| Compiler ISA / buffer reuse | 26 native checks passed |
| Compiler kernel suite | 57 native checks passed, including masked boundaries |
| Qwen2.5-0.5B-Instruct | Two six-position runs, 3,780 native dispatches each |
| Model numerical checks | 459 tensors per position; 2,754 checks per run |
| Address relocation | VA slots 0 and 7; all 2,754 GPU tensor pairs bit-identical |
| Reference-disabled execution | 630 dispatches; 459 GPU tensors bit-identical to verified position 0 |
| Logits | Maximum absolute error 0.000722175 against the independent FP64 graph; top token agrees at every position |
| G17 retained suite | Attempted on Linux: 7 modules passed, 9 failed with unavailable Apple decoder/frameworks or macOS tools |
| M5 execution / M1 macOS / Metal | Not run; macOS executors remain future work |

The reference-disabled run replaces the CPU reference factory and checkpoint
floating-point reader with functions that raise if called. It completed without
using either. Its native receipt remains `COMPLETED_UNVERIFIED`; the separate
comparison against recorded verified GPU tensors passed. See
[the actual guard script](no-reference-run.py) and [its log](no-reference.log).

The model uses fixed input history `9707,11,12890`, then feeds generated tokens
back through the same resident graph. The six next tokens are
`271,358,369,264,1661,2618`; the requested four generated tokens decode to
` for a good job`. This is greedy decoding of raw tokens, not a chat-template
demonstration or a language-quality benchmark.

Recorded bounds precede dispatch. The model's logit bound is
`0.05 + 0.003*abs(reference)`; operation-specific intermediate bounds are in
[bounds.json](model/bounds.json). [error-summary.json](model/error-summary.json)
aggregates observed errors by category. GPU argmax is checked against the FP64
reference independently of generated text.

Every full bundle passed the offline artifact audit after execution. Shader and
launch-state hashes, completion, canaries and saved numerical outputs were
rechecked. The model graph has no CPU tensor-operation fallback; the FP64 graph
is an explicit diagnostic. Fault recovery, other GPU revisions, long contexts,
native macOS submission and performance comparisons are outside this receipt.

The [source tree hashes](source-tree-sha256.json), per-run compiler/executor
hashes, platform identity, checkpoint hashes, model plan and command list define
what was run. Follow [the Asahi instructions](../../examples/asahi/README.md)
to produce new evidence; use new output directories to preserve these bundles.
The original release manifest describes the archival G17 baseline, not these
new fork additions. G17 source was left unchanged and M5 execution was not
revalidated.
