# M1 macOS execution receipts

This is the historical partial-native receipt. [v2](../macos-m1-v2/README.md)
records the completed native route; the measurements below retain their original outcomes.

[receipt.json](receipt.json) records two separate outcomes on base M1 G13G,
macOS 27.0.1 (26A434): verified Metal inference and partial direct IOGPU execution.
[Build/run commands and limits](../../examples/macos/README.md) describe both.

| Route / check | Result |
|---|---|
| Metal handwritten smoke | 81 PASS |
| Metal ISA/reuse | 26 PASS |
| Metal compiled kernels | 57 PASS |
| Metal Qwen, fixed history, six positions | 3,780 launches and 2,754 tensor checks PASS |
| Metal Qwen, allocation slot 7 | Same counts PASS; all 2,754 GPU tensors bit-identical to slot 0 |
| Metal Qwen without CPU-reference access, slot 11 | 630 launches; all 459 GPU tensors bit-identical to the verified first position |
| Direct IOGPU smoke / ISA | 81 / 26 PASS |
| Direct IOGPU complete kernel suite | FAIL: first 39 checks pass, Q4 refused before submit |
| Direct IOGPU Q4 bring-up attempts | FAIL: GPU status `0xb`; different code placement also returned `0x5` on store |
| Direct IOGPU full model | Unavailable / not run |
| Hardware-free G13/macOS checks | 48 PASS; see [build-test.log](build-test.log) |
| Retained G17 release tests | 184 PASS across 16 modules; CPU checks only |

The per-suite folders retain summaries, platform data and hashes captured during
each run. The model folders also retain checkpoint identity, bounds and compiled
program plans. [relocation-check.json](relocation-check.json) records changed GPU
addresses and exact tensor comparisons. [no-reference-run.py](no-reference-run.py)
reproduces the control that makes `Reference` and `Checkpoint.floats` raise if
called, with an independent comparison against already verified GPU outputs.
The unverified run itself keeps status `COMPLETED_UNVERIFIED`.

[failures/](failures/) preserves failing launch receipts, shader bytes/disassembly,
uniform pointers and USC/CDM packets. No failure is relabeled as a pass. The
native launch envelope currently refuses the failing larger resource profile;
the kernel suite stops at that refusal, retaining overall FAIL.

Full local bundles are in `results/macos-final/`. Their offline audits recalculate
outputs and tensor bounds and check code, state, Metal placement and native
completion notifications. Manifest paths and SHA256 hashes are in the receipt.
These compact summaries omit full archives, buffers and model tensors; they are
**not** standalone complete audit bundles. Reproduce the suites to obtain them.

Each run retains its actual source hashes. Subsequent source changes were only
attribution/docstring edits in `iogpu.py`; reproducing the corresponding old
hashes confirmed that no executable statements changed. The current identity
is retained in [current-source-sha256.json](current-source-sha256.json).

The retained G17 release checks are reported separately in the receipt; no M5
GPU execution is part of this work. Old Asahi receipts remain historical records
of the Linux route, including their then-unimplemented macOS status.
