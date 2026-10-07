# Complete M1 macOS execution receipts

[receipt.json](receipt.json) records completed Metal and direct IOGPU coverage on
base M1 G13G, macOS 27.0.1 build 26A434. [Run commands and limits](../../examples/macos/README.md)
include the local native support setup. This is correctness evidence on that
specific machine and OS build.

| Check | Result |
|---|---|
| Metal smoke / ISA / compiled kernels | 81 / 26 / 57 PASS; retained in [v1](../macos-m1-v1/README.md) |
| Metal Qwen, six positions | 3,780 launches and 2,754 tensor checks PASS; retained in v1 |
| Direct IOGPU smoke / ISA / compiled kernels | 81 / 26 / 57 PASS, including Q4 |
| Direct IOGPU Qwen, six positions | 3,780 launches and 2,754 tensor checks PASS |
| Native allocation slot 7 | Same counts PASS; all 2,754 GPU tensors match slot 0 exactly |
| Native and Metal tensor comparison | All 2,754 GPU tensors bit-identical |
| Native reference-disabled control, slot 11 | 630 launches; all 459 tensors match the verified first position exactly |
| Native command/encoder IDs | 7,560 distinct IDs in each full native model run |
| Hardware-free G13/macOS checks | 53 PASS |
| Retained G17 release checks | 184 PASS across 16 modules; CPU checks from v1 |

Both model routes generated tokens `[369, 264, 1661, 2618]`, text ` for a good job`,
from input `[9707, 11, 12890]`. Every native tensor check and top-token comparison
passes the independently implemented FP64 reference bounds.

The direct runtime preserves locally collected driver helper code, installs
compiled programs at immutable entries beyond it, allocates model tensors
separately from the captured views, handles exact-end notification-ring wraps,
and allocates fresh command/encoder IDs for each launch. The support cache is
generated in `build/macos/` through Metal pipeline creation in a separate process
with GPU submissions refused. It is ignored by Git and is not distributed here.
Its manifest and hash are retained. Native model execution calls no Metal APIs
and rejects AGXMetal in its process.

[verify-controls.py](verify-controls.py) reproduces the relocation, Metal byte
comparison and trace-ID checks from full bundles. [no-reference-run.py](no-reference-run.py)
blocks `Reference` and `Checkpoint.floats` during native inference and then compares
all first-position tensors against an already verified GPU bundle. That raw run
keeps its honest `COMPLETED_UNVERIFIED` status; its independent comparison is PASS.

The compact suite folders retain summaries, platform data and source identities.
Model folders add checkpoint, plan and bounds. Full local bundles are under
`results/macos-native-fix/` and `results/macos-final/`; the shared auditor checks
all recorded outputs, tensor bounds, code/state hashes and native notifications.
Full-bundle manifest hashes are in the receipt. These compact files omit full
buffers, archives, support bytes and model tensors and are not complete standalone
audit bundles.

[failures/](failures/) retains the earlier Q4 fault, ring-boundary rejection,
oversized arena residency failure and stale trace-ID failure. Their statuses
remain FAIL. [v1](../macos-m1-v1/README.md) preserves the earlier partial native
route and its original evidence.
