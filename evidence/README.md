# evidence/

The receipts behind the public claims, as recorded. Each claim in the [README](../README.md) and the
[demonstrations document](../docs/demonstrations.md) links to the file it rests on; the [evidence record](../docs/g17-tensorops-machine-model.md)
cites them by section.

| Claim | Receipts |
|---|---|
| Base M1 GPT-2 validation and Apple compiler comparison; native reliability limits | [`macos-gpt2-v1/`](macos-gpt2-v1/README.md), including [receipt](macos-gpt2-v1/receipt.json) and separate timing cohorts |
| Base M1 G13 on macOS: verified Metal and direct IOGPU inference | [`macos-m1-v2/`](macos-m1-v2/README.md), including the [receipt](macos-m1-v2/receipt.json); [`macos-m1-v1/`](macos-m1-v1/README.md) retains the earlier partial route and failures |
| The decode comparison (162.2 / 184.5 / 213.4 tokens/s, through Metal; MM 25.211) | [`g17-matched-study-v1/`](g17-matched-study-v1/): [`decode-shared-history.json`](g17-matched-study-v1/decode-shared-history.json), per-kernel identity in [`kernels.json`](g17-matched-study-v1/kernels.json), what can and cannot be reproduced in [`inputs-recovery.json`](g17-matched-study-v1/inputs-recovery.json) |
| Qwen and MiniLM below Metal (MM 25.210) | [`g17-native-qwen-generation.json`](g17-native-qwen-generation.json) with its [independent check](g17-native-qwen-generation-independent-check.json); [`g17-native-encoder-guarded-retrieval.json`](g17-native-encoder-guarded-retrieval.json) with its [independent check](g17-native-encoder-guarded-retrieval-independent.json); the pinned model inputs in [`g17-inference-models-v1/`](g17-inference-models-v1/); the measured OS build in [`g17-native-callback-ownership.json`](g17-native-callback-ownership.json) |
| The controls that fail, kept as failures (MM 25.210.10) | `g17-native-qwen-framework-fp32-control.json`, `g17-native-qwen-*-failure.json`, `g17-resident-*-failure.json` |
| Apple's int8 `matmul2d` baseline (MM 25.212) | [`g17-mpp-int8-v1/`](g17-mpp-int8-v1/), with its [provenance](g17-mpp-int8-v1/provenance.json) |
| The historical int8 measurement it qualifies (MM 25.161) | [`g17-three-threads-v1/`](g17-three-threads-v1/) |

The other `g17-native-*`, `g17-resident-*` and `g17-inference-*` files are the stages of the native path the evidence
record's section 25.210 walks through: lowering, preparation, references, startup and timing.
