# results/

Retained runs, extracted from the research checkout's evidence archives with their recorded SHA-256s checked
(RELEASE-MANIFEST.json lists each with the archive it came from).

| Directory | What it is | Who reads it |
|---|---|---|
| [`g17-tensor-feedmodes-v1/`](g17-tensor-feedmodes-v1/) | the B-feed runs of MM 25.103: inputs, the program, the GPU's recorded output and the receipt | [workflow 2](../examples/README.md#2-a-tensor-rule-against-hardware-evidence) |
| [`g17-tensor-common-witness-v1/`](g17-tensor-common-witness-v1/), [`g17-tensor-variant-witness-v1/`](g17-tensor-variant-witness-v1/), [`g17-tensor-stride-prediction-v1/`](g17-tensor-stride-prediction-v1/) | witness programs: Metal sources this project wrote, the objects Apple's compiler built from them, and their decoder listings | **the compiler**: `agxforge/g17/tensorlower.py` reads them as encoding templates for some tensor-lowering paths; only the witnesses it names are included |
