# Provenance of the released files

Every file in this release, by what it is and where it came from. Each file's category is also in RELEASE-MANIFEST.json (`provenance`). The sections below list the files in each category that involves Apple's tools or third-party material; the four larger categories are listed only in RELEASE-MANIFEST.json.

This project's own code and writing are licensed under the MIT License ([LICENSE](LICENSE)). Third-party material retains its applicable terms ([THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)).

| Category | Files | Bytes | What it is |
|---|---|---|---|
| apple-compiled program object | 10 | 29,408 | a program object produced by Apple's Metal compiler from a Metal source this project wrote (a witness the compiler's tensor lowering reads its templates from) |
| decoder listings of this project's programs | 11 | 278,476 | instruction listings Apple's G17 decoder produced from the witness objects above (programs compiled from this project's Metal sources); the listing's content is the decoded program |
| Apple decoder metadata dumps | 2 | 2,242,397 | dumps of Apple's own decoder metadata - the AGX3 instruction descriptors, register classes and register names inside GPUCompiler.framework's libLLVM - written by tools/agx3meta.c, which reads them from Apple's library at run time; the compiler's decoder interface reads them for operand counts and register names |
| derived from Apple's decoder metadata | 1 | 155,210 | a translation table this project generated (tools/g17renumber.py --write): it maps the opcode and register numbering of the macOS 27 (26A434) decoder back to the numbering of the original build, by matching register and register-class NAMES between the two builds' metadata dumps, aligning the two instruction-descriptor tables as monotone sequences anchored on 3,244 opcodes decoded under both builds, and fitting the immediate tags the new decoder adds; it contains id pairs and fitted constants, not Apple's tables |
| third-party model metadata | 7 | 45,551 | upstream configuration files and checkpoint headers copied or extracted from public Hugging Face repositories (Apache-2.0); see THIRD-PARTY-NOTICES.md; no weights |
| this project's compiled programs | 8 | 25,088 | programs and images this project's compiler and image author produced (in Apple's container formats) |
| recovered tables | 299 | 69,488,779 | this project's tables of the instruction set and machine, built by its own measurement and analysis (much of it from decoding programs with Apple's decoder and comparing with Apple's compiler) |
| measurement receipts | 132 | 11,413,865 | records of this project's measurements and checks; some carry model outputs, Apple tool version strings, driver log lines or decoder text as recorded |
| release-authored | 17 | 73,003 | written for this release |
| project source and writing | 475 | 14,968,718 | this project's code, tests, model configurations, Metal sources, documents and figures (the documents quote short passages of Apple SDK headers where a claim rests on them) |

## apple-compiled program object

- `results/g17-tensor-common-witness-v1/tensor-b-stride.o`
- `results/g17-tensor-common-witness-v1/tensor-common.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-6.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-6.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-128.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-256.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-80.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-96.o`
- `results/g17-tensor-variant-witness-v1/tensor-c-stride-48.o`
- `results/g17-tensor-variant-witness-v1/tensor-leading-buf.o`

## decoder listings of this project's programs

- `results/g17-tensor-common-witness-v1/tensor-b-stride.instructions.json`
- `results/g17-tensor-common-witness-v1/tensor-common.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-18.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-6.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-6.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-128.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-256.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-80.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-96.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-c-stride-48.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-leading-buf.instructions.json`

## Apple decoder metadata dumps

- `isa/g17-agx3meta-instrs.txt`
- `isa/g17-agx3meta-regs.txt`

## derived from Apple's decoder metadata

- `tools/agx3renumber.h`

## third-party model metadata

- `evidence/g17-inference-models-v1/minilm/config.json`
- `evidence/g17-inference-models-v1/minilm/modules.json`
- `evidence/g17-inference-models-v1/minilm/pooling.json`
- `evidence/g17-inference-models-v1/minilm/safetensors-header.json`
- `evidence/g17-inference-models-v1/minilm/sentence-config.json`
- `evidence/g17-inference-models-v1/qwen/config.json`
- `evidence/g17-inference-models-v1/qwen/safetensors-header.json`

## this project's compiled programs

- `results/g17-tensor-feedmodes-v1/feed_B_half/program.bin`
- `results/g17-tensor-feedmodes-v1/feed_B_half/scan.arc.metallib`
- `results/g17-tensor-feedmodes-v1/feed_B_half/scan.lib.metallib`
- `results/g17-tensor-feedmodes-v1/feed_B_half/scan.o`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/program.bin`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/scan.arc.metallib`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/scan.lib.metallib`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/scan.o`

## Not included

No Apple framework, SDK file or binary is redistributed: the decoder is loaded from the reader's own macOS at run time, and the native tools are built from source on the reader's machine. Two text files do carry Apple's own data: `isa/g17-agx3meta-instrs.txt` and `isa/g17-agx3meta-regs.txt`, the instruction-descriptor and register tables `tools/agx3meta.c` reads out of `GPUCompiler.framework` at run time and prints. They are committed so that a compile is hermetic and reproducible: `cc`'s release guard reads them on every compile, and `test_g17abi` holds a cold compile to zero external processes and to a fixed code hash. No model weights or tokenizer files are included; the workflows fetch them at pinned revisions and hashes.
