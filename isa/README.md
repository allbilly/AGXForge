# isa/

The recovered instruction set and machine tables: this project's own measurement and analysis, much of it from decoding
programs with Apple's decoder and comparing them with Apple's compiler ([PROVENANCE.md](../PROVENANCE.md) says which
files are what). The compiler reads a handful of them at compile time; the rest are read by the kernel builders (which
record their inputs' hashes as provenance), the CPU emulator and the tests.

| Read when compiling | What it holds |
|---|---|
| [`tensor-isa.toml`](tensor-isa.toml), [`g17-tensor-mma-fieldmap.json`](g17-tensor-mma-fieldmap.json), [`g17-tensor-mem-memmap.json`](g17-tensor-mem-memmap.json) | the tensor unit's instruction fields and memory forms |
| [`g17-scalar-isa.toml`](g17-scalar-isa.toml), [`g17-form-opcodes.json`](g17-form-opcodes.json), [`g17-form-constants.toml`](g17-form-constants.toml), [`g17-template-opcodes.json`](g17-template-opcodes.json) | scalar forms and their encodings |
| [`g17-operand-maps.jsonl`](g17-operand-maps.jsonl) and its overlays | the measured operand-field maps the assembler encodes with |
| [`g17-contract.jsonl`](g17-contract.jsonl), [`g17-authoring.jsonl`](g17-authoring.jsonl), [`g17-slot-truth-by-length.json`](g17-slot-truth-by-length.json), [`g17-auth-lengths.json`](g17-auth-lengths.json), [`g17-auth-lifetimes.json`](g17-auth-lifetimes.json), [`g17-auth-on-templates.json`](g17-auth-on-templates.json) | per-form contracts, instruction lengths and register lifetimes the release checks enforce |
| [`g17-agx3meta-instrs.txt`](g17-agx3meta-instrs.txt), [`g17-agx3meta-regs.txt`](g17-agx3meta-regs.txt) | dumps of Apple's decoder metadata: operand counts and register names |

Also worth opening: [`g17-opcode-glossary.json`](g17-opcode-glossary.json) (every named opcode, with its confidence and
the evidence-record sections behind it) and [`g17-measurement-platform.json`](g17-measurement-platform.json) (the
hardware, OS build and toolchain every measurement was taken on).
