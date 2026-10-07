# Documentation

| Document | Read it for |
|---|---|
| [M1 macOS execution](../examples/macos/README.md) | Verified G13 Metal and direct IOGPU inference, build/run commands and measured limits |
| [demonstrations.md](demonstrations.md) | the three demonstrations, each with its scope and links into the evidence |
| [g17-technical-reference.pdf](g17-technical-reference.pdf) | how the machine, its instruction set, the compiler and both execution paths work; every statement cites the evidence record (LaTeX source in [tex/](tex/), `make -C docs/tex` rebuilds it) |
| [g17-tensorops-machine-model.md](g17-tensorops-machine-model.md) | the evidence record: every measurement in numbered sections ("MM 25.211"), corrections and failed controls kept in place |
| [g17-tensorops-accelerator-recon.md](g17-tensorops-accelerator-recon.md) | the earlier tensor-unit investigation the evidence record cites by section number ("section 132") |
| [findings.md](findings.md) | episodes where a measurement overturned a claim |
| [g17-isa-reference.md](g17-isa-reference.md) | the instruction set by family, generated from `isa/g17.yaml`: each form and the evidence level behind it |
| [agx3-oracle.md](agx3-oracle.md) | how the project runs Apple's own G17 decoder to check its encodings |
| [OMITTED.md](OMITTED.md) | the research files the evidence record links to that this release does not carry, and why |
| [execution-validation.md](execution-validation.md) | the rules for dispatching programs this project authored, and why they exist |
| [figures/](figures/) | the README's figures, drawn from the receipts by `tools/g17figures.py` |

The evidence record and the recon document are the research record as written. They cite files, tools and notes
that this curated release does not carry (the release keeps the receipts behind its public claims; see
[RELEASE.md](../RELEASE.md)); every such link now leads to an entry in [OMITTED.md](OMITTED.md) that names the target
and why it was left out. They use the project's earlier name, triad, in places.
