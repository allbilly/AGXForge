# tools/

Command-line entry points and the modules they import. `tools/` is on the import path of the code that uses it, so its
files keep their research names (`g17*.py`); most of the ~250 here are imported by the few entry points below, not
run directly. Start from the [example workflows](../examples/README.md), which call the right ones.

| To | Entry point |
|---|---|
| Build the native tools | the [Makefile](../Makefile): [`agx3dis.c`](agx3dis.c) (Apple's decoder, loaded from `GPUCompiler.framework`), [`g17decodegen.m`](g17decodegen.m) (the Metal executor), [`g17twinrun.m`](g17twinrun.m), [`g17scanworker.m`](g17scanworker.m) |
| Compare this machine with the measured one | [`g17platform.py`](g17platform.py) `--check` (or `make platform`) |
| Run the retained tests | [`release_tests.py`](release_tests.py) (or `make test`) |
| Run a program on the CPU from its bytes | [`g17emu.py`](g17emu.py), the emulator (its framing uses Apple's decoder) |
| Reproduce the matched decode study (GPU) | [`g17deliver.py`](g17deliver.py), [`g17modelbuild.py`](g17modelbuild.py), [`g17twin.py`](g17twin.py) with the twins in [`twins/`](twins/), [`g17forcehistory.py`](g17forcehistory.py), [`g17promptids.py`](g17promptids.py) |
| Reproduce MiniLM and Qwen below Metal (GPU) | [`g17checkpoint.py`](g17checkpoint.py), [`g17inferencelower.py`](g17inferencelower.py), [`g17decoderlower.py`](g17decoderlower.py), [`g17inferenceprepare.py`](g17inferenceprepare.py), [`g17inferencebundle.py`](g17inferencebundle.py), [`g17nativegenerate.py`](g17nativegenerate.py), [`g17nativeretrieval.py`](g17nativeretrieval.py), and the independent checks [`g17generationcheck.py`](g17generationcheck.py), [`g17retrievalcheck.py`](g17retrievalcheck.py) |
| Measure Apple's int8 `matmul2d` baseline (GPU) | [`g17mppint8.py`](g17mppint8.py) |
| Take the machine-wide GPU lock | [`g17gpulock.py`](g17gpulock.py), [`g17gpulock.h`](g17gpulock.h): every dispatching tool takes it |
| Regenerate the figures | [`g17figures.py`](g17figures.py) |

The decode study's kernels and graph assembly live in the package, [`agxforge/kernels/`](../agxforge/kernels/__init__.py)
and [`agxforge/inference/`](../agxforge/inference/__init__.py). Their old modules here (`g17qmv.py`, `g17attn.py`,
`g17gen.py`, `g17q4graph.py`, `g17prefillgraph.py`, `g17prefillmma.py`, `g17rows.py`, ...) are compatibility entries:
each keeps its command line and forwards everything else to the package. `g17deliver.py` builds each bundle with its
verification inputs and checks it on the GPU before delivering it.

The rest: the stages of the native model pipeline (`g17inference*.py`, `g17resident*.py`, `g17native*.py`), the arithmetic
references the tests and workflow 2 use (`g17tensorcommonruntime.py`), and the helpers those import.
