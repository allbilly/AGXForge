# Mesa AGX userspace compiler on M1 macOS

**Mesa 26.2.4 generated the code that passed all 30 GPU checks**, using existing
AGXForge IR and the existing macOS Metal executor. The measured machine is a
base M1 MacBook Air on macOS 27.0.1 build 26A434. No Linux DRM submission or
Asahi kernel driver is involved. Apple's compiler creates the carrier archive;
Mesa's substituted AGX bytes execute.

The checks cover 20 vector copy/add/subtract/multiply cases, four original-layout
GPT-2 Conv1D projections, three reductions and three GELU cases. Sizes include
31/33/65/129-thread tails and the complete GPT-2 **768→2304** Q/K/V projection
shape. All launches completed with intact guards. The large projection's
maximum absolute error against independent FP64 computation was
**1.612e-6**, within the declared `3e-6 + 1e-5·|reference|` tolerance.

That projection uses **14 register halfwords and 162 code bytes** from Mesa,
versus **34 halfwords and 204 bytes** from AGXForge's scalar G13 compiler.
Throughput was not benchmarked. The [summary](summary.json) records resource
counts and numerical errors for every case. All **79 hardware-free G13/macOS
checks** also passed; see [the test log](unit-checks.log).

[Run instructions](../../examples/macos/README.md#optional-mesa--asahi-compiler)
describe the optional builder and adapter. Mesa compiler sources were checked
against the SHA-256-pinned official release. A small Meson patch skips a
duplicate macOS `genxml` visit; the compiler implementation is unchanged.
The [build identity](mesa-build-identity.json) records all checked source hashes,
the adapter hash and the helper executable hash. The existing G13 compiler
still supplies conservative IR/CFG admission; Mesa supplies the executed bytes.

Run `python3 -S evidence/macos-mesa-v1/verify.py` to check the compact file index,
decode and validate all submitted shaders, tie them to the compiler records,
and recompute all numerical checks from the retained expected and GPU output
arrays. No Mesa rebuild or NumPy installation is required. Inputs and Metal
carrier archives remain in the complete local bundle under
`results/macos-mesa-v3`; [the receipt](receipt.json) records that bundle's
manifest hash and the compact omissions. This compact bundle verifies the
captured outputs against captured references; recreating the references and
dispatches requires the full verification example. Only trailing whitespace in
human-readable disassembly is normalized; shader tokens and binary bytes are unchanged.

This is a working scalar compiler integration, with a deliberately limited
ABI: 32-bit values and buffers, fixed pointer uniforms, no shader preamble,
scratch, shared memory or promoted constants. Full GPT-2/Qwen inference through
Mesa, direct IOGPU launches of Mesa code, FP16/BF16, tensor operations, other GPUs
and other OS builds remain unverified.
