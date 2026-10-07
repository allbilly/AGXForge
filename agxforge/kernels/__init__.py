"""The decode kernels: each one a layout, a builder that emits IR for agxforge.g17's compiler, and a reference.

A decoder layer of the quantized decode graph runs, in order:

    norm.build_rmsnorm_wide            attention RMSNorm (one row, one 1,024-thread threadgroup)
    qmv.build_qmv2                     the fused q, k, v projection (4- or 8-bit affine weights, split-K)
    attention.build_attn_split_rope    RoPE, the KV-cache append, the split softmax and its merge, one dispatch
    qmv.build_qmv2                     the output projection plus the first residual
    norm.build_rmsnorm_wide            FFN RMSNorm
    qmv.build_qmv2                     w1 and w3 with SwiGLU fused
    qmv.build_qmv2                     w2 plus the second residual

and after the last layer the final norm, the vocabulary projection, and argmax.build_pass1 / argmax.build_gen, which
choose the next token and write its embedding on the GPU, so a generation needs no host round trip between tokens.

Each kernel module holds its layout functions (where every input and output sits in the three buffers), its builder,
and a numerical reference that computes the kernel's value in the kernel's own fp32 operation order, so most kernels
are checked bit for bit. The exceptions are the forms that use the hardware exp2 (op1272, not reproducible on the CPU):
the hardware-exp2 attention, which the matched study's config selects, and the fast SwiGLU are checked within a stated
enclosure of the true value, with a wrong-base control that must fall outside it. The shared pieces:

    numerics    the fp32 arithmetic models the references use (ALU rounding, correctly rounded rsqrt and recip,
                exp2_soft, the MMA)
    emit        the IR helpers the builders share (constants, the rounding corrections, the carrier MMA)
    bundle      a compiled kernel as the bundle directory the executors load

The prompt runs first as a PREFILL section, M rows at a time per layer (agxforge.inference.prefill places it):

    norm (batched rows)              the RMSNorms over the chunk's rows
    qmm.build_dequant + gemm_generic each projection: the q4/q8 weights dequantized to fp16 once, then the compiler's
                                     tensor GEMM class (built by tools/g17tensorcommonruntime.py from qmm.role_spec)
    prefill_attention                the chunk's KV append (and the scalar attention route)
    prefill_mma.build_mma_rego       the attention on the tensor units, register-O, with the causal skip
    rows.build_rows                  the residual, SwiGLU and fold row kernels

agxforge.inference.assemble builds the model's graph of these kernels from a deliver index, the bundles a model config
selects; tools/g17deliver.py builds that index and verifies every bundle on the GPU before it is delivered.
"""
