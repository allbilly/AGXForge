"""Model inference built from agxforge.kernels: from a model config to tokens on the GPU.

THE ROUTE, for one config (tools/models/internlm2_q4_spec_code.json is the matched study's, MM 25.211):

    1. the config        which variant of each kernel every role uses (qmv: {"qkv": {"ks": 2, "lean": true, ...}},
                         "norm", "attn", "gen"), the KV capacity, the prompt, and the prefill section's route
    2. catalog           each variant's layout: catalog.qmv_layout(4, "qkv", variant) and the rest
    3. kernels           agxforge.kernels builds each program from its layout (qmv.build_qmv2, attention.
                         build_attn_split_rope, prefill_mma.build_mma_rego, ...), compiled by agxforge.g17
    4. verify, deliver   tools/g17deliver.py builds every bundle with test inputs, dispatches it once on the GPU over a
                         sentinel, and delivers it only if it passes its declared numerical check: bit-exact against its
                         kernel's reference, or, for the hardware-exp2 attention the study's config selects (and the
                         fast SwiGLU), an enclosure with a wrong-base control that must fail. index.deliver puts it in a
                         content-addressed deliver root; the root's index.json (index.SCHEMA) lists it and records the
                         check it passed
    5. assemble          assemble.configure(root, config); assemble.build() selects every dispatch from the index,
                         solves the arena offsets (graph.Solver), adds the prefill section (prefill.section) and writes
                         graph.json and the weight arenas; models gives the model's shapes
    6. execute           execute.decode(graph.json, tokens, out) runs it on the Metal executor (tools/g17decodegen)
    7. compare           assemble.simulate() runs the decode graph on the CPU, every dispatch by its kernel's reference,
                         and tools/g17modelbuild.py requires the GPU's tokens to equal the simulator's; the matched
                         study's comparison with mlx-lm and with Apple-compiled twins is tools/g17twin.py

The commands for steps 4 to 7 are tools/g17deliver.py build, tools/g17q4graph.py build | simulate and
tools/g17modelbuild.py, which runs build, simulate and the GPU check in one.

    models      each model's shapes (InternLM2.5-1.8B, the Qwen3 family)
    catalog     which form of each kernel a model uses, and its layout
    index       the deliver index's contract, and content-addressed delivery
    assemble    the model's dispatch graph from a deliver index and a config, and its CPU simulator
    prefill     the graph's prefill section: the prompt as M-row passes, run once before decode
    graph       the dispatch (Op), the arena offset solver and the simulator's arena stepping
    execute     running a built graph on the Metal executor (the only module here that dispatches GPU work)
"""
