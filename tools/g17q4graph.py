#!/usr/bin/env python3
"""THE QUANTIZED CHAINED TOKEN (docs/g17-tensorops-machine-model.md 25.138.5): InternLM2.5-1.8B at 4 (or 8) bits,
mlx-lm's own affine group-64 weights, as one ordered dispatch list for Set C's executor (tools/g17decodegen.m).

    python3 tools/g17q4graph.py prepare --bits 4    (system python3 with MLX, its own process) the mlx checkpoint's
                                                    tensors as npy in our row order, and the dequantized embedding
    python3 tools/g17q4graph.py build --bits 4 --deliver DIR --config tools/models/internlm2_q4_best.json
                                                    graph.json + weight arenas
    python3 tools/g17q4graph.py simulate --bits 4 --deliver DIR --config FILE
                                                    the graph on the CPU, every dispatch by its exact-order reference
    (tools/g17modelbuild.py runs build, simulate and the GPU check in one command.)

A LAYER is 7 dispatches (the fused, device-resident graph; the 9-dispatch form below was the first, 25.138.5):
  attn_norm -> qkv qmv -> attention (RoPE, append, split and merge in one dispatch, 25.140.5) ->
  wo qmv + residual1 -> ffn_norm -> w1+w3+SwiGLU qmv -> w2 qmv + residual2
[Corrected 2026-09-26: this said 9 dispatches: attn_norm -> qkv -> RoPE append -> split -> merge -> wo -> ffn_norm ->
ffn -> w2. That route and its fixed bundle directories are gone; see the note above DELIVER_ROOT.]
ONE region R of 12,288 bytes carries the residual stream through every layer: h fp32 [2048] at R + 0 (written by
wo + residual1, read by ffn_norm and w2 + residual2), x fp16 [2048] at R + 8,192 (written by w2 + residual2 as the
next layer's input, read by attn_norm and wo + residual1; the embedding row is written there for layer 0). The head
is the final norm and ONE lm_head qmv (92,544 rows).

WEIGHTS are mlx_lm.convert's -q output as it is (W uint32 low field first, scales and biases bf16, group 64). Only
rows move: wqkv from the checkpoint's per-KV-group packing [q 2g, q 2g+1, k g, v g] to q heads 0..15 | k 0..7 |
v 0..7, and w1 and w3 concatenated for the fused SwiGLU. A row permutation moves each row's scales and biases
with it, so no value changes. The embedding is mx.dequantize of the quantized table (MLX's own embedding values),
stored fp16.

THE ASSEMBLER MOVED to agxforge.inference.assemble. This file keeps the command line above and `prepare`, and forwards
every other attribute to the package module, so `import g17q4graph` keeps working.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import os as _os
import sys as _sys

# THE REPOSITORY IMPORT ROOT COMES FIRST, ahead of the package import: run by absolute path from another directory
# with PYTHONPATH unset, nothing else puts the checkout on sys.path.
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.inference import assemble as _impl  # noqa: E402
from agxforge.g17 import compat as _compat  # noqa: E402

_compat.install(__name__, _impl)


def configure(deliver, config):
    """agxforge.inference.assemble.configure, and for a Qwen3 config also tools/g17qwen3.configure: that module's shape
    constants are what the deliverer's references read, and before the move configuring the graph bound them."""
    _impl.configure(deliver, config)
    if _impl.CONFIG.get("arch", "internlm2") == "qwen3":
        _sys.path.insert(0, _os.path.join(_REPO_ROOT, "tools"))
        import g17qwen3
        g17qwen3.configure(_impl.CONFIG.get("model", "qwen3-0.6b"))


def mlx_dir(bits):
    return Path.home() / "models" / ("internlm2_5-1_8b-chat-mlx-q%d" % bits)


def _qkv_rows():
    """new row -> checkpoint row, for the packed wqkv (4,096 rows)."""
    groups = _impl.M.HEADS // _impl.M.KV_HEADS
    old = np.arange(4096).reshape(_impl.M.KV_HEADS, groups + 2, _impl.M.HEAD_DIM)
    q = old[:, :groups].reshape(-1)
    return np.concatenate([q, old[:, groups].reshape(-1), old[:, groups + 1].reshape(-1)])


def prepare(bits):
    import mlx.core as mx
    src = mlx_dir(bits)
    t = {}
    for f in sorted(src.glob("*.safetensors")):
        t.update(mx.load(str(f)))
    W = _impl.weights_dir(bits)
    W.mkdir(parents=True, exist_ok=True)

    def u16(a):
        return np.array(a.view(mx.uint16))

    def trio(prefix):
        return np.array(t[prefix + ".weight"]), u16(t[prefix + ".scales"]), u16(t[prefix + ".biases"])

    perm = _qkv_rows()
    for l in range(_impl.M.LAYERS):
        p = "model.layers.%d." % l
        w, s, b = trio(p + "attention.wqkv")
        np.savez(W / ("L%d_qkv.npz" % l), W=w[perm], S=s[perm], B=b[perm])
        np.savez(W / ("L%d_wo.npz" % l), **dict(zip("WSB", trio(p + "attention.wo"))))
        w1, s1, b1 = trio(p + "feed_forward.w1")
        w3, s3, b3 = trio(p + "feed_forward.w3")
        np.savez(W / ("L%d_ffn.npz" % l), W=np.concatenate([w1, w3]), S=np.concatenate([s1, s3]),
                 B=np.concatenate([b1, b3]))
        np.savez(W / ("L%d_w2.npz" % l), **dict(zip("WSB", trio(p + "feed_forward.w2"))))
        for name, key in (("g1", "attention_norm"), ("g2", "ffn_norm")):
            np.save(W / ("L%d_%s.npy" % (l, name)), np.array(t[p + key + ".weight"].astype(mx.float16)))
    np.save(W / "norm.npy", np.array(t["model.norm.weight"].astype(mx.float16)))
    np.savez(W / "lm.npz", **dict(zip("WSB", trio("output"))))
    e = t["model.tok_embeddings.weight"]
    emb = mx.dequantize(e, t["model.tok_embeddings.scales"], t["model.tok_embeddings.biases"], group_size=64,
                        bits=bits)
    np.save(W / "embed.npy", np.array(emb.astype(mx.float16)))
    _write_base_record()
    print("prepared", W)


# THE BASE RECORD the assembler reads (M.OUT/graph/graph.json): the default prompt for a config that names none, the
# rope tables and the notes. tools/g17modelgraph.py's fp16 graph used to be its only writer, so a checkout that had
# never built that graph could not assemble a q4 graph; prepare writes the same fields, and never replaces a record
# that exists. Qwen3's prepare writes its own (tools/g17qwen3.py).
DEFAULT_PROMPT = [1, 918, 11498, 2327, 1197, 48304, 416]


def _write_base_record():
    G = _impl.M.OUT / "graph"
    if (G / "graph.json").exists():
        return
    G.mkdir(parents=True, exist_ok=True)
    pos = np.arange(272)[:, None] * (_impl.M.ROPE_THETA ** (-np.arange(0, 128, 2, dtype=np.float64) / 128))[None, :]
    np.savez(G / "rope_tables.npz", cos=np.cos(pos).astype(np.float32), sin=np.sin(pos).astype(np.float32))
    (G / "rope_cos.f32").write_bytes(np.cos(pos).astype("<f4").tobytes())       # raw [272 x 64]
    (G / "rope_sin.f32").write_bytes(np.sin(pos).astype("<f4").tobytes())
    (G / "graph.json").write_text(json.dumps(dict(
        model=_impl.M.MODEL_ID, prompt_ids=DEFAULT_PROMPT,
        tables=dict(embedding=str(_impl.weights_dir(4) / "embed.npy"), rope=str(G / "rope_tables.npz")),
        rope_cos_file=str(G / "rope_cos.f32"), rope_sin_file=str(G / "rope_sin.f32"),
        notes="embedding_row: fp16 row of the embedding table at the token (prompt tokens first, then the "
              "argmax of the previous logits over the first `vocab`); rope_*_row: row kv_len of the rope table; "
              "len words: kv_len (the new token's position). One command buffer per token."), indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "build", "simulate"))
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--deliver", type=Path, help="the deliver root (tools/g17deliver.py build --out); build and simulate")
    ap.add_argument("--config", type=Path, help="the model config (tools/models/*.json); build and simulate")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return prepare(a.bits)
    if (a.deliver is None) != (a.config is None):
        ap.error("--deliver and --config go together")
    if a.deliver is not None:
        configure(a.deliver, a.config)
    _impl._require()
    if a.cmd == "build":
        _impl.build(a.bits)
        return 0
    _impl.simulate(a.bits, a.tokens)
    return 0


if os.environ.get("G17_Q4_DELIVER"):
    # DEPRECATED (2026-09-26): the environment interface, still honoured for callers that import this module after
    # setting it (tools/g17prefillhead.py). Pass --deliver / --config instead.
    print("g17q4graph: G17_Q4_DELIVER / G17_Q4_CONFIG are deprecated; pass --deliver DIR --config FILE",
          file=sys.stderr)
    if not os.environ.get("G17_Q4_CONFIG"):
        raise SystemExit("g17q4graph: G17_Q4_DELIVER is set without G17_Q4_CONFIG")
    configure(os.environ["G17_Q4_DELIVER"], os.environ["G17_Q4_CONFIG"])


if __name__ == "__main__":
    sys.exit(main())
