"""The models the graph assembler builds for: each one's shapes, as agxforge.inference.assemble reads them.

A model is a record of upper-case shape attributes (D_MODEL, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB, ...), the
checkpoint's identity (MODEL_ID), where its graphs are written (OUT, under the checkout's results/), and layer_spec(n),
the decoder layer's shape at a KV length of n that the kernels' references take.

The assembler tells models apart by the attributes they carry, not by name: QK_NORM inserts Qwen3's per-head RMSNorm of
q and k, VOCAB_PAD pads the batched head's argmax, FFN_PREFILL widens the prefill's FFN. So each record carries exactly
the attributes its source module had (tools/g17realmodel.py for InternLM2, tools/g17qwen3.py for Qwen3), no more.

    INTERNLM2       InternLM2.5-1.8B-chat (MM 25.138)
    qwen3(name)     Qwen3-0.6B or Qwen3-8B (MM 25.182), from QWEN3
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from agxforge.kernels import numerics

# THE REPOSITORY ROOT: graphs are written under the checkout's results/ directory
_REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Model(SimpleNamespace):
    """One model's shapes. layer_spec(length): the decoder layer at KV length `length`."""

    def layer_spec(self, length):
        if self.ARCH == "internlm2":
            return numerics.LayerSpec(d_model=self.D_MODEL, n_heads=self.HEADS, head_dim=self.HEAD_DIM,
                                      ffn_dim=self.FFN, kv_len=length, storage="half", norm_eps=self.EPS,
                                      rope_base=self.ROPE_THETA, k_route=True, n_kv_heads=self.KV_HEADS)
        # numerics.LayerSpec requires n_heads x head_dim == d_model, which is InternLM2's shape and not Qwen3's (16 x
        # 128 = 2,048 against d_model 1,024); the delivered graph reads only d_model from it, so a plain record
        return SimpleNamespace(d_model=self.D_MODEL, n_heads=self.HEADS, head_dim=self.HEAD_DIM, ffn_dim=self.FFN,
                               kv_len=length, storage="half", norm_eps=self.EPS, rope_base=self.ROPE_THETA,
                               k_route=True, n_kv_heads=self.KV_HEADS, kv_heads=self.KV_HEADS)


_I_OUT = _REPO_ROOT / "results" / "g17-model-internlm2"
INTERNLM2 = Model(ARCH="internlm2", MODEL_ID="internlm/internlm2_5-1_8b-chat", OUT=_I_OUT, WEIGHTS=_I_OUT / "weights",
                  D_MODEL=2048, HEADS=16, KV_HEADS=8, HEAD_DIM=128, FFN=8192, LAYERS=24, VOCAB=92544, LM_BLOCK=8192,
                  VOCAB_PADDED=98304, ROPE_THETA=1.0e6, EPS=1.0e-5)
del _I_OUT

# THE QWEN3 FAMILY. qwen3-0.6b is byte-for-byte what MM 25.182-25.189 built; qwen3-8b has 32 query heads against 8 KV
# heads (GQA 4), d_model 4,096, 36 layers and an untied head.
QWEN3 = {
    "qwen3-0.6b": dict(D_MODEL=1024, LAYERS=28, HEADS=16, KV_HEADS=8, HEAD_DIM=128, FFN=3072, VOCAB=151936,
                       VOCAB_PAD=152064, FFN_PREFILL=4096, TIED=True, MODEL_ID="Qwen/Qwen3-0.6B", OUT="g17-model-qwen3"),
    "qwen3-8b": dict(D_MODEL=4096, LAYERS=36, HEADS=32, KV_HEADS=8, HEAD_DIM=128, FFN=12288, VOCAB=151936,
                     VOCAB_PAD=152064, FFN_PREFILL=16384, TIED=False, MODEL_ID="Qwen/Qwen3-8B", OUT="g17-model-qwen3-8b"),
}
QWEN3_EPS, QWEN3_ROPE_THETA = 1e-6, 1e6


def qwen3(name="qwen3-0.6b"):
    """The Qwen3 model `name` (a key of QWEN3), with the derived widths tools/g17qwen3.configure binds."""
    m = dict(QWEN3[name])
    m["OUT"] = _REPO_ROOT / "results" / m["OUT"]
    return Model(ARCH="qwen3", MODEL=name, EPS=QWEN3_EPS, ROPE_THETA=QWEN3_ROPE_THETA, QK_NORM=True,
                 QKV=(m["HEADS"] + 2 * m["KV_HEADS"]) * m["HEAD_DIM"], FFN_DIM=m["FFN"],
                 PAD=m["VOCAB_PAD"] - m["VOCAB"], **m)
