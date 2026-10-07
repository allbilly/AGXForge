"""Persistent native Qwen2 scalar execution. CPU work here prepares resources only."""
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
from . import kernels as k


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.identity = json.loads((self.directory / "checkpoint.json").read_text())
        self.path = self.directory / "model.safetensors"
        with self.path.open("rb") as file:
            self.mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        length, = struct.unpack_from("<Q", self.mapping)
        if length > len(self.mapping)-8: raise ValueError("truncated safetensors header")
        self.offset = length+8
        self.tensors = json.loads(self.mapping[8:self.offset])
        self.tensors.pop("__metadata__", None)
        for name, tensor in self.tensors.items():
            lo, hi = tensor["data_offsets"]
            unit = {"BF16": 2, "F16": 2, "F32": 4}.get(tensor["dtype"])
            if unit is None or lo < 0 or hi < lo or self.offset+hi > len(self.mapping) or hi-lo != math.prod(tensor["shape"])*unit:
                raise ValueError(f"invalid tensor extent/dtype {name}")

    def raw(self, name):
        lo, hi = self.tensors[name]["data_offsets"]
        return self.mapping[self.offset+lo:self.offset+hi]

    def floats(self, name, first=None, last=None):
        """Reference-only decoding. The executor uploads raw BF16 without calling this."""
        import numpy as np
        tensor = self.tensors[name]
        shape = tensor["shape"]
        dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}[tensor["dtype"]]
        lo, _ = tensor["data_offsets"]
        a = np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=self.offset+lo)
        if first is not None: a = a[first:last]
        if tensor["dtype"] == "BF16": return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return a.astype(np.float64)

    def receipt(self):
        result = dict(self.identity)
        result["files"] = {}
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            path = self.directory / name
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(8*1024*1024), b""): digest.update(chunk)
            result["files"][name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
        return result


class QwenPlan:
    def __init__(self, checkpoint, capacity=32):
        c = checkpoint.config
        if c["model_type"] != "qwen2" or c.get("hidden_act") != "silu" or c.get("use_sliding_window") or c.get("rope_scaling"):
            raise ValueError("only the Qwen2 scalar architecture without sliding windows or RoPE scaling is supported")
        if not c.get("tie_word_embeddings"):
            raise ValueError("this first profile requires tied word embeddings")
        self.d, self.ffn, self.vocab = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
        self.heads, self.kvh, self.layers = c["num_attention_heads"], c["num_key_value_heads"], c["num_hidden_layers"]
        self.hd = self.d // self.heads
        if self.heads % self.kvh or self.d != self.heads*self.hd or self.hd & (self.hd-1):
            raise ValueError("unsupported head geometry")
        if not 1 <= capacity <= 1024 or capacity & (capacity-1) or capacity > c["max_position_embeddings"]:
            raise ValueError("capacity must be a power of two in 1..1024 within the model position limit")
        self.capacity, self.eps, self.theta = capacity, c["rms_norm_eps"], c["rope_theta"]
        self.names = {"model.embed_tokens.weight": [self.vocab, self.d], "model.norm.weight": [self.d]}
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            self.names[prefix+"input_layernorm.weight"] = [self.d]
            self.names[prefix+"post_attention_layernorm.weight"] = [self.d]
            for role, width in (("q", self.d), ("k", self.kvh*self.hd), ("v", self.kvh*self.hd)):
                self.names[prefix+f"self_attn.{role}_proj.weight"] = [width, self.d]
                self.names[prefix+f"self_attn.{role}_proj.bias"] = [width]
            self.names[prefix+"self_attn.o_proj.weight"] = [self.d, self.d]
            for role in ("gate", "up"): self.names[prefix+f"mlp.{role}_proj.weight"] = [self.ffn, self.d]
            self.names[prefix+"mlp.down_proj.weight"] = [self.d, self.ffn]
        for name, shape in self.names.items():
            tensor = checkpoint.tensors.get(name)
            if not tensor or tensor["shape"] != shape or tensor["dtype"] != "BF16":
                raise ValueError(f"checkpoint does not match BF16 profile: {name}")
        builders = {
            "embedding": k.embedding(self.vocab, self.d), "sum_squares": k.reduce_rows(1, self.d, "squares"),
            "rms_scale": k.norm_scale(self.d, self.eps), "rms_apply": k.norm_apply(self.d),
            "q": k.gemv(self.d, self.d, bias=True), "kv": k.gemv(self.kvh*self.hd, self.d, bias=True),
            "rope_q": k.rope(self.heads, self.hd), "rope_k": k.rope(self.kvh, self.hd),
            "kv_write": k.kv_write(self.kvh*self.hd, capacity),
            "scores": k.scores(self.heads, self.kvh, self.hd, capacity),
            "softmax_max": k.reduce_rows(self.heads, capacity, "max"),
            "softmax_exp": k.softmax_element(self.heads, capacity, "exp"),
            "softmax_sum": k.reduce_rows(self.heads, capacity, "sum"),
            "softmax_normalize": k.softmax_element(self.heads, capacity, "normalize"),
            "attend": k.attend(self.heads, self.kvh, self.hd, capacity),
            "o": k.gemv(self.d, self.d), "residual": k.vector("add", self.d),
            "gate_up": k.gemv(self.ffn, self.d), "swiglu": k.vector("swiglu", self.ffn),
            "down": k.gemv(self.d, self.ffn), "logits": k.gemv(self.vocab, self.d), "argmax": k.argmax(self.vocab)
        }
        # Compile EVERY selected kernel before any GPU/model allocation.
        self.programs = {name: spec.compile() for name, spec in builders.items()}
        self.requirements = {name: sorted(spec.required_capabilities()) for name, spec in builders.items()}

    def descriptor(self):
        return dict(architecture="qwen2", capacity=self.capacity, tensor_operations="GPU only",
                    prefill="token by token through decode kernels", requirements=self.requirements,
                    programs={name: program.descriptor() for name, program in self.programs.items()})


class Qwen:
    def __init__(self, executor, checkpoint, plan, evidence=None):
        self.gpu, self.plan, self.evidence = executor, plan, Path(evidence) if evidence else None
        self.position, self.dispatches = 0, 0
        self.weights = {}
        for name in plan.names:
            raw = checkpoint.raw(name)
            self.weights[name] = executor.buffer(len(raw), "read", raw)
        self.params = executor.buffer(8, "read")
        # Static model constants, independent of activations and token positions.
        self.invfreq = executor.buffer(plan.hd//2*4, "read", struct.pack("<" + "f"*(plan.hd//2),
            *(plan.theta**(-2*j/plan.hd) for j in range(plan.hd//2))))
        self.headmap = executor.buffer(plan.heads*4, "read", struct.pack("<" + "I"*plan.heads,
            *(h//(plan.heads//plan.kvh) for h in range(plan.heads))))
        d, f, kv, hc = plan.d, plan.ffn, plan.kvh*plan.hd, plan.heads*plan.capacity
        sizes = dict(hidden=d*4, normalized=d*4, sum=4, scale=4, q=d*4, k=kv*4, v=kv*4,
                     rotated_q=d*4, rotated_k=kv*4, scores=hc*4, maximum=plan.heads*4, denom=plan.heads*4,
                     exps=hc*4, probs=hc*4, attention=d*4, projection=d*4, residual=d*4,
                     gate=f*4, up=f*4, activation=f*4, down=d*4, logits=plan.vocab*4, argmax=4)
        self.temp = {name: executor.buffer(size) for name, size in sizes.items()}
        size = plan.capacity*kv*4
        self.cache = [(executor.buffer(size, data=bytes(size)), executor.buffer(size, data=bytes(size))) for _ in range(plan.layers)]

    def call(self, name, buffers, label):
        self.dispatches += 1
        folder = self.evidence / f"{self.dispatches:06d}-{label}" if self.evidence else None
        p = self.plan.programs[name]
        return self.gpu.dispatch(p, buffers, p.logical_threads, evidence=folder, capture_buffers=False)

    def normalize(self, x, gain, label):
        t = self.temp
        self.call("sum_squares", [t["sum"], x], label+"-squares")
        self.call("rms_scale", [t["scale"], t["sum"]], label+"-scale")
        self.call("rms_apply", [t["normalized"], x, gain, t["scale"]], label+"-apply")
        return t["normalized"]

    def consume(self, token, observer=None):
        p, t, w = self.plan, self.temp, self.weights
        if type(token) is not int or not 0 <= token < p.vocab or self.position >= p.capacity:
            raise ValueError("token/position outside model and KV extents")
        self.params.write(struct.pack("<II", token, self.position))
        def watch(name, buf):
            if observer: observer(name, buf.read())
        self.call("embedding", [t["hidden"], w["model.embed_tokens.weight"], self.params], "embedding")
        watch("embedding", t["hidden"])
        for layer in range(p.layers):
            prefix = f"model.layers.{layer}."; label = f"layer-{layer}"
            x = t["hidden"]
            norm = self.normalize(x, w[prefix+"input_layernorm.weight"], label+"-norm1")
            watch(label+"-norm1", norm)
            for role in ("q", "k", "v"):
                self.call("q" if role == "q" else "kv", [t[role], norm,
                    w[prefix+f"self_attn.{role}_proj.weight"], w[prefix+f"self_attn.{role}_proj.bias"]], label+"-"+role)
                watch(label+"-"+role, t[role])
            self.call("rope_q", [t["rotated_q"], t["q"], self.invfreq, self.params], label+"-rope-q")
            self.call("rope_k", [t["rotated_k"], t["k"], self.invfreq, self.params], label+"-rope-k")
            watch(label+"-rope-q", t["rotated_q"]); watch(label+"-rope-k", t["rotated_k"])
            kc, vc = self.cache[layer]
            self.call("kv_write", [kc, t["rotated_k"], self.params], label+"-cache-k")
            self.call("kv_write", [vc, t["v"], self.params], label+"-cache-v")
            watch(label+"-cache-k", kc); watch(label+"-cache-v", vc)
            self.call("scores", [t["scores"], t["rotated_q"], kc, self.headmap, self.params], label+"-scores")
            watch(label+"-scores", t["scores"])
            self.call("softmax_max", [t["maximum"], t["scores"]], label+"-softmax-max")
            self.call("softmax_exp", [t["exps"], t["scores"], t["maximum"]], label+"-softmax-exp")
            self.call("softmax_sum", [t["denom"], t["exps"]], label+"-softmax-sum")
            self.call("softmax_normalize", [t["probs"], t["exps"], t["denom"]], label+"-softmax-normalize")
            watch(label+"-probs", t["probs"])
            self.call("attend", [t["attention"], t["probs"], vc, self.headmap], label+"-attend")
            watch(label+"-attention", t["attention"])
            self.call("o", [t["projection"], t["attention"], w[prefix+"self_attn.o_proj.weight"]], label+"-o")
            watch(label+"-o", t["projection"])
            self.call("residual", [t["residual"], x, t["projection"]], label+"-residual1")
            watch(label+"-residual1", t["residual"])
            norm = self.normalize(t["residual"], w[prefix+"post_attention_layernorm.weight"], label+"-norm2")
            watch(label+"-norm2", norm)
            for role in ("gate", "up"):
                self.call("gate_up", [t[role], norm, w[prefix+f"mlp.{role}_proj.weight"]], label+"-"+role)
                watch(label+"-"+role, t[role])
            self.call("swiglu", [t["activation"], t["gate"], t["up"]], label+"-swiglu")
            watch(label+"-activation", t["activation"])
            self.call("down", [t["down"], t["activation"], w[prefix+"mlp.down_proj.weight"]], label+"-down")
            watch(label+"-down", t["down"])
            self.call("residual", [t["hidden"], t["residual"], t["down"]], label+"-residual2")
            watch(label+"-residual2", t["hidden"])
        norm = self.normalize(t["hidden"], w["model.norm.weight"], "final-norm")
        watch("final-norm", norm)
        self.call("logits", [t["logits"], norm, w["model.embed_tokens.weight"]], "logits")
        watch("logits", t["logits"])
        self.call("argmax", [t["argmax"], t["logits"]], "argmax")
        result, = struct.unpack("<I", t["argmax"].read())
        self.position += 1
        return result
