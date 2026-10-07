"""Persistent FP32 GPT-2 inference using only authored G13 tensor programs."""
from pathlib import Path
import struct
from . import kernels as k
from . import gpt2_kernels as g


class GPT2Plan:
    def __init__(self, checkpoint, capacity=32):
        c = checkpoint.config
        if (c.get("model_type") != "gpt2" or c.get("activation_function") != "gelu_new"
                or c.get("add_cross_attention", False) or not c.get("tie_word_embeddings", True)
                or not c.get("scale_attn_weights", True) or c.get("scale_attn_by_inverse_layer_idx", False)
                or c.get("reorder_and_upcast_attn", False)):
            raise ValueError("only tied, self-attention GPT-2 with gelu_new and standard attention scaling is supported")
        self.d, self.heads, self.layers = c["n_embd"], c["n_head"], c["n_layer"]
        self.ffn, self.vocab = c.get("n_inner") or 4*self.d, c["vocab_size"]
        self.positions, self.eps = c["n_positions"], c["layer_norm_epsilon"]
        if min(self.d, self.heads, self.layers, self.ffn, self.vocab, self.positions) <= 0:
            raise ValueError("positive GPT-2 dimensions required")
        self.hd = self.d // self.heads
        if self.d != self.heads*self.hd or self.hd & (self.hd-1):
            raise ValueError("head dimension must be a power of two")
        if type(capacity) is not int or not 1 <= capacity <= min(1024, self.positions) or capacity & (capacity-1):
            raise ValueError("capacity must be a power of two within 1..1024 and the position table")
        self.capacity = capacity
        # The public checkpoint uses GPT2Model names; LMHead checkpoints may retain this prefix.
        self.prefix = "" if "wte.weight" in checkpoint.tensors else "transformer."
        self.names = {self.prefix+"wte.weight": [self.vocab, self.d],
                      self.prefix+"wpe.weight": [self.positions, self.d],
                      self.prefix+"ln_f.weight": [self.d], self.prefix+"ln_f.bias": [self.d]}
        for layer in range(self.layers):
            prefix = self.prefix+f"h.{layer}."
            for norm in ("ln_1", "ln_2"):
                for kind in ("weight", "bias"): self.names[prefix+norm+"."+kind] = [self.d]
            for role, inputs, outputs in (("attn.c_attn", self.d, 3*self.d), ("attn.c_proj", self.d, self.d),
                                          ("mlp.c_fc", self.d, self.ffn), ("mlp.c_proj", self.ffn, self.d)):
                self.names[prefix+role+".weight"] = [inputs, outputs]
                self.names[prefix+role+".bias"] = [outputs]
        for name, shape in self.names.items():
            tensor = checkpoint.tensors.get(name)
            if not tensor or tensor["shape"] != shape or tensor["dtype"] != "F32":
                raise ValueError(f"checkpoint does not match FP32 GPT-2 profile: {name}")
        if "lm_head.weight" in checkpoint.tensors:
            head = checkpoint.tensors["lm_head.weight"]
            if head["shape"] != [self.vocab, self.d] or head["dtype"] != "F32" or checkpoint.raw("lm_head.weight") != checkpoint.raw(self.prefix+"wte.weight"):
                raise ValueError("lm_head must match the tied token embedding")
        builders = {
            "embedding": k.embedding(self.vocab, self.d, "f32"),
            "position_embedding": g.position_embedding(self.positions, self.d),
            "sum": k.reduce_rows(1, self.d), "norm_stats": g.layer_norm_stats(self.d, self.eps),
            "norm_apply": g.layer_norm_apply(self.d), "qkv": g.conv1d(self.d, 3*self.d),
            "split": g.split_qkv(self.d), "kv_write": k.kv_write(self.d, capacity),
            "scores": k.scores(self.heads, self.heads, self.hd, capacity),
            "softmax_max": k.reduce_rows(self.heads, capacity, "max"),
            "softmax_exp": k.softmax_element(self.heads, capacity, "exp"),
            "softmax_sum": k.reduce_rows(self.heads, capacity),
            "softmax_normalize": k.softmax_element(self.heads, capacity, "normalize"),
            "attend": k.attend(self.heads, self.heads, self.hd, capacity),
            "o": g.conv1d(self.d, self.d), "residual": k.vector("add", self.d),
            "up": g.conv1d(self.d, self.ffn), "gelu": g.gelu(self.ffn),
            "down": g.conv1d(self.ffn, self.d), "logits": k.gemv(self.vocab, self.d, "f32"),
            "argmax": k.argmax(self.vocab)
        }
        self.programs = {name: spec.compile() for name, spec in builders.items()}
        # Reject programs outside the measured macOS ABI before any model allocation.
        if any(p.register_halfs > 80 or len(p.bindings) > 8 for p in self.programs.values()):
            raise ValueError("GPT-2 kernels exceed the measured macOS register/binding profile")
        self.requirements = {name: sorted(spec.required_capabilities()) for name, spec in builders.items()}

    def descriptor(self):
        return dict(architecture="gpt2", capacity=self.capacity, tensor_operations="GPU only",
                    storage="FP32, original Conv1D layout", prefill="token by token through decode kernels",
                    requirements=self.requirements,
                    programs={name: program.descriptor() for name, program in self.programs.items()})


class GPT2:
    def __init__(self, executor, checkpoint, plan, evidence=None):
        self.gpu, self.plan, self.evidence = executor, plan, Path(evidence) if evidence else None
        self.position, self.dispatches = 0, 0
        self.weights = {}
        for name in plan.names:
            raw = checkpoint.raw(name)
            self.weights[name] = executor.buffer(len(raw), "read", raw)
        self.params = executor.buffer(8, "read")
        self.headmap = executor.buffer(plan.heads*4, "read", struct.pack("<"+"I"*plan.heads, *range(plan.heads)))
        d, f, hc = plan.d, plan.ffn, plan.heads*plan.capacity
        sizes = dict(embedding=d*4, position_embedding=d*4, hidden=d*4, normalized=d*4, sum=4, stats=8,
                     qkv=3*d*4, q=d*4, k=d*4, v=d*4, scores=hc*4, maximum=plan.heads*4,
                     denom=plan.heads*4, exps=hc*4, probs=hc*4, attention=d*4, projection=d*4,
                     residual=d*4, up=f*4, activation=f*4, down=d*4, logits=plan.vocab*4, argmax=4)
        self.temp = {name: executor.buffer(size) for name, size in sizes.items()}
        size = plan.capacity*d*4
        self.cache = [(executor.buffer(size, data=bytes(size)), executor.buffer(size, data=bytes(size))) for _ in range(plan.layers)]

    def call(self, name, buffers, label):
        self.dispatches += 1
        folder = self.evidence/f"{self.dispatches:06d}-{label}" if self.evidence else None
        p = self.plan.programs[name]
        return self.gpu.dispatch(p, buffers, p.logical_threads, evidence=folder, capture_buffers=False)

    def normalize(self, x, prefix, label):
        t, w = self.temp, self.weights
        self.call("sum", [t["sum"], x], label+"-sum")
        self.call("norm_stats", [t["stats"], x, t["sum"]], label+"-stats")
        self.call("norm_apply", [t["normalized"], x, w[prefix+".weight"], w[prefix+".bias"], t["stats"]], label+"-apply")
        return t["normalized"]

    def consume(self, token, observer=None):
        p, t, w = self.plan, self.temp, self.weights
        if type(token) is not int or not 0 <= token < p.vocab or self.position >= p.capacity:
            raise ValueError("token/position outside model and KV extents")
        self.params.write(struct.pack("<II", token, self.position))
        def watch(name, buf):
            if observer: observer(name, buf.read())
        self.call("embedding", [t["embedding"], w[p.prefix+"wte.weight"], self.params], "embedding")
        watch("embedding", t["embedding"])
        self.call("position_embedding", [t["position_embedding"], w[p.prefix+"wpe.weight"], self.params], "position-embedding")
        watch("position-embedding", t["position_embedding"])
        self.call("residual", [t["hidden"], t["embedding"], t["position_embedding"]], "residual-embedding")
        watch("residual-embedding", t["hidden"])
        for layer in range(p.layers):
            prefix, label = p.prefix+f"h.{layer}.", f"layer-{layer}"
            norm = self.normalize(t["hidden"], prefix+"ln_1", label+"-norm1")
            watch(label+"-norm1", norm)
            self.call("qkv", [t["qkv"], norm, w[prefix+"attn.c_attn.weight"], w[prefix+"attn.c_attn.bias"]], label+"-qkv")
            watch(label+"-qkv", t["qkv"])
            self.call("split", [t["q"], t["k"], t["v"], t["qkv"]], label+"-split")
            for role in ("q", "k", "v"): watch(label+"-"+role, t[role])
            kc, vc = self.cache[layer]
            self.call("kv_write", [kc, t["k"], self.params], label+"-cache-k")
            self.call("kv_write", [vc, t["v"], self.params], label+"-cache-v")
            watch(label+"-cache-k", kc); watch(label+"-cache-v", vc)
            self.call("scores", [t["scores"], t["q"], kc, self.headmap, self.params], label+"-scores")
            watch(label+"-scores", t["scores"])
            self.call("softmax_max", [t["maximum"], t["scores"]], label+"-softmax-max")
            self.call("softmax_exp", [t["exps"], t["scores"], t["maximum"]], label+"-softmax-exp")
            self.call("softmax_sum", [t["denom"], t["exps"]], label+"-softmax-sum")
            self.call("softmax_normalize", [t["probs"], t["exps"], t["denom"]], label+"-softmax-normalize")
            watch(label+"-probs", t["probs"])
            self.call("attend", [t["attention"], t["probs"], vc, self.headmap], label+"-attend")
            watch(label+"-attention", t["attention"])
            self.call("o", [t["projection"], t["attention"], w[prefix+"attn.c_proj.weight"], w[prefix+"attn.c_proj.bias"]], label+"-o")
            watch(label+"-o", t["projection"])
            self.call("residual", [t["residual"], t["hidden"], t["projection"]], label+"-residual1")
            watch(label+"-residual1", t["residual"])
            norm = self.normalize(t["residual"], prefix+"ln_2", label+"-norm2")
            watch(label+"-norm2", norm)
            self.call("up", [t["up"], norm, w[prefix+"mlp.c_fc.weight"], w[prefix+"mlp.c_fc.bias"]], label+"-up")
            watch(label+"-up", t["up"])
            self.call("gelu", [t["activation"], t["up"]], label+"-gelu")
            watch(label+"-activation", t["activation"])
            self.call("down", [t["down"], t["activation"], w[prefix+"mlp.c_proj.weight"], w[prefix+"mlp.c_proj.bias"]], label+"-down")
            watch(label+"-down", t["down"])
            self.call("residual", [t["hidden"], t["residual"], t["down"]], label+"-residual2")
            watch(label+"-residual2", t["hidden"])
        norm = self.normalize(t["hidden"], p.prefix+"ln_f", "final-norm")
        watch("final-norm", norm)
        self.call("logits", [t["logits"], norm, w[p.prefix+"wte.weight"]], "logits")
        watch("logits", t["logits"])
        self.call("argmax", [t["argmax"], t["logits"]], "argmax")
        result, = struct.unpack("<I", t["argmax"].read())
        self.position += 1
        return result
