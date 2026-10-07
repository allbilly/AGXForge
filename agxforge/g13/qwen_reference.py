"""Independent FP64 Qwen2 oracle. Never used to supply execution tensors."""
import numpy as np


class Reference:
    def __init__(self, checkpoint, plan):
        self.checkpoint, self.plan, self.position = checkpoint, plan, 0
        self.keys = np.zeros((plan.layers, plan.capacity, plan.kvh, plan.hd), np.float64)
        self.values = np.zeros_like(self.keys)
        self.mapping = np.repeat(np.arange(plan.kvh), plan.heads//plan.kvh)

    def matvec(self, name, x):
        rows = self.checkpoint.tensors[name]["shape"][0]
        result = np.empty(rows, np.float64)
        for first in range(0, rows, 2048):
            last = min(first+2048, rows)
            result[first:last] = self.checkpoint.floats(name, first, last) @ x
        return result

    def normalize(self, x, weight):
        return x/np.sqrt(np.mean(x*x)+self.plan.eps)*self.checkpoint.floats(weight)

    def rotary(self, x):
        hd, theta = self.plan.hd, self.plan.theta
        # Match the declared FP32 storage of the static model frequency table.
        inv = (theta**(-np.arange(hd//2, dtype=np.float64)*2/hd)).astype(np.float32).astype(np.float64)
        angles = self.position*inv
        lo, hi = x[:, :hd//2], x[:, hd//2:]
        return np.concatenate([lo*np.cos(angles)-hi*np.sin(angles), hi*np.cos(angles)+lo*np.sin(angles)], axis=1)

    def consume(self, token):
        p, c = self.plan, self.checkpoint
        trace = {}
        x = c.floats("model.embed_tokens.weight", token, token+1)[0]
        trace["embedding"] = x.copy()
        for layer in range(p.layers):
            prefix, label = f"model.layers.{layer}.", f"layer-{layer}"
            norm = self.normalize(x, prefix+"input_layernorm.weight"); trace[label+"-norm1"] = norm
            qkv = {}
            for role in ("q", "k", "v"):
                qkv[role] = self.matvec(prefix+f"self_attn.{role}_proj.weight", norm)+c.floats(prefix+f"self_attn.{role}_proj.bias")
                trace[label+"-"+role] = qkv[role]
            q = self.rotary(qkv["q"].reshape(p.heads, p.hd))
            key = self.rotary(qkv["k"].reshape(p.kvh, p.hd))
            trace[label+"-rope-q"] = q; trace[label+"-rope-k"] = key
            self.keys[layer, self.position] = key
            self.values[layer, self.position] = qkv["v"].reshape(p.kvh, p.hd)
            trace[label+"-cache-k"] = self.keys[layer].copy()
            trace[label+"-cache-v"] = self.values[layer].copy()
            # Keep the mapped-head axis explicit in the independent oracle.
            ks = np.take(self.keys[layer, :self.position+1], self.mapping, axis=1)
            scores = np.einsum('hd,thd->ht', q, ks)/np.sqrt(p.hd)
            full = np.full((p.heads, p.capacity), -np.inf, np.float64)
            full[:, :self.position+1] = scores; trace[label+"-scores"] = full
            probs = np.exp(scores-scores.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
            full_probs = np.zeros_like(full); full_probs[:, :self.position+1] = probs
            trace[label+"-probs"] = full_probs
            vs = np.take(self.values[layer, :self.position+1], self.mapping, axis=1)
            attention = np.einsum('ht,thd->hd', probs, vs).reshape(-1); trace[label+"-attention"] = attention
            projection = self.matvec(prefix+"self_attn.o_proj.weight", attention); trace[label+"-o"] = projection
            x = x+projection; trace[label+"-residual1"] = x
            norm = self.normalize(x, prefix+"post_attention_layernorm.weight"); trace[label+"-norm2"] = norm
            gate = self.matvec(prefix+"mlp.gate_proj.weight", norm); trace[label+"-gate"] = gate
            up = self.matvec(prefix+"mlp.up_proj.weight", norm); trace[label+"-up"] = up
            activation = gate/(1+np.exp(-gate))*up; trace[label+"-activation"] = activation
            down = self.matvec(prefix+"mlp.down_proj.weight", activation); trace[label+"-down"] = down
            x = x+down; trace[label+"-residual2"] = x
        normalized = self.normalize(x, "model.norm.weight"); trace["final-norm"] = normalized
        trace["logits"] = self.matvec("model.embed_tokens.weight", normalized)
        self.position += 1
        return trace
