"""Independent FP64 GPT-2 graph; never supplies inference tensors or tokens."""
import numpy as np


class Reference:
    def __init__(self, checkpoint, plan):
        self.checkpoint, self.plan, self.position = checkpoint, plan, 0
        self.keys = np.zeros((plan.layers, plan.capacity, plan.heads, plan.hd), np.float64)
        self.values = np.zeros_like(self.keys)

    def conv1d(self, name, x):
        return x @ self.checkpoint.floats(name+".weight") + self.checkpoint.floats(name+".bias")

    def normalize(self, x, prefix):
        centered = x-x.mean()
        return centered/np.sqrt(np.mean(centered*centered)+self.plan.eps)*self.checkpoint.floats(prefix+".weight")+self.checkpoint.floats(prefix+".bias")

    def consume(self, token):
        p, c = self.plan, self.checkpoint
        trace = {}
        embedding = c.floats(p.prefix+"wte.weight", token, token+1)[0]
        position = c.floats(p.prefix+"wpe.weight", self.position, self.position+1)[0]
        trace["embedding"], trace["position-embedding"] = embedding, position
        x = embedding+position; trace["residual-embedding"] = x
        for layer in range(p.layers):
            prefix, label = p.prefix+f"h.{layer}.", f"layer-{layer}"
            norm = self.normalize(x, prefix+"ln_1"); trace[label+"-norm1"] = norm
            qkv = self.conv1d(prefix+"attn.c_attn", norm); trace[label+"-qkv"] = qkv
            q, key, value = np.split(qkv, 3)
            for role, tensor in (("q", q), ("k", key), ("v", value)): trace[label+"-"+role] = tensor
            self.keys[layer, self.position] = key.reshape(p.heads, p.hd)
            self.values[layer, self.position] = value.reshape(p.heads, p.hd)
            trace[label+"-cache-k"], trace[label+"-cache-v"] = self.keys[layer].copy(), self.values[layer].copy()
            scores = np.einsum("hd,thd->ht", q.reshape(p.heads, p.hd), self.keys[layer, :self.position+1])/np.sqrt(p.hd)
            full = np.full((p.heads, p.capacity), -np.inf, np.float64)
            full[:, :self.position+1] = scores; trace[label+"-scores"] = full
            probs = np.exp(scores-scores.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
            full_probs = np.zeros_like(full); full_probs[:, :self.position+1] = probs; trace[label+"-probs"] = full_probs
            attention = np.einsum("ht,thd->hd", probs, self.values[layer, :self.position+1]).reshape(-1)
            trace[label+"-attention"] = attention
            projection = self.conv1d(prefix+"attn.c_proj", attention); trace[label+"-o"] = projection
            x = x+projection; trace[label+"-residual1"] = x
            norm = self.normalize(x, prefix+"ln_2"); trace[label+"-norm2"] = norm
            up = self.conv1d(prefix+"mlp.c_fc", norm); trace[label+"-up"] = up
            activation = .5*up*(1+np.tanh(np.sqrt(2/np.pi)*(up+.044715*up**3)))
            trace[label+"-activation"] = activation
            down = self.conv1d(prefix+"mlp.c_proj", activation); trace[label+"-down"] = down
            x = x+down; trace[label+"-residual2"] = x
        norm = self.normalize(x, p.prefix+"ln_f"); trace["final-norm"] = norm
        logits = np.empty(p.vocab, np.float64)
        for first in range(0, p.vocab, 2048):
            last = min(first+2048, p.vocab)
            logits[first:last] = c.floats(p.prefix+"wte.weight", first, last) @ norm
        trace["logits"] = logits
        self.position += 1
        return trace
