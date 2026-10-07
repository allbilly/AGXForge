"""A model's forward pass as ONE ordered list of dispatches over shared arenas (MM 25.138.3).

Every program binds three buffers and indexes from binding + 0. A dataflow edge "producer writes its buffer 3 at
OUT, consumer reads its buffer 1 at IN" is the constraint base(producer, 3) + OUT = base(consumer, 1) + IN. The
constraints form a forest over the (dispatch, slot) bases; Solver (a union-find with offsets) solves it, and each
tree is placed in the activation arena ACT, GAP apart and ALIGN aligned. Op is one dispatch: its bundle, layout,
kind and the byte extent of each slot. Sim steps a built graph on the CPU over the same arenas, token by token; a
graph's simulator supplies run(), each dispatch's effect computed by its kernel's reference.

Moved from tools/g17modelgraph.py, which keeps the fp16 graph of MM 25.138.3 and its simulator.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


ACT = "act"


ALIGN = 256


GAP = 1 << 20                                  # between placed trees


def _al(v, a=ALIGN):
    return -(-v // a) * a


class Op:
    """One dispatch: its bundle key, layout and kind; slots 1 (a), 2 (b), 3 (c) and their byte extents."""

    def __init__(self, name, key, kind, lay, extents, **meta):
        self.name, self.key, self.kind, self.lay, self.ext, self.meta = name, key, kind, lay, extents, meta

    def __repr__(self):
        return "Op(%s)" % self.name


class Solver:
    def __init__(self):
        self.parent, self.off = {}, {}

    def find(self, v):
        if v not in self.parent:
            self.parent[v], self.off[v] = v, 0
            return v, 0
        acc = 0
        while self.parent[v] != v:
            acc += self.off[v]
            v = self.parent[v]
        return v, acc

    def same(self, u, du, v, dv):
        """base(u) + du == base(v) + dv."""
        ru, ou = self.find(u)
        rv, ov = self.find(v)
        if ru == rv:
            if ou + du != ov + dv:
                raise ValueError("inconsistent constraint %s+%d = %s+%d" % (u, du, v, dv))
            return
        # base(u) = base(ru) + ou; base(ru) = base(rv) + (ov + dv - ou - du)
        self.parent[ru], self.off[ru] = rv, ov + dv - ou - du


# ---------------------------------------------------------------------------------------------------
# the CPU simulator

NAN = np.frombuffer(np.array([0x7FC00000], "<u4").tobytes(), "<f4")[0]


class Sim:
    """The graph's arenas as bytes, stepped one token at a time: the per-token host writes, then every dispatch's
    effect through run(), which a graph's simulator defines (each dispatch computed by the reference its GPU check
    uses), then the logits read back."""

    def __init__(self, g):
        self.g = g
        self.arena = {ACT: np.zeros(g["arenas"][ACT], np.uint8), "zero": np.zeros(g["arenas"]["zero"], np.uint8)}
        for init in g["arena_init"]:
            self.arena[init["arena"]] = np.memmap(init["file"], np.uint8, "r")
        self.embed = np.load(g["tables"]["embedding"], mmap_mode="r")
        z = np.load(g["tables"]["rope"])
        self.cos, self.sin = z["cos"], z["sin"]

    def v(self, arena, off, n, dt):
        return np.frombuffer(self.arena[arena][off:off + n * np.dtype(dt).itemsize].tobytes(), dt, n)

    def w(self, off, arr):
        raw = np.ascontiguousarray(arr).tobytes()
        self.arena[ACT][off:off + len(raw)] = np.frombuffer(raw, np.uint8)

    def poison(self, off, n):
        self.w(off, np.full(n // 4, NAN, "<f4"))

    def token(self, tok, pos, ops):
        g = self.g
        for pw in g["per_token_writes"]:
            src, off = pw["source"], pw["offset"]
            if src == "embedding_row":
                self.w(off, np.asarray(self.embed[int(tok)], np.float16))
            elif src in ("rope_len_word", "split_len_word"):
                self.w(off, np.array([pos], "<u4"))
            elif src == "rope_cos_row":
                self.w(off, self.cos[pos])
            elif src == "rope_sin_row":
                self.w(off, self.sin[pos])
        for d, op in zip(g["dispatches"], ops):
            self.run(d, op, pos)
        lr = g["logits_readback"]
        return self.v(lr["arena"], lr["offset"], g["vocab"], "<f4").copy()

    def run(self, d, op, pos):
        raise NotImplementedError("a graph's simulator defines run(): %s" % op.kind)
