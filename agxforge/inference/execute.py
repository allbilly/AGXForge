"""Running a built graph: the Metal executor, and what it reports.

The Metal executor runs AGXForge-authored G17 programs through Metal compute pipelines. It is tools/g17decodegen (Objective-C, tools/g17decodegen.m; `make native-tools` builds it). It loads every
bundle of graph.json as a Metal compute pipeline, binds each dispatch's three buffers at their offsets in shared arenas,
runs the graph's prefill section if it has one, then decodes: each token is ONE command buffer of the whole model, the
next token chosen on the GPU by the graph's own argmax and generation step, nothing copied between dispatches.
Command buffers take the machine-wide GPU lock (tools/g17gpulock.h).

It is not the below-Metal runtime (agxforge.g17.runtime and tools/g17inference*.py), which submits through Apple's
private IOGPU interface with no Metal in the process.

    decode(graph, tokens, out)      run it; returns the executor's report (out_ids, the timings, per sequence for a
                                    batched graph), which it also writes to `out`

The executor reads further switches from its environment (DECODEGEN_*: dumps, profiles, the prefill in one command
buffer); decode() passes `env` through. DECODEGEN_PIPELINED, which decode() sets by default, runs the tokens as a
pipeline of command buffers instead of waiting on each one (MM 25.142.8).

THIS DISPATCHES GPU WORK. Nothing in agxforge.inference besides decode() does.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# THE REPOSITORY ROOT: the executor is built beside its source in tools/
_REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def executor():
    """The executor binary: tools/g17decodegen, built by `make native-tools` (absent until then)."""
    return _REPO_ROOT / "tools" / "g17decodegen"


def decode(graph, tokens, out, binary=None, pipelined=True, env=None, timeout=900):
    """Decode `tokens` tokens with the graph at `graph` (graph.json) and return the report written to `out`."""
    binary = Path(binary) if binary is not None else executor()
    if not binary.is_file():
        raise FileNotFoundError("%s: build the executor first (make native-tools)" % binary)
    run_env = dict(os.environ if env is None else env)
    if pipelined:
        run_env["DECODEGEN_PIPELINED"] = "1"
    p = subprocess.run([str(binary), str(graph), str(out), str(int(tokens))], env=run_env, capture_output=True,
                       text=True, timeout=timeout, cwd=str(_REPO_ROOT))
    if p.returncode != 0:
        raise RuntimeError("%s failed (%d): %s" % (binary.name, p.returncode, (p.stderr or p.stdout)[-2000:]))
    return json.loads(Path(out).read_text())
