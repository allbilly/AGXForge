"""Apple-compiled scalar MSL baseline for the GPT-2 graph, using the same dispatch ABI.

This comparison path deliberately executes Apple's compiler output. AGXForge's
G13 bytes remain a logical program descriptor, never the baseline shader image.
"""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from agxforge.runtime.macos import Executor as MetalExecutor

EXECUTOR_NAME = "Apple-compiled scalar Metal"


def sources(plan):
    d, f, v, h, cap, hd = plan.d, plan.ffn, plan.vocab, plan.heads, plan.capacity, plan.hd
    result = {}
    def add(role, names, body, integers=()):
        args = [f"device {'uint' if n in integers else 'float'} *{n} [[buffer({i})]]" for i, n in enumerate(names)]
        args.append("uint t [[thread_position_in_grid]]")
        result[role] = "#include <metal_stdlib>\nusing namespace metal;\nkernel void carrier("+", ".join(args)+") {\n"+f"if (t >= {plan.programs[role].logical_threads}u) return;\n"+body+"\n}\n"
    add("embedding", ["out", "weights", "params"], f"out[t] = weights[params[0]*{d}u+t];", ("params",))
    add("position_embedding", ["out", "weights", "params"], f"out[t] = weights[params[1]*{d}u+t];", ("params",))
    for role, width, rows, operation in (("sum", d, 1, "sum"), ("softmax_sum", cap, h, "sum"), ("softmax_max", cap, h, "max")):
        seed = "-INFINITY" if operation == "max" else "0.0f"
        step = "total = max(total, x[t*WIDTH+i]);" if operation == "max" else "total += x[t*WIDTH+i];"
        add(role, ["out", "x"], f"float total = {seed}; for (uint i=0; i<{width}u; ++i) {{ "+step.replace("WIDTH", f"{width}u")+" } out[t]=total;")
    add("norm_stats", ["out", "x", "sum"], f"float mean=sum[0]/{d}.0f; float total=0; for (uint i=0; i<{d}u; ++i) {{ float a=x[i]-mean; total=fma(a,a,total); }} out[0]=mean; out[1]=rsqrt(total/{d}.0f+{plan.eps:.9g}f);")
    add("norm_apply", ["out", "x", "gain", "bias", "stats"], "out[t]=(x[t]-stats[0])*stats[1]*gain[t]+bias[t];")
    for role, inputs, outputs in (("qkv", d, 3*d), ("o", d, d), ("up", d, f), ("down", f, d)):
        add(role, ["out", "x", "weights", "bias"], f"float total=0; for (uint i=0; i<{inputs}u; ++i) total=fma(x[i],weights[i*{outputs}u+t],total); out[t]=total+bias[t];")
    add("split", ["q", "k", "v", "packed"], f"q[t]=packed[t]; k[t]=packed[t+{d}u]; v[t]=packed[t+{2*d}u];")
    add("kv_write", ["cache", "x", "params"], f"cache[params[1]*{d}u+t]=x[t];", ("params",))
    add("scores", ["out", "q", "k", "headmap", "params"], f"uint head=t/{cap}u, token=t%{cap}u; uint qb=head*{hd}u, kb=(token*{h}u+headmap[head])*{hd}u; float total=0; for (uint i=0; i<{hd}u; ++i) total=fma(q[qb+i],k[kb+i],total); out[t]=token>params[1] ? -INFINITY : total*{1/math.sqrt(hd):.9g}f;", ("headmap", "params"))
    add("softmax_exp", ["out", "x", "row"], f"out[t]=exp2((x[t]-row[t/{cap}u])*{math.log2(math.e):.9g}f);")
    add("softmax_normalize", ["out", "x", "row"], f"out[t]=x[t]/row[t/{cap}u];")
    add("attend", ["out", "probs", "v", "headmap"], f"uint head=t/{hd}u, comp=t%{hd}u; uint vb=headmap[head]*{hd}u+comp; float total=0; for (uint i=0; i<{cap}u; ++i) total=fma(probs[head*{cap}u+i],v[i*{d}u+vb],total); out[t]=total;", ("headmap",))
    add("residual", ["out", "a", "b"], "out[t]=a[t]+b[t];")
    # tanh's fast-math lowering overflows for an observed finite activation.
    # Use the equivalent stable sigmoid form, matching the G13 kernel arithmetic.
    add("gelu", ["out", "x"], f"float a=x[t]; float inner=a+0.044715f*a*a*a; out[t]=a/(1.0f+exp2({-2*math.sqrt(2/math.pi)*math.log2(math.e):.9g}f*inner));")
    add("logits", ["out", "x", "weights"], f"float total=0; for (uint i=0; i<{d}u; ++i) total=fma(weights[t*{d}u+i],x[i],total); out[t]=total;")
    add("argmax", ["out", "x"], f"float best=-INFINITY; uint index=0; for (uint i=0; i<{v}u; ++i) {{ if (x[i]>best) {{ best=x[i]; index=i; }} }} out[0]=index;", ("out",))
    if set(result) != set(plan.programs): raise ValueError("incomplete GPT-2 MSL baseline")
    return {plan.programs[role].code_hash: source for role, source in result.items()}


class Executor(MetalExecutor):
    def __init__(self, plan, **kwargs):
        self.msl = sources(plan)
        super().__init__(**kwargs)

    def platform(self):
        result = super().platform()
        result.update(executor=EXECUTOR_NAME, shader_compiler="Apple xcrun metal, default optimization and fast math")
        return result

    def prepare(self, program):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        if program.code_hash not in self.programs:
            source = self.msl[program.code_hash]
            stem = self.directory/program.code_hash
            path = stem.with_suffix(".metal"); path.write_text(source)
            air, library, archive = (stem.with_suffix(ext) for ext in (".air", ".lib.metallib", ".arc.metallib"))
            subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(path), "-o", str(air)], check=True, capture_output=True)
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)], check=True, capture_output=True)
            if self.lib.am_carrier(self.handle, os.fsencode(library), os.fsencode(archive)): self.raise_error()
            pipeline = self.lib.am_pipeline(self.handle, os.fsencode(library), os.fsencode(archive))
            if not pipeline: self.raise_error()
            self.programs[program.code_hash] = (pipeline, dict(msl=source.encode(), library=library.read_bytes(), archive=archive.read_bytes()))
        return self.programs[program.code_hash]

    def dispatch(self, program, buffers, threads, evidence=None, capture_buffers=True):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        if threads != program.logical_threads or len(buffers) != len(program.bindings): raise ValueError("baseline launch extent/bindings")
        needs = {"read": {"read"}, "write": {"write"}, "read_write": {"read", "write"}}
        for binding in program.bindings:
            buf = buffers[binding.slot]
            if (buf.executor is not self or buf.size < binding.min_bytes or buf.address % binding.alignment
                    or not needs[binding.access] <= needs[buf.access]): raise ValueError("baseline buffer contract")
        pipeline, data = self.prepare(program)
        record = dict(status="PREPARED", executor=EXECUTOR_NAME, program=program.descriptor(),
                      program_role="logical graph descriptor; G13 bytes are not executed",
                      logical_threads=threads, workgroup_size=program.workgroup_size,
                      buffers=[dict(binding=b.name, **buffers[b.slot].record()) for b in program.bindings])
        folder = Path(evidence) if evidence else None
        if folder:
            folder.mkdir(parents=True, exist_ok=False)
            for name, raw in (("shader.metal", data["msl"]), ("metal-library.bin", data["library"]), ("metal-archive.bin", data["archive"])):
                (folder/name).write_bytes(raw)
                record[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        bound = list(buffers)+[self.dummy]*(8-len(buffers))
        native = (C.c_void_p*8)(*(buf.handle for buf in bound))
        start = time.monotonic_ns()
        try:
            global_x = (threads+31)//32*32
            if self.lib.am_submit(self.handle, pipeline, native, 8, 64, global_x, 32, self.timeout_ms): self.raise_error()
            if not all(buf.canaries_ok() for buf in bound): raise RuntimeError("baseline GPU canary changed")
            record.update(status="COMPLETED", fence_completed=True, canaries_intact=True, host_submit_wait_ns=time.monotonic_ns()-start)
            if folder and capture_buffers:
                for binding in program.bindings: (folder/f"slot-{binding.slot}-after.bin").write_bytes(buffers[binding.slot].read())
        except BaseException as error:
            self.poisoned = True; record.update(status="FAIL", error=str(error)); raise
        finally:
            if folder: (folder/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
        return record
