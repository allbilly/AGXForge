"""Conservative scalar G13 backend for AGXForge's existing SSA IR.

Reuse the IR's width/bit-pattern semantics, select G13 forms, reserve r0 for
execution control, allocate distinct registers, and refuse spills or arbitrary
CFGs. Admitted loops are structured do-while loops with a provable +1 counter.
The G17 compiler and its decoder validation are untouched.
"""
from agxforge.g17 import ir
from .abi import Binding, G13Program
from .encode import assemble
from .admission import Unsupported, admit


def compile_function(fn, *, threads, extents, workgroup_size=32):
    admit(fn, threads=threads, extents=extents, workgroup_size=workgroup_size)
    buffers = sorted(fn.buffers, key=lambda b: b.slot)
    ops = [op for block in fn.blocks for op in block.ops]
    regs, next_reg = {}, 3  # r0 execution state; r1 scratch; r2 mask bound
    for op in ops:
        if op.dest is not None and op.kind != "cmp":
            if op.dest.type not in (ir.I32, ir.F32, ir.I16, ir.F16):
                raise Unsupported(f"unsupported value width {op.dest.type}")
            if next_reg >= 122:
                raise Unsupported("register pressure requires spilling; scratch execution is not verified")
            regs[op.dest] = f"r{next_reg}" + ("l" if op.dest.type in (ir.I16, ir.F16) else "")
            next_reg += 1
    lines = ["mov_imm r0l, 0", "get_sr r1, sr80", f"mov_imm r2, {threads}, 0",
             "if_icmp r0l, ult, r1, r2, 1", "jmp_exec_none @program_end"]
    builtin_ids, accessed = {80}, {b: set() for b in buffers}
    positions = {b: i for i, b in enumerate(fn.blocks)}
    phis = {b: [o for o in b.ops if o.kind == "phi"] for b in fn.blocks}
    initialized = set()
    open_loop = None

    def src(value):
        if isinstance(value, ir.Imm):
            if not 0 <= value.v <= 255:
                raise Unsupported("large immediates must be explicit IR constants")
            return str(value.v)
        if value not in regs: raise Unsupported(f"undefined SSA value {value}")
        return regs[value]

    def copy_phi(target, latch):
        for op in phis[target]:
            if len(op.args) != 2: raise Unsupported("loop phi requires entry and latch")
            value = op.args[1 if latch else 0]
            if latch and value in [p.dest for p in phis[target]] and value is not op.dest:
                raise Unsupported("cyclic phi parallel copies require additional lowering")
            lines.append(f"mov {src(op.dest)}, {src(value)}")

    for bi, block in enumerate(fn.blocks):
        if phis[block] and block not in initialized:
            raise Unsupported("phi block lacks an initialized structured entry")
        lines.append(block.label + ":")
        if block.term is None: raise Unsupported(f"unterminated block {block.label}")
        for op in block.ops:
            k, a, at = op.kind, op.args, op.attrs
            dest = regs.get(op.dest)
            if k == "const":
                bits = 16 if op.dest.type in (ir.I16, ir.F16) else 32
                lines.append(f"mov_imm {dest}, {a[0].v & ((1 << bits) - 1)}" + (", 0" if bits == 32 else ""))
            elif k == "builtin":
                axis = at.get("axis", "x")
                if axis != "x": raise Unsupported("the first executor implements one-dimensional launches")
                sr = {"thread_position_in_grid": 80, "thread_position_in_threadgroup": 48,
                      "threadgroup_position_in_grid": 0, "thread_index_in_simdgroup": 52,
                      "simdgroup_index_in_threadgroup": 53}.get(at["which"])
                if sr is None: raise Unsupported(f"unsupported builtin {at['which']}")
                sr += "xyz".index(axis) if sr in (0, 48, 80) else 0
                builtin_ids.add(sr); lines.append(f"get_sr {dest}, sr{sr}")
            elif k in ("load", "store_at"):
                buf, index = a[:2]
                if buf not in accessed: raise Unsupported("access to undeclared buffer")
                if at.get("shift16") or at.get("scale", 1) != 1 or at.get("component") is not None:
                    raise Unsupported("special G17 memory addressing is not a G13 contract")
                displacement = at.get("offset", 0) + at.get("disp", 0)
                idx = src(index)
                if displacement:
                    if not 0 <= displacement <= 255: raise Unsupported("large displacement must be IR arithmetic")
                    lines.append(f"iadd r1, {idx}, {displacement}"); idx = "r1"
                width = at.get("width", "word")
                if width not in ("word", "half"): raise Unsupported("only 16/32-bit scalar accesses")
                required = "half" if buf.elem in (ir.I16, ir.F16, "bfloat") else "word"
                if width != required: raise Unsupported("memory width disagrees with buffer element")
                if k == "load" and ((width == "half") != (op.dest.type in (ir.I16, ir.F16))):
                    raise Unsupported("memory width disagrees with destination")
                u = buf.slot * 2
                if k == "load":
                    accessed[buf].add("read")
                    lines += [f"device_load 0, i{16 if width == 'half' else 32}, x, {dest}, u{u}_u{u+1}, {idx}, unsigned, lsl 0", "wait 0"]
                else:
                    if (width == "half") != (a[2].type in (ir.I16, ir.F16)):
                        raise Unsupported("memory width disagrees with store source")
                    accessed[buf].add("write")
                    lines += [f"device_store 0, i{16 if width == 'half' else 32}, x, {src(a[2])}, u{u}_u{u+1}, {idx}, unsigned, lsl 0, 0", "wait 0"]
            elif k in ("fadd", "fsub", "fmul", "fma"):
                operands = [src(x) for x in a]
                if k == "fsub": operands[1] += ".neg"
                mnemonic = dict(fadd="fadd32", fsub="fadd32", fmul="fmul32", fma="fmadd32")[k]
                lines.append(f"{mnemonic} {dest}, " + ", ".join(operands))
            elif k in ("add", "sub", "mul", "madd"):
                operands = [src(x) for x in a]
                if k in ("add", "sub"): lines.append(f"{'isub' if k == 'sub' else 'iadd'} {dest}, " + ", ".join(operands))
                else: lines.append(f"imadd {dest}, " + ", ".join(operands if k == "madd" else operands + ["0"]))
            elif k in ("and", "or", "xor"):
                lines.append(f"{k} {dest}, {src(a[0])}, {src(a[1])}")
            elif k in ("shl", "shr"):
                mnemonic = "bfi" if k == "shl" else "bfeil"
                operands = f"0, {src(a[0])}"
                lines.append(f"{mnemonic} {dest}, {operands}, {src(a[1])}, mask 0xFFFFFFFF")
            elif k in ("rcp", "rsqrt", "exp2", "log2"):
                lines.append(f"{k} {dest}, {src(a[0])}")
            elif k == "sin_turns":
                lines += [f"floor r1, {src(a[0])}", f"fadd32 {dest}, {src(a[0])}, r1.neg",
                          f"fmul32 {dest}, {dest}, 4.0", f"sin_pt_1 r1, {dest}",
                          f"sin_pt_2 {dest}, r1", f"fmul32 {dest}, r1, {dest}"]
            elif k == "f16_to_f32": lines.append(f"fmul32 {dest}, {src(a[0])}, 1.0")
            elif k == "f32_to_f16_rte": lines.append(f"fmul32 {dest}, {src(a[0])}, 1.0")
            elif k in ("u32_to_f32", "i32_to_f32", "f32_to_u32", "f32_to_i32"):
                mode = {"u32_to_f32": "u32_to_f", "i32_to_f32": "s32_to_f", "f32_to_u32": "f_to_u32", "f32_to_i32": "f_to_s32"}[k]
                lines.append(f"convert {mode}, {dest}, {src(a[0])}, {'rte' if k.endswith('f32') else 'rtz'}")
            elif k == "bitcast":
                if (op.dest.type in (ir.I16, ir.F16)) != (a[0].type in (ir.I16, ir.F16)):
                    raise Unsupported("bitcast must preserve representation width")
                lines.append(f"mov {dest}, {src(a[0])}")
            elif k in ("csel", "fcsel", "icmp", "fmax", "fmin"):
                if k in ("fmax", "fmin"):
                    lines.append(f"fcmpsel {'gt' if k == 'fmax' else 'lt'}, {dest}, {src(a[0])}, {src(a[1])}, {src(a[0])}, {src(a[1])}")
                else:
                    rel = ({"eq": "eq", "lt": "lt", "gt": "gt"} if k == "fcsel" else
                           {"eq": "ueq", "lt": "ult", "gt": "ugt"}).get(at.get("rel", "eq"))
                    if rel is None: raise Unsupported("unsupported integer comparison")
                    xy = [src(x) for x in a[2:]] if k in ("csel", "fcsel") else ["1", "0"]
                    lines.append(f"{'fcmpsel' if k == 'fcsel' else 'icmpsel'} {rel}, {dest}, {src(a[0])}, {src(a[1])}, {xy[0]}, {xy[1]}")
            elif k in ("phi", "cmp"): pass
            elif k == "br":
                target = a[0]
                if positions.get(target) != bi + 1: raise Unsupported("only structured forward fallthrough branches")
                if phis[target]:
                    if open_loop is not None: raise Unsupported("nested loops require separate execution-mask validation")
                    copy_phi(target, False); initialized.add(target)
                    open_loop = target
                    lines.append("if_fcmp r0l, eq, 0.0, 0.0, 1")
            elif k == "br_cond":
                predicate, target, after = a
                if target is not open_loop or target not in initialized or positions[target] > bi or positions.get(after) != bi + 1:
                    raise Unsupported("only structured do-while latch branches")
                compare = predicate.op
                if compare.kind != "cmp" or compare.attrs.get("pred") != "lt" or compare.attrs.get("source_modifier", 0):
                    raise Unsupported("loop latch must be an unsigned < comparison")
                increment, bound = compare.args
                step = increment.op
                if (step.kind != "add" or not isinstance(step.args[1], ir.Imm) or step.args[1].v != 1 or
                    step.args[0] not in [p.dest for p in phis[target]]):
                    raise Unsupported("loop counter must advance by exactly one")
                phi = step.args[0].op
                seed = phi.args[0]
                if (not isinstance(seed, ir.Value) or seed.op.kind != "const" or seed.op.args[0].v != 0 or phi.args[1] is not increment):
                    raise Unsupported("loop counter must start at zero and latch its increment")
                limit = bound.v if isinstance(bound, ir.Imm) else compare.attrs.get("cap")
                if type(limit) is not int or not 0 < limit <= 10000000:
                    raise Unsupported("loop requires a finite positive static cap")
                # Every counter transfer must preserve its unsigned width. The
                # latch compares the increment, so it must reach the cap before
                # either the add or the phi copy can wrap back to zero.
                widths = {16 if v.type in (ir.I16, ir.F16) else 32
                          for v in (seed, phi.dest, increment)}
                if len(widths) != 1:
                    raise Unsupported("loop counter seed, phi and increment must preserve width")
                if limit >= 1 << next(iter(widths)):
                    raise Unsupported("loop cap would allow counter overflow")
                copy_phi(target, True)
                lines.append(f"mov_imm r1, {limit}, 0")
                lines.append(f"while_icmp r0l, ult, {src(increment)}, r1, 1")
                if isinstance(bound, ir.Value): lines.append(f"while_icmp r0l, ult, {src(increment)}, {src(bound)}, 1")
                lines += [f"jmp_exec_any @{target.label}", "pop_exec r0l, 1"]
                open_loop = None
            elif k == "ret":
                if bi != len(fn.blocks) - 1: raise Unsupported("return must be at structured exit")
            else: raise Unsupported(f"unsupported G13 IR operation {k}")
    if open_loop is not None: raise Unsupported("loop entry lacks a closing bounded latch")
    lines += ["program_end:", "pop_exec r0l, 1", "stop"]
    contract = []
    for buf in buffers:
        access = accessed[buf]
        contract.append(Binding(buf.name, buf.slot, "read_write" if len(access) == 2 else next(iter(access), "read"),
                                extents[buf.name], ir.ELEM_BYTES[buf.elem]))
    code = assemble(lines) + bytes.fromhex("0800") * 8
    return G13Program(fn.name, code, tuple(contract), next_reg * 2, len(buffers) * 4,
                      workgroup_size=workgroup_size, builtins=tuple(sorted(builtin_ids)),
                      reserved_register_halfs=tuple(range(6)),
                      bounds_policy="masked", origin="agxforge_ir", logical_threads=threads)
