"""Scalar IR/CFG admission shared by authored G13 and Mesa code generation.

This checks semantics and launch contracts without allocating machine registers
or emitting instructions. Compiler-specific limits belong in the backends.
"""
from dataclasses import dataclass
from agxforge.g17 import ir
from .abi import Binding


class Unsupported(ValueError): pass


ARITY = dict(const=1, builtin=0, load=2, store_at=3, phi=2, cmp=2, br=1, br_cond=3, ret=0,
             fadd=2, fsub=2, fmul=2, fma=3, add=2, sub=2, mul=2, madd=3,
             shl=2, shr=2, bitcast=1, rcp=1, rsqrt=1, exp2=1, log2=1, sin_turns=1,
             f16_to_f32=1, f32_to_f16_rte=1, u32_to_f32=1, i32_to_f32=1,
             f32_to_u32=1, f32_to_i32=1, csel=4, fcsel=4, icmp=2, fmax=2, fmin=2,
             **{"and": 2, "or": 2, "xor": 2})
ATTRS = {"builtin": {"which", "axis"}, "load": {"width", "offset", "disp", "scale", "shift16"},
         "store_at": {"width", "offset", "disp", "scale", "shift16"},
         "cmp": {"pred", "source_modifier", "cap"}, "csel": {"rel"}, "fcsel": {"rel"}, "icmp": {"rel"}}
BUILTINS = {"thread_position_in_grid", "thread_position_in_threadgroup", "threadgroup_position_in_grid",
            "thread_index_in_simdgroup", "simdgroup_index_in_threadgroup"}


def bits(value):
    if isinstance(value, ir.Imm): return 32
    if value.type not in (ir.I32, ir.F32, ir.I16, ir.F16):
        raise Unsupported(f"unsupported value width {value.type}")
    return 16 if value.type in (ir.I16, ir.F16) else 32


@dataclass(frozen=True)
class Contract:
    name: str
    bindings: tuple
    logical_threads: int
    uniform_halfs: int
    workgroup_size: int = 32


def admit(fn, *, threads, extents, workgroup_size=32):
    if fn.threadgroup:
        raise Unsupported("threadgroup memory/barriers require a separately verified capability level")
    if not fn.blocks or len({b.label for b in fn.blocks}) != len(fn.blocks):
        raise Unsupported("nonempty uniquely labelled CFG required")
    if type(threads) is not int or not 0 < threads < 2**31:
        raise Unsupported("fixed positive logical launch extent required")
    if type(workgroup_size) is not int or not 32 <= workgroup_size <= 1024 or workgroup_size % 32:
        raise Unsupported("workgroup must contain complete SIMD groups")
    buffers = sorted(fn.buffers, key=lambda b: b.slot)
    if not buffers or [b.slot for b in buffers] != list(range(len(buffers))):
        raise Unsupported("G13 buffers must use dense slots")
    if len({b.name for b in buffers}) != len(buffers) or set(extents) != {b.name for b in buffers}:
        raise Unsupported("every unique buffer must declare its required extent")
    for buf in buffers:
        if buf.elem not in (ir.I32, ir.F32, ir.I16, ir.F16, "bfloat"):
            raise Unsupported(f"unsupported element {buf.elem}")
        if type(extents[buf.name]) is not int or extents[buf.name] <= 0:
            raise Unsupported("positive buffer extents required")
    ops = [op for block in fn.blocks for op in block.ops]
    defined, accessed = set(), {b: set() for b in buffers}
    for block in fn.blocks:
        if not block.ops: raise Unsupported("each block must end with exactly one terminator")
        seen_non_phi = False
        for i, op in enumerate(block.ops):
            kind, args = op.kind, op.args
            if kind not in ARITY or len(args) != ARITY[kind]:
                raise Unsupported(f"unsupported operation or operand count: {kind}")
            if set(op.attrs) - ATTRS.get(kind, set()):
                raise Unsupported(f"unsupported G13 attributes on {kind}: {sorted(op.attrs)}")
            if (kind in ("br", "br_cond", "ret")) != (i == len(block.ops)-1):
                raise Unsupported("each block must end with exactly one terminator")
            if (op.dest is None) != (kind in ("store_at", "br", "br_cond", "ret")):
                raise Unsupported("operation definition disagrees with its kind")
            if kind == "phi":
                if seen_non_phi: raise Unsupported("two-input phis must precede the loop body")
                if isinstance(args[0], ir.Value) and args[0] not in defined:
                    raise Unsupported("phi entry must already be defined")
            else:
                seen_non_phi = True
                if any(isinstance(a, ir.Value) and a not in defined for a in args):
                    raise Unsupported("SSA use before definition")
            if op.dest is not None:
                bits(op.dest)
                if op.dest in defined or op.dest.op is not op:
                    raise Unsupported("SSA definitions must be unique")
                defined.add(op.dest)
            if kind in ("load", "store_at"):
                buf, index = args[:2]
                if buf not in accessed: raise Unsupported("access to undeclared buffer")
                if not isinstance(index, (ir.Value, ir.Imm)):
                    raise Unsupported("scalar buffer index required")
                if isinstance(index, ir.Value) and bits(index) != 32:
                    raise Unsupported("buffer index must be 32 bits")
                if op.attrs.get("shift16") or op.attrs.get("scale", 1) != 1:
                    raise Unsupported("special G17 memory addressing is not a G13 contract")
                if any(type(op.attrs.get(a, 0)) is not int for a in ("offset", "disp")):
                    raise Unsupported("constant element displacement required")
                width = op.attrs.get("width", "word")
                required = "half" if ir.ELEM_BYTES[buf.elem] == 2 else "word"
                if width != required: raise Unsupported("memory width disagrees with buffer element")
                value = op.dest if kind == "load" else args[2]
                if not isinstance(value, ir.Value) or bits(value) != ir.ELEM_BYTES[buf.elem]*8:
                    raise Unsupported("memory width disagrees with scalar value")
                if kind == "store_at" and buf.elem == "bfloat":
                    raise Unsupported("BF16 narrowing/store semantics are not established")
                accessed[buf].add("read" if kind == "load" else "write")
            elif kind == "builtin":
                if op.attrs.get("which") not in BUILTINS or op.attrs.get("axis", "x") != "x":
                    raise Unsupported("unsupported one-dimensional builtin")
                if bits(op.dest) != 32: raise Unsupported("builtin must be 32 bits")
            elif kind == "const":
                if not isinstance(args[0], ir.Imm): raise Unsupported("constant bit pattern required")
            elif kind not in ("br", "br_cond", "ret"):
                if any(not isinstance(a, (ir.Value, ir.Imm)) for a in args):
                    raise Unsupported("scalar operands required")
                if kind in ("csel", "fcsel", "icmp") and op.attrs.get("rel", "eq") not in ("eq", "lt", "gt"):
                    raise Unsupported("unsupported comparison relation")
                if kind == "bitcast" and bits(op.dest) != bits(args[0]):
                    raise Unsupported("bitcast must preserve representation width")
                if kind in ("f16_to_f32", "f32_to_f16_rte"):
                    pair = (bits(args[0]), bits(op.dest))
                    if pair != ((16, 32) if kind == "f16_to_f32" else (32, 16)):
                        raise Unsupported("floating conversion width mismatch")
    if any(isinstance(a, ir.Value) and a not in defined for op in ops for a in op.args):
        raise Unsupported("undefined phi latch")
    positions = {b: i for i, b in enumerate(fn.blocks)}
    phis = {b: [op for op in b.ops if op.kind == "phi"] for b in fn.blocks}
    initialized, open_loop = set(), None
    for bi, block in enumerate(fn.blocks):
        if phis[block] and block not in initialized:
            raise Unsupported("phi block lacks an initialized structured entry")
        term = block.ops[-1]
        if term.kind == "br":
            target = term.args[0]
            if positions.get(target) != bi+1: raise Unsupported("only structured forward fallthrough branches")
            if phis[target]:
                if open_loop is not None: raise Unsupported("nested loops require separate execution-mask validation")
                for phi in phis[target]:
                    if any(bits(a) != bits(phi.dest) for a in phi.args):
                        raise Unsupported("loop phi must preserve width")
                initialized.add(target); open_loop = target
        elif term.kind == "br_cond":
            predicate, target, after = term.args
            if target is not open_loop or target not in initialized or positions.get(after) != bi+1:
                raise Unsupported("only structured do-while latch branches")
            if not isinstance(predicate, ir.Value) or predicate.op.kind != "cmp":
                raise Unsupported("loop latch must be an unsigned < comparison")
            compare = predicate.op
            if compare.attrs.get("pred") != "lt" or compare.attrs.get("source_modifier", 0):
                raise Unsupported("loop latch must be an unsigned < comparison")
            increment, bound = compare.args
            if not isinstance(increment, ir.Value): raise Unsupported("loop counter must advance by exactly one")
            step = increment.op
            if (step.kind != "add" or not isinstance(step.args[1], ir.Imm) or step.args[1].v != 1 or
                    not isinstance(step.args[0], ir.Value) or step.args[0].op not in phis[target]):
                raise Unsupported("loop counter must advance by exactly one")
            phi = step.args[0].op; seed = phi.args[0]
            if not isinstance(seed, ir.Value) or seed.op.kind != "const" or seed.op.args[0].v != 0 or phi.args[1] is not increment:
                raise Unsupported("loop counter must start at zero and latch its increment")
            limit = bound.v if isinstance(bound, ir.Imm) else compare.attrs.get("cap")
            if type(limit) is not int or not 0 < limit <= 10000000:
                raise Unsupported("loop requires a finite positive static cap")
            widths = {bits(v) for v in (seed, phi.dest, increment)}
            if len(widths) != 1: raise Unsupported("loop counter seed, phi and increment must preserve width")
            if limit >= 1 << next(iter(widths)): raise Unsupported("loop cap would allow counter overflow")
            open_loop = None
        elif term.kind == "ret" and bi != len(fn.blocks)-1:
            raise Unsupported("return must be at structured exit")
    if open_loop is not None: raise Unsupported("loop entry lacks a closing bounded latch")
    for op in ops:
        if op.kind == "cmp" and any(op.dest in other.args and other.kind != "br_cond" for other in ops):
            raise Unsupported("cmp is a branch predicate; use icmp for a scalar value")
    bindings = tuple(Binding(buf.name, buf.slot,
        "read_write" if len(accessed[buf]) == 2 else next(iter(accessed[buf]), "read"),
        extents[buf.name], ir.ELEM_BYTES[buf.elem]) for buf in buffers)
    return Contract(fn.name, bindings, threads, len(buffers)*4, workgroup_size)
