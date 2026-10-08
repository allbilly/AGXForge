"""Opt-in Mesa 26.2.4 code generation for the bounded scalar M1 ABI.

Shared scalar IR/CFG admission supplies bindings without invoking another
code generator. Only Mesa's emitted bytes execute; no Linux driver is used.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from . import ir
from .admission import Unsupported, admit, bits, ARITY
from .abi import G13Program
from .decode import decode

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPILER = ROOT / "build/macos/mesa-26.2.4/agxforge-bridge/agxforge_mesa"
OPERATIONS = frozenset(ARITY)
CAPABILITIES = frozenset({"scalar", "integer", "bounded_loops", "fp16", "bf16_storage", "approx_math"})


def lower(kernel):
    """Return the native adapter protocol and independently admitted ABI."""
    fn = kernel.function
    missing = kernel.required_capabilities() - CAPABILITIES
    if missing: raise Unsupported(f"Mesa scalar profile requires unsupported capabilities {sorted(missing)}")
    ops = [op for block in fn.blocks for op in block.ops]
    unsupported = {op.kind for op in ops} - OPERATIONS
    if unsupported: raise Unsupported(f"Mesa bridge does not implement {sorted(unsupported)}")
    admission = admit(fn, threads=kernel.threads, extents=kernel.extents)
    if len(admission.bindings) > 8: raise Unsupported("Mesa carrier supports eight buffers")
    ids = {op.dest: i + 1 for i, op in enumerate(o for o in ops if o.dest is not None)}
    next_id = len(ids) + 1
    lines = [f"AGXFORGE_MESA_V2 {len(fn.buffers)} {kernel.threads}"]

    def emit(op, d=0, a=0, b=0, c=0, parameter=0):
        lines.append(f"{op} {d} {a} {b} {c} {parameter}")

    def operand(value):
        nonlocal next_id
        if isinstance(value, ir.Value): return ids[value]
        if not isinstance(value, ir.Imm): raise Unsupported("expected scalar operand")
        result = next_id; next_id += 1
        emit("imm", result, value.v & 0xffffffff, parameter=32)
        return result

    for block in fn.blocks:
        for op in block.ops:
            kind, args = op.kind, op.args
            dest = ids.get(op.dest, 0)
            if kind == "const":
                width = bits(op.dest)
                emit("imm", dest, args[0].v & ((1 << width) - 1), parameter=width)
            elif kind == "builtin":
                builtin = {"thread_position_in_grid": 80, "thread_position_in_threadgroup": 48,
                           "threadgroup_position_in_grid": 0, "thread_index_in_simdgroup": 52,
                           "simdgroup_index_in_threadgroup": 53}[op.attrs["which"]]
                emit("thread", dest, parameter=builtin)
            elif kind in ("load", "store_at"):
                index = operand(args[1])
                displacement = op.attrs.get("offset", 0) + op.attrs.get("disp", 0)
                if displacement:
                    constant = operand(ir.Imm(displacement)); result = next_id; next_id += 1
                    emit("add", result, index, constant, parameter=32); index = result
                width = 16 if op.attrs.get("width", "word") == "half" else 32
                emit("load" if kind == "load" else "store", dest, args[0].slot, index,
                     operand(args[2]) if kind == "store_at" else 0, width)
            elif kind == "br":
                phis = [phi for phi in args[0].ops if phi.kind == "phi"]
                for phi in phis: emit("phi_seed", ids[phi.dest], operand(phi.args[0]), parameter=bits(phi.dest))
                if phis: emit("loop_begin")
            elif kind == "phi": emit("phi_load", dest)
            elif kind == "br_cond":
                for phi in args[1].ops:
                    if phi.kind == "phi": emit("phi_store", ids[phi.dest], operand(phi.args[1]))
                emit("loop_end", a=operand(args[0]))
            elif kind == "cmp":
                if op.attrs.get("pred") != "lt" or op.attrs.get("source_modifier", 0):
                    raise Unsupported("Mesa bridge supports unsigned less-than loop predicates")
                emit("cmp", dest, operand(args[0]), operand(args[1]), parameter=op.attrs.get("cap") or 0)
            elif kind in ("csel", "fcsel", "icmp"):
                relation = {"eq": 0, "lt": 1, "gt": 2}[op.attrs.get("rel", "eq")]
                left, right = operand(args[0]), operand(args[1])
                if kind == "icmp": emit("icmp", dest, left, right, parameter=relation)
                else:
                    predicate = next_id; next_id += 1
                    emit("fcmp" if kind == "fcsel" else "icmp", predicate, left, right, parameter=relation)
                    emit("select", dest, predicate, operand(args[2]), operand(args[3]), bits(op.dest))
            elif kind != "ret":
                operands = [operand(a) for a in args]
                emit(kind, dest, *operands, parameter=bits(op.dest))
    if next_id >= 8192 or len(lines) > 8192:
        raise Unsupported("Mesa adapter instruction limit exceeded")
    return "\n".join(lines) + "\n", admission


def compile(kernel, output, *, compiler=DEFAULT_COMPILER):
    """Compile into a new evidence directory; validate bytes before GPU use."""
    protocol, admission = lower(kernel)
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise FileNotFoundError("build the optional Mesa adapter with tools/build_mesa_agx.py first")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    compiler_hash = hashlib.sha256(compiler.read_bytes()).hexdigest()
    identity_path = compiler.parent / "build-identity.json"
    identity = json.loads(identity_path.read_text())
    if identity.get("compiler_sha256") != compiler_hash:
        raise RuntimeError("compiler does not match recorded Mesa build")
    adapter_hash = hashlib.sha256((ROOT / "tools/mesa_agx/bridge.c").read_bytes()).hexdigest()
    if identity.get("adapter_sha256") != adapter_hash:
        raise RuntimeError("Mesa adapter changed; rebuild it before compilation")
    (output / "build-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    (output / "input.ir").write_text(protocol)
    result = subprocess.run([str(compiler), str((output / "input.ir").resolve()), str(output.resolve())],
                            capture_output=True, text=True, timeout=120)
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    if result.returncode: raise RuntimeError(f"Mesa compilation failed: {result.stderr.strip()}")
    if compiler_hash != hashlib.sha256(compiler.read_bytes()).hexdigest():
        raise RuntimeError("compiler changed during compilation")
    metadata = json.loads((output / "metadata.json").read_text())
    code = (output / "shader.bin").read_bytes()
    program = from_binary(admission, code, metadata)
    manifest = dict(compiler_sha256=compiler_hash, compiler=metadata["compiler"], bridge_protocol_version=2,
                    build_identity_sha256=hashlib.sha256((output / "build-identity.json").read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),
                    code_sha256=program.code_hash, descriptor=program.descriptor())
    (output / "compiler-identity.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return program


class Compiler:
    """Model-plan compiler with a separate native-code receipt for each kernel."""
    name = "Mesa 26.2.4 AGX"

    def __init__(self, output, *, compiler=DEFAULT_COMPILER):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.compiler = Path(compiler)
        self.programs = {}

    def __call__(self, kernel, name=None):
        name = name or kernel.function.name
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("compiler receipt name must be a simple identifier")
        program = compile(kernel, self.output / name, compiler=self.compiler)
        self.programs[name] = json.loads((self.output / name / "compiler-identity.json").read_text())
        return program

    def descriptor(self):
        return dict(name=self.name, bridge_protocol_version=2,
                    programs={name: dict(receipt=f"compiler/{name}/compiler-identity.json", **identity)
                              for name, identity in self.programs.items()})


def from_binary(admission, code, metadata):
    """Admit only the compiler ABI that the measured Metal carrier supplies."""
    expected = dict(compiler="Mesa 26.2.4 AGX", uniform_halfs=admission.uniform_halfs,
                    binary_size=len(code), main_size=len(code), main_offset=0,
                    has_preamble=False, scratch_bytes=0, threadgroup_bytes=0,
                    rodata_halfs=0, workgroup_size=[32, 1, 1])
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Mesa metadata outside measured macOS ABI")
    registers = metadata.get("register_halfs")
    if type(registers) is not int or not 2 <= registers <= 80:
        raise ValueError("Mesa registers exceed macOS carrier")
    listing = decode(code)
    builtins = tuple(sorted({i.fields["SR"] for i in listing if i.mnemonic == "get_sr"}))
    return G13Program(admission.name, code, admission.bindings, registers, admission.uniform_halfs,
                      reserved_register_halfs=(0, 1), builtins=builtins, bounds_policy="masked",
                      origin="Mesa 26.2.4 NIR -> AGX", logical_threads=admission.logical_threads)
