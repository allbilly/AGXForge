"""Strict G13 instruction framing and validation; no GPUCompiler.framework."""
from dataclasses import dataclass
import re
from ._vendor import applegpu

ALLOWED = frozenset({
    "mov_imm", "get_sr", "bitop", "iadd", "imadd", "bfi", "bfeil", "asr",
    "fadd32", "fmul32", "fmadd32", "fadd16", "fmul16", "fmadd16",
    "device_load", "device_store", "wait", "stop", "trap", "convert",
    "icmpsel", "fcmpsel", "if_icmp", "if_fcmp", "else_icmp", "else_fcmp",
    "pop_exec", "while_icmp", "while_fcmp", "jmp_exec_any", "jmp_exec_none",
    "rcp", "rsqrt", "exp2", "sin_pt_1", "sin_pt_2", "log2", "floor"
})


@dataclass(frozen=True)
class Decoded:
    offset: int
    size: int
    mnemonic: str
    assembly: str
    fields: dict


def decode(code):
    offset, result = 0, []
    while offset < len(code):
        number = applegpu.opcode_to_number(code[offset:])
        for desc in applegpu.instruction_descriptors:
            if desc.matches(number):
                size = desc.decode_size(number)
                if offset + size > len(code):
                    raise ValueError(f"truncated instruction at {offset}")
                # Decode only this instruction, not bits belonging to its successor.
                number = applegpu.opcode_to_number(code[offset:offset + size])
                if desc.decode_remainder(number):
                    raise ValueError(f"unowned encoding bits at {offset}")
                result.append(Decoded(offset, size, desc.name,
                    str(desc.disassemble(number, pc=offset)), dict(desc.decode_fields(number))))
                offset += size
                break
        else:
            raise ValueError(f"unknown instruction at {offset}")
    return tuple(result)


def validate(code, register_halfs, uniform_halfs):
    listing = decode(code)
    boundaries = {i.offset for i in listing}
    stopped = False
    for ins in listing:
        if ins.mnemonic not in ALLOWED:
            raise ValueError(f"unsupported G13 form {ins.mnemonic}")
        if stopped and ins.mnemonic != "trap":
            raise ValueError("only trap padding may follow stop")
        if ins.mnemonic == "trap" and not stopped:
            raise ValueError("trap is supported only as unreachable padding")
        if ins.mnemonic == "stop":
            stopped = True
        if ins.mnemonic.startswith("jmp_exec"):
            delta = applegpu.sign_extend(ins.fields["off"], 32)
            if ins.offset + delta not in boundaries:
                raise ValueError("branch target is not an instruction boundary")
        operands = ins.assembly.split(None, 1)[1].split(",") if len(ins.assembly.split(None, 1)) == 2 else []
        for operand in operands:
            if not re.fullmatch(r"[ru]\d+[lh]?(?:_[ru]\d+[lh]?)*(?:\.[a-z]+)*", operand.strip()):
                continue
            for prefix, number, half in re.findall(r"([ru])(\d+)([lh]?)", operand):
                high = int(number) * 2 + (1 if half == "l" else 2)
                footprint = register_halfs if prefix == "r" else uniform_halfs
                if high > footprint:
                    raise ValueError(f"{prefix}{number}{half} exceeds declared register footprint")
    if not stopped:
        raise ValueError("program has no stop")
    return listing


def disassemble(code):
    return "\n".join(f"{i.offset:04x}: {code[i.offset:i.offset+i.size].hex():24} {i.assembly}" for i in decode(code)) + "\n"
