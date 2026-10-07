"""Encode the supported G13 forms with the pinned applegpu reference tables."""
from ._vendor import applegpu
import re


def instruction(line):
    try:
        return _instruction(line)
    except Exception as error:
        raise ValueError(f"invalid G13 instruction {line!r}: {error}") from error


def _instruction(line):
    parts = line.strip().split(None, 1)
    if not parts:
        raise ValueError("empty instruction")
    mnemonic = parts[0]
    operands = [s.strip() for s in parts[1].split(",")] if len(parts) == 2 else []
    for desc in applegpu.instruction_descriptors:
        fields = desc.fields_for_mnem(mnemonic, operands)
        if fields is None:
            continue
        rewritten = desc.rewrite_operands_strings(mnemonic, operands)
        if len(rewritten) > len(desc.ordered_operands):
            raise ValueError(f"too many operands: {line}")
        for operand, spelling in zip(desc.ordered_operands, rewritten):
            if isinstance(operand, applegpu.IntegerFieldDesc):
                value = applegpu.try_parse_integer(spelling)
                if value is None or not 0 <= value < (1 << operand.size):
                    raise ValueError(f"out-of-range immediate: {spelling}")
            operand.encode_string(fields, spelling)
        for operand in desc.ordered_operands[len(rewritten):]:
            operand.encode_string(fields, "")
        for name, subfields in desc.merged_fields:
            size = sum(width for _, width, field in desc.fields if field in {n for n, _ in subfields})
            if not 0 <= fields[name] < (1 << size):
                raise ValueError(f"out-of-range field {name}: {line}")
        encoded = desc.to_bytes(desc.encode_fields(fields))
        decoded_fields = dict(desc.decode_fields(applegpu.opcode_to_number(encoded)))
        for operand, spelling in zip(desc.ordered_operands, rewritten):
            # Upstream's research assembler may silently ignore source modifiers or
            # truncate registers. Require exact register/flag preservation here.
            if re.fullmatch(r"[ru]\d+[lh]?(?:_[ru]\d+[lh]?)*(?:\.[a-z]+)*", spelling):
                if str(operand.decode(decoded_fields)) != spelling:
                    raise ValueError(f"register operand changed during encoding: {spelling}")
        return encoded
    raise ValueError(f"unsupported instruction: {line}")


def assemble(lines):
    """Two-pass label resolution. Branch displacements are relative to their PC."""
    labels, pending, offset = {}, [], 0
    for text in lines:
        line = text.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            label = line[:-1]
            if label in labels:
                raise ValueError(f"duplicate label {label}")
            labels[label] = offset
            continue
        encoded = instruction(line.split("@", 1)[0] + "pc+0") if "@" in line else instruction(line)
        pending.append((offset, line, len(encoded)))
        offset += len(encoded)
    result = bytearray()
    for pc, line, size in pending:
        if "@" in line:
            prefix, label = line.split("@")
            if label not in labels:
                raise ValueError(f"unknown label {label}")
            delta = labels[label] - pc
            line = prefix + (f"pc+{delta}" if delta >= 0 else f"pc-{abs(delta)}")
        encoded = instruction(line)
        if len(encoded) != size:
            raise ValueError("label resolution changed instruction size")
        result.extend(encoded)
    return bytes(result)
