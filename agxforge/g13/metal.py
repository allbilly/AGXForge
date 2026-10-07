"""Author a G13 carrier archive with bounded, checked Mach-O traversal.

The container and metadata come from a local Apple-compiled carrier. Only its
_agc.main code range is replaced. This is a Metal execution path, not IOGPU.
"""
import struct

MAGIC = b"\xcf\xfa\xed\xfe"


def carrier_source():
    # Eight buffer pointers occupy u0..u15 (32 uniform halfwords). Enough live
    # scalar values reserve more registers and text than the selected kernels.
    lines = ["#include <metal_stdlib>", "using namespace metal;", "kernel void carrier("]
    lines += [f"device uint *p{i} [[buffer({i})]]," for i in range(8)]
    lines += ["uint tid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]],",
              "uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],",
              "uint simd [[simdgroup_index_in_threadgroup]]) {"]
    lines += [f"uint v{i} = p{i % 8}[tid + {i * 32}];" for i in range(40)]
    for round_ in range(8):
        lines += [f"v{i} = (v{i} * {1664525 + 2 * i}u + v{(i + 13) % 40}) ^ {1013904223 + round_}u;"
                  for i in range(40)]
    lines += [f"p{i}[tid] = " + " ^ ".join(f"v{j}" for j in range(i, 40, 8)) +
              " ^ lid ^ gid ^ lane ^ simd;" for i in range(8)]
    return "\n".join(lines + ["}"]) + "\n"


def code_range(archive):
    """Return the unique native _agc.main range; malformed containers fail closed."""
    candidates = []
    start = archive.find(MAGIC)
    while start >= 0:
        obj = memoryview(archive)[start:]
        try:
            _, cpu, _, _, ncmds, commands_size, _, _ = struct.unpack_from("<8I", obj)
            if cpu != 0x1000013 or ncmds > 256 or commands_size > len(obj) - 32:
                raise ValueError("not a bounded Apple GPU object")
            off, text, symbols, section_count = 32, None, None, 0
            for _ in range(ncmds):
                cmd, size = struct.unpack_from("<II", obj, off)
                if size < 8 or off + size > 32 + commands_size:
                    raise ValueError("invalid load command")
                if cmd == 0x19:
                    if size < 72: raise ValueError("truncated segment command")
                    nsections, = struct.unpack_from("<I", obj, off + 64)
                    if 72 + nsections * 80 > size:
                        raise ValueError("invalid section count")
                    for index in range(nsections):
                        pos = off + 72 + index * 80
                        name = bytes(obj[pos:pos+16]).rstrip(b"\0")
                        segment = bytes(obj[pos+16:pos+32]).rstrip(b"\0")
                        addr, length, fileoff = struct.unpack_from("<QQI", obj, pos + 32)
                        if fileoff + length > len(obj): raise ValueError("section outside archive")
                        if name == b"__text" and segment == b"__TEXT":
                            if text is not None: raise ValueError("ambiguous native text")
                            text = (addr, fileoff, length, section_count+index+1)
                    section_count += nsections
                elif cmd == 2:
                    if size < 24 or symbols is not None: raise ValueError("invalid symbol command")
                    symbols = struct.unpack_from("<4I", obj, off + 8)
                off += size
            if off != 32 + commands_size or text is None or symbols is None:
                raise ValueError("missing native text/symbols")
            symoff, count, stroff, strlen = symbols
            if count > 65536 or symoff + count * 16 > len(obj) or stroff + strlen > len(obj):
                raise ValueError("symbol table outside archive")
            for index in range(count):
                nameoff, kind, section, _, value = struct.unpack_from("<IBBHQ", obj, symoff + index * 16)
                if nameoff >= strlen: raise ValueError("string offset outside table")
                end = archive.find(b"\0", start + stroff + nameoff, start + stroff + strlen)
                if end < 0: raise ValueError("unterminated symbol")
                name = archive[start + stroff + nameoff:end]
                if name == b"_agc.main" and kind & 0x0e == 0x0e and section:
                    if section != text[3]: raise ValueError("main outside native text section")
                    relative = value - text[0]
                    if not 0 <= relative < text[2]: raise ValueError("entry outside native text")
                    candidates.append((start + text[1] + relative, text[2] - relative))
        except (ValueError, struct.error):
            pass # Outer AIR/container objects are not native shader objects.
        start = archive.find(MAGIC, start + 4)
    if len(candidates) != 1:
        raise ValueError(f"expected one G13 native main, found {len(candidates)}")
    return candidates[0]


def replace_code(archive, code):
    offset, capacity = code_range(archive)
    if len(code) > capacity: raise ValueError("shader exceeds carrier text capacity")
    return archive[:offset] + code + archive[offset + len(code):], offset, capacity
