"""Handwritten G13 bring-up kernels. These do not use the new IR compiler."""
from .abi import G13Program, Binding
from .encode import assemble

OPERATIONS = ("store", "lane", "group", "global", "copy", "add", "mul", "sub", "fma")


def program(operation, n):
    if operation not in OPERATIONS or n <= 0 or n % 32:
        raise ValueError("handwritten smoke kernels require complete 32-lane groups")
    lines = ["get_sr r0, sr80"]
    if operation == "store": lines += ["mov_imm r1, 0x3f800000, 0"]
    elif operation in ("lane", "group", "global"):
        lines += [f"get_sr r1, sr{dict(lane=52, group=0, global_=80).get(operation, 80)}"]
    else:
        lines += ["device_load 0, i32, x, r1, u4_u5, r0, unsigned, lsl 0", "wait 0"]
        if operation != "copy":
            lines += ["device_load 1, i32, x, r2, u2_u3, r0, unsigned, lsl 0", "wait 1"]
            lines += [{
                "add": "fadd32 r1, r1, r2", "mul": "fmul32 r1, r1, r2",
                "sub": "fadd32 r1, r1, r2.neg", "fma": "fmadd32 r1, r1, r2, -1.0"
            }[operation]]
    lines += ["device_store 0, i32, x, r1, u0_u1, r0, unsigned, lsl 0, 0", "wait 0", "stop"]
    return G13Program("handwritten_" + operation, assemble(lines) + bytes.fromhex("0800") * 8,
        (Binding("output", 0, "write", n * 4), Binding("b", 1, "read", n * 4), Binding("a", 2, "read", n * 4)),
        6, 12, builtins=tuple(sorted({80, {"lane": 52, "group": 0}.get(operation, 80)})),
        reserved_register_halfs=(), logical_threads=n)
