#!/usr/bin/env python3
"""Freeze the measured base-M1 26A434 launch envelope from an AGXC v2 capture.

Capture schema and relocation were inspected in the user's applegpu experiments
(experimental/iokit_capture.c and cap_format.py). This parser is independent.
The result contains launch state, not an Apple framework or shader library.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import zlib


def read_capture(data):
    if struct.unpack_from("<4I", data)[:2] != (0x43584741, 2): raise ValueError("expected AGXC v2")
    offset, events = 16, []
    def read(size):
        nonlocal offset
        if size < 0 or offset + size > len(data): raise ValueError("truncated capture")
        result = data[offset:offset+size]; offset += size; return result
    def unpack(fmt): return struct.unpack(fmt, read(struct.calcsize(fmt)))
    while offset < len(data):
        kind = data[offset]
        if kind == 1:
            _, client, status, _ = unpack("<B3xIiI")
            if client != 0x100005 or status: raise ValueError("wrong/failed user client")
        elif kind == 2:
            _, _, selector, count, size = unpack("<B3xIIII")
            scalars = list(unpack("<"+"Q"*count)); payload = read(size)
            status, count, size = unpack("<iII")
            outputs = list(unpack("<"+"Q"*count)); output = read(size)
            if status or outputs: raise ValueError("unsupported/failed call")
            if selector not in (8,17):
                events.append(dict(kind="call", selector=selector, scalars=scalars, input=payload.hex(), output=output.hex()))
        elif kind == 4:
            _, memory_kind, _, _, address, size = unpack("<BBHIQQ")
            payload = read(size)
            if memory_kind not in (1,2): raise ValueError("unsupported snapshot kind")
            events.append(dict(kind="memory", memory_kind=memory_kind, address=address, bytes=size,
                               data=base64.b64encode(zlib.compress(payload)).decode()))
        elif kind == 3:
            _, _, trap, queue, size, _, fourth, snap_size = unpack("<B3xII4xQQQQI4x")
            payload = read(snap_size); status, = unpack("<i")
            if trap or queue != 1 or size != 64 or status or snap_size != 64:
                raise ValueError("unsupported/failed submission")
            events.append(dict(kind="submit", record=payload.hex()))
        else: raise ValueError(f"unknown capture event {kind}")
    return events


def freeze(path, output):
    if output.exists(): raise ValueError("choose a new template destination")
    raw = path.read_bytes(); events = read_capture(raw)
    if sum(e["kind"] == "submit" for e in events) != 1: raise ValueError("one submission required")
    parents = {}
    calls = [e for e in events if e["kind"] == "call"]
    for e in calls:
        payload = bytearray.fromhex(e["input"])
        if e["selector"] == 9:
            gpu, cpu = struct.unpack_from("<QQ", bytes.fromhex(e["output"]))
            if cpu: parents[cpu] = gpu
            payload[96:104] = bytes(8) # Userspace Metal-object pointers are not used by this transport.
        elif e["selector"] == 7:
            payload[:0x400] = bytes(0x400)
            label = b"AGXForge G13 native"
            payload[:len(label)] = label
        e["input"] = payload.hex()
    for e in events:
        if e["kind"] == "memory" and (e["memory_kind"] == 2 or parents.get(e["address"]) == 0):
            e["data"] = base64.b64encode(zlib.compress(bytes(e["bytes"]))).decode()
    result = dict(schema_version=1, profile="base M1 G13G / macOS 26A434",
                  capture_sha256=hashlib.sha256(raw).hexdigest(), events=events,
                  arena_cpu=next(cpu for cpu,gpu in parents.items() if gpu == 0x1500000000),
                  code_cpu=next(cpu for cpu,gpu in parents.items() if gpu == 0),
                  usc_cpu=next(cpu for cpu,gpu in parents.items() if gpu == 0x18000),
                  cdm_cpu=next(cpu for cpu,gpu in parents.items() if gpu == 0x1500088000),
                  uniform_cpu=next(cpu for cpu,gpu in parents.items() if gpu == 0x15000b8000)+0x7fa0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); freeze(args.capture, args.output)
