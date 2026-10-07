#!/usr/bin/env python3
"""Offline decoding of the observed IOGPU submission base record on 25G83.

This is a byte-layout decoder, not dispatch authorization or a validity checker.
It does not establish that identifiers belong to live registered objects, that
callbacks are callable, or that the driver-specific trailing fields are correct.
Evidence: g17-first-dispatch-trials.md and retained IOGPU code/ObjC metadata.
"""
import struct

BASE_BYTES = 48


def decode_record(data):
    if len(data) < BASE_BYTES:
        raise ValueError(f"submission base record needs 48 bytes, got {len(data)}")
    kernel, segment, sideband, padding0 = struct.unpack_from("<4I", data)
    scheduled, completed = struct.unpack_from("<2Q", data, 16)
    debug, padding1, tail = struct.unpack_from("<IIQ", data, 32)
    return {"kernel_shmem_id": kernel, "segment_shmem_id": segment,
            "sideband_shmem_id": sideband, "padding_0c": padding0,
            "scheduled_callback_or_token": scheduled,
            "completed_callback_or_token": completed,
            "debug_shmem_id": debug, "padding_24": padding1,
            "uninterpreted_qword_28": tail, "extension_hex": data[48:].hex()}


def decode_records(data, count, stride):
    if count < 0 or stride < BASE_BYTES:
        raise ValueError("invalid count or base-record stride")
    if len(data) != count * stride:
        raise ValueError("capture length must equal count times stride")
    return [decode_record(data[i*stride:(i+1)*stride]) for i in range(count)]


def decode_shmem_output(data):
    """Selector 14 structure output, from IOGPUDeviceCreateDeviceShmem data flow."""
    if len(data) != 16:
        raise ValueError("device shmem output must have exactly 16 bytes")
    address, size, identifier = struct.unpack("<QII", data)
    return {"cpu_address": address, "size": size, "shmem_id": identifier}
