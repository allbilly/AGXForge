"""Direct base-M1 IOGPU executor for the measured macOS 26A434 envelope.

Shader, uniform, register and CDM fields are authored per program. Resources and
the persistent queue use a measured driver envelope with fresh CPU/GPU addresses.
No Metal API or AGXMetal driver supplies dispatch. On macOS 27 the system IOKit
dependency closure can incidentally load Metal.framework.
"""
# SPDX-License-Identifier: MIT
# USC/CDM fields: Copyright 2021-2025 Alyssa Rosenzweig;
# Copyright 2023-2025 Valve Corporation (Mesa cmdbuf.xml).
# See experimental/macos-launch/THIRD-PARTY-NOTICES.md.
import base64
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import zlib
from .macos import Executor as MetalExecutor, Buffer as MetalBuffer, ROOT, source_identity, acquire_gpu_locks
from agxforge.g13.abi import G13Program
from agxforge.g13.decode import disassemble

EXECUTOR_NAME = "direct IOGPU G13"
TEMPLATE = Path(__file__).with_name("iogpu-26A434.json")
SUPPORT = ROOT/"build/macos/iogpu-support.bin"
CODE_START = 0x5000
CODE_BYTES = 65536


def load_support():
    try:
        data = SUPPORT.read_bytes()
        receipt = json.loads(SUPPORT.with_suffix(".json").read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("prepare the local driver support first: make macos-support") from error
    if (receipt.get("schema_version") != 1 or receipt.get("build") != "26A434" or
            receipt.get("gpu") != "Apple M1" or receipt.get("code_start") != CODE_START or
            receipt.get("bytes") != CODE_START or len(data) != CODE_START or
            receipt.get("sha256") != hashlib.sha256(data).hexdigest() or
            receipt.get("template_sha256") != hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() or
            not any(data[64:0x2840])):
        raise RuntimeError("local driver support is incompatible or changed; run make macos-support")
    return data, receipt


def append_arena_resource(page, slot):
    # 26A434 adds a 24-byte list prefix to the Mesa IOAccel segment header.
    # The measured list has 16 resources in three fixed 64-byte groups.
    if (len(page) != 16384 or struct.unpack_from("<I",page,0x24)[0] != 0x800000f0 or
            struct.unpack_from("<II",page,0x40) != (16,3) or slot != 27 or
            any(page[0x108:0x148])):
        raise ValueError("native residency list profile changed")
    data = bytearray(page)
    struct.pack_into("<I",data,0x24,0x80000130)
    struct.pack_into("<II",data,0x40,17,4)
    struct.pack_into("<I",data,0x108,slot)
    struct.pack_into("<I",data,0x120,0x20)
    struct.pack_into("<H",data,0x138,1)
    struct.pack_into("<H",data,0x146,1)
    return bytes(data)


def native_library():
    if platform.system() != "Darwin": raise RuntimeError("macOS required")
    if subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip() != "26A434":
        raise RuntimeError("native launch envelope is measured on macOS build 26A434 only")
    path = ROOT/"build/macos/libagxforge_iogpu.dylib"
    if not path.is_file(): raise RuntimeError("build the native helper first: make macos-tools")
    lib = C.CDLL(str(path))
    signatures = {
        "an_open": (C.c_void_p, []), "an_close": (None, [C.c_void_p]),
        "an_error": (C.c_char_p, []), "an_no_agx_metal": (C.c_int, []),
        "an_call": (C.c_int, [C.c_void_p, C.c_uint32, C.POINTER(C.c_uint64), C.c_uint32,
                               C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32]),
        "an_submit": (C.c_int, [C.c_void_p, C.c_void_p, C.c_uint32]),
        "an_wait": (C.c_int, [C.c_void_p, C.c_void_p, C.c_uint64, C.c_uint64, C.c_void_p, C.c_uint32])
    }
    for name, (restype, argtypes) in signatures.items():
        method = getattr(lib, name); method.restype = restype; method.argtypes = argtypes
    return lib


class Relocations:
    def __init__(self): self.ranges = []; self.exact = {}
    def add(self, old, new, size=0):
        self.exact[old] = new
        if size: self.ranges.append((old,new,size))
    def address(self, value):
        if value in self.exact: return self.exact[value]
        for old,new,size in self.ranges:
            if old <= value < old+size: return new+value-old
        return value
    def patch(self, raw):
        data = bytearray(raw)
        # Match candidates against the original bytes. Overlapping 64-bit fields
        # must never turn a previous patch into a second relocation candidate.
        for offset in range(0,len(raw)-7,4):
            old, = struct.unpack_from("<Q",raw,offset)
            new = self.address(old)
            if old != new: struct.pack_into("<Q",data,offset,new)
        return bytes(data)


class Buffer(MetalBuffer):
    def __init__(self, executor, size, access="read_write", data=None):
        executor._require_open()
        if type(size) is not int or size <= 0 or access not in ("read","write","read_write"):
            raise ValueError("positive extent and tensor access mode required")
        self.executor, self.size, self.access = executor, size, access
        self.offset, self.allocated_size = 64, size+128
        cursor = (executor.cursor+63)//64*64
        if cursor+self.allocated_size > executor.arena_bytes: raise ValueError("native tensor arena exhausted")
        self.cpu = executor.arena_cpu+cursor
        self.raw_address = executor.arena_gpu+cursor
        self.address = self.raw_address+self.offset
        executor.cursor = cursor+self.allocated_size
        executor.buffers.append(self)
        C.memset(self.cpu,0xa5,self.allocated_size)
        if data is not None: self.write(data)


class Executor(MetalExecutor):
    def __init__(self, device=None, va_slot=0, timeout_ms=5000, arena_bytes=32*1024*1024):
        if device is not None or type(va_slot) is not int or not 0 <= va_slot < 16:
            raise ValueError("default base M1; va_slot 0..15 required")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000: raise ValueError("timeout 1..60000 ms required")
        if type(arena_bytes) is not int or not 128*1024 <= arena_bytes <= 2**31 or arena_bytes % 16384:
            raise ValueError("arena must be page aligned in 128 KiB..2 GiB")
        support, self.support_receipt = load_support()
        self.lib, self.handle = native_library(), None
        self.buffers, self.locks, self.keep, self.pages = [], [], [], []
        self.poisoned, self.timeout_ms, self.arena_bytes = False, timeout_ms, arena_bytes
        self.cursor = 65536+va_slot*16384
        self.relocations, self.regions, self.gpu_regions = Relocations(), {}, {}
        self.code_cursor, self.programs = CODE_START, {}
        self.trace_cursor = self.trace_end = 0
        self.template = json.loads(TEMPLATE.read_text())
        try:
            acquire_gpu_locks(self.locks)
            self.handle=self.lib.an_open()
            if not self.handle: self.raise_error()
            for event in self.template["events"]:
                if event["kind"] == "call": self.setup_call(event)
                elif event["kind"] == "memory": self.setup_memory(event)
                elif event["kind"] == "submit":
                    self.submit_record=C.create_string_buffer(self.relocations.patch(bytes.fromhex(event["record"])))
            self.code_cpu=self.relocations.address(self.template["code_cpu"])
            self.usc_cpu=self.relocations.address(self.template["usc_cpu"])
            self.cdm_cpu=self.relocations.address(self.template["cdm_cpu"])
            self.uniform_cpu=self.relocations.address(self.template["uniform_cpu"])
            self.code_gpu_base=self.gpu_regions[self.template["code_cpu"]]
            self.usc_gpu=self.gpu_regions[self.template["usc_cpu"]]
            self.uniform_gpu=self.gpu_regions[self.template["uniform_cpu"]-0x7fa0]+0x7fa0
            C.memmove(self.code_cpu,support,len(support))
            self.allocate_arena()
            self.tokens=struct.unpack_from("<QQ",self.submit_record.raw,16)
        except BaseException: self.close();raise

    def raise_error(self): raise RuntimeError(self.lib.an_error().decode())
    def close(self):
        if self.handle: self.lib.an_close(self.handle);self.handle=None
        for stream in self.locks:stream.close()
        self.locks.clear();self.keep.clear()
    def buffer(self,size,access="read_write",data=None):
        self._require_open()
        if self.poisoned:raise RuntimeError("executor stopped after a failed dispatch")
        return Buffer(self,size,access,data)

    def setup_call(self,event):
        raw=bytes.fromhex(event["input"]);old_out=bytes.fromhex(event["output"])
        request=bytearray(self.relocations.patch(raw))
        scalars=(C.c_uint64*len(event["scalars"]))(*event["scalars"])
        output=C.create_string_buffer(len(old_out)) if old_out else None
        if self.lib.an_call(self.handle,event["selector"],scalars,len(scalars),bytes(request),len(request),output,len(old_out)):
            self.raise_error()
        live=output.raw if output else b""
        if event["selector"]==9:
            old_gpu,old_cpu,old_desc=struct.unpack_from("<3Q",old_out)
            gpu,cpu,desc=struct.unpack_from("<3Q",live)
            old_size,=struct.unpack_from("<Q",old_out,40);size,=struct.unpack_from("<Q",live,40)
            if struct.unpack_from("<I",old_out,36)[0]!=struct.unpack_from("<I",live,36)[0]:
                raise RuntimeError("native resource slot layout changed")
            if old_cpu:
                if not cpu or size!=old_size:raise RuntimeError("native allocation extent changed")
                self.relocations.add(old_cpu,cpu,old_size);self.gpu_regions[old_cpu]=gpu
            self.relocations.add(old_desc,desc)
            if old_gpu:self.relocations.add(old_gpu,gpu,old_size if old_cpu else 0)
        elif event["selector"] in (14,16):
            old,=struct.unpack_from("<Q",old_out);address,=struct.unpack_from("<Q",live)
            if not address:raise RuntimeError("missing native shared page")
            self.relocations.add(old,address,16384)
            if event["selector"]==16:self.ring=address
            else:self.regions[old]=address
        elif event["selector"]==7 and struct.unpack_from("<I",live)[0]!=1:
            raise RuntimeError("native queue profile changed")

    def allocate_arena(self):
        event = next(e for e in self.template["events"] if e["kind"] == "call" and e["selector"] == 9 and
                     struct.unpack_from("<Q",bytes.fromhex(e["output"]),8)[0] == self.template["arena_cpu"])
        request = bytearray.fromhex(event["input"])
        struct.pack_into("<Q",request,72,self.arena_bytes)
        output = C.create_string_buffer(88)
        if self.lib.an_call(self.handle,9,None,0,bytes(request),len(request),output,88):self.raise_error()
        self.arena_gpu,self.arena_cpu = struct.unpack_from("<QQ",output.raw)
        self.arena_slot, = struct.unpack_from("<I",output.raw,36)
        size, = struct.unpack_from("<Q",output.raw,40)
        if not self.arena_cpu or size != self.arena_bytes or self.arena_slot != 27:
            raise RuntimeError("native tensor arena allocation profile changed")
        address,page = self.pages[0]
        self.pages[0] = (address,append_arena_resource(page,self.arena_slot))

    def launch_pages(self):
        if self.trace_cursor+2 > self.trace_end:
            output = C.create_string_buffer(16)
            if self.lib.an_call(self.handle,6,None,0,None,0,output,16):self.raise_error()
            self.trace_cursor,self.trace_end = struct.unpack("<QQ",output.raw)
            if not 0 < self.trace_cursor < self.trace_end <= 2**32:
                raise RuntimeError("native trace ID range profile changed")
        command,encoder = self.trace_cursor,self.trace_cursor+1
        self.trace_cursor += 2
        pages = [bytearray(data) for _,data in self.pages]
        for offset in (0,0x18):struct.pack_into("<Q",pages[0],offset,command)
        struct.pack_into("<Q",pages[0],0x28,encoder)
        struct.pack_into("<I",pages[1],0x23c,encoder)
        return (command,encoder), [bytes(page) for page in pages]

    def setup_memory(self,event):
        raw=zlib.decompress(base64.b64decode(event["data"]))
        if len(raw)!=event["bytes"]:raise ValueError("truncated native envelope")
        if event["memory_kind"]==2:
            storage=C.create_string_buffer(raw);self.keep.append(storage)
            self.relocations.add(event["address"],C.addressof(storage),len(raw))
        address=self.relocations.address(event["address"])
        if address==event["address"]:raise ValueError("unrelocated native CPU image")
        data=self.relocations.patch(raw)
        C.memmove(address,data,len(data))
        if event["address"] in self.regions:self.pages.append((address,data))

    def platform(self):
        self._require_open()
        return dict(system="Darwin",macos=platform.mac_ver()[0],build="26A434",gpu="Apple M1",driver="IOGPU",
                    executor=EXECUTOR_NAME,kernel=platform.uname()._asdict(),python=platform.python_version(),
                    page_bytes=os.sysconf("SC_PAGE_SIZE"),agx_metal_absent=bool(self.lib.an_no_agx_metal()),
                    metal_api_calls=0,arena_bytes=self.arena_bytes,arena_resource_slot=self.arena_slot,
                    support=self.support_receipt,
                    template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest())

    def prepare(self, program):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        if not isinstance(program, G13Program): raise ValueError("validated G13Program required")
        if (len(program.bindings) > 8 or program.register_halfs > 80 or
                program.workgroup_size != 32):
            raise ValueError("IOGPU profile: at most 8 buffers, 80 register halfwords and 32 lanes")
        if program.code_hash not in self.programs:
            offset = (self.code_cursor+63)//64*64
            if offset+len(program.code) > CODE_BYTES:
                raise ValueError("native code heap exhausted")
            # The envelope invokes its helper at 0x40. Keep that driver code
            # intact and give each compiled program an immutable entry.
            C.memmove(self.code_cpu+offset,program.code,len(program.code))
            self.programs[program.code_hash] = offset
            self.code_cursor = offset+len(program.code)
        return self.programs[program.code_hash]

    def dispatch(self,program,buffers,threads,evidence=None,capture_buffers=True):
        self._require_open()
        if self.poisoned:raise RuntimeError("executor stopped after a failed dispatch")
        code_offset = self.prepare(program)
        if type(threads) is not int or threads!=program.logical_threads or len(buffers)!=len(program.bindings):
            raise ValueError("launch extent or binding count disagrees with program")
        if program.bounds_policy=="complete_groups" and threads%program.workgroup_size:raise ValueError("unmasked shader requires complete groups")
        needs={"read":{"read"},"write":{"write"},"read_write":{"read","write"}}
        for binding in program.bindings:
            buf=buffers[binding.slot]
            if buf.executor is not self or buf.size<binding.min_bytes or buf.address%binding.alignment:
                raise ValueError("buffer ownership, extent or alignment")
            if not needs[binding.access]<=needs[buf.access]:raise ValueError("buffer access contract")
        import time
        global_x=(threads+program.workgroup_size-1)//program.workgroup_size*program.workgroup_size
        pointers=struct.pack("<"+"Q"*len(buffers),*(buf.address for buf in buffers))
        C.memmove(self.uniform_cpu,pointers,len(pointers))
        code_gpu=self.code_gpu_base+code_offset
        reserved_uniform_halfs = 32
        C.memset(self.uniform_cpu+len(pointers),0,reserved_uniform_halfs*2-len(pointers))
        usc=struct.pack("<Q",0x1d|(reserved_uniform_halfs<<20)|((self.uniform_gpu>>2)<<26))
        usc+=struct.pack("<IHIIH",0x904d,0x0c0d,code_gpu,0x0100108d,0x88)
        cdm=struct.pack("<11I",0x1004,self.usc_gpu,global_x,1,1,
                        program.workgroup_size,1,1,0x60000168,0x40000000,0)
        C.memmove(self.usc_cpu,usc,len(usc));C.memmove(self.cdm_cpu,cdm,len(cdm))
        trace_ids,pages = self.launch_pages()
        for (address,_),data in zip(self.pages,pages):C.memmove(address,data,len(data))
        states=[("usc.bin",usc),("cdm.bin",cdm),("iogpu-submit.bin",self.submit_record.raw[:64]),
                ("iogpu-pages.bin",b"".join(pages))]
        record=dict(status="PREPARED",executor=EXECUTOR_NAME,program=program.descriptor(),logical_threads=threads,
                    global_threads=[global_x,1,1],workgroup_size=[program.workgroup_size,1,1],
                    group_count=[global_x//program.workgroup_size,1,1],
                    buffers=[dict(binding=b.name,**buffers[b.slot].record()) for b in program.bindings],
                    shader_gpu_address=code_gpu,usc_gpu_address=self.usc_gpu,uniforms_gpu_address=self.uniform_gpu,
                    notification_tokens=list(self.tokens), uniform_register_halfs_reserved=32,
                    trace_ids=list(trace_ids),
                    register_halfs_reserved=128,
                    support_sha256=self.support_receipt["sha256"],
                    template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest())
        folder=Path(evidence) if evidence else None
        if folder:
            folder.mkdir(parents=True,exist_ok=False)
            (folder/"shader.bin").write_bytes(program.code);(folder/"shader.txt").write_text(disassemble(program.code))
            (folder/"uniforms.bin").write_bytes(pointers)
            record["uniforms.bin"]=dict(sha256=hashlib.sha256(pointers).hexdigest(),bytes=len(pointers))
            for name,data in states:
                (folder/name).write_bytes(data);record[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            if capture_buffers:
                for b in program.bindings:(folder/f"slot-{b.slot}-before.bin").write_bytes(buffers[b.slot].read())
            (folder/"launch.json").write_text(json.dumps(record,indent=2)+"\n")
        start=time.monotonic_ns()
        try:
            if self.lib.an_submit(self.handle,self.submit_record,64):self.raise_error()
            receipt=C.create_string_buffer(80)
            wait_status=self.lib.an_wait(self.handle,self.ring,*self.tokens,receipt,self.timeout_ms)
            record["completion"]=receipt.raw.hex()
            if wait_status:self.raise_error()
            if not all(buf.canaries_ok() for buf in buffers):raise RuntimeError("GPU buffer canary changed")
            record.update(status="COMPLETED",fence_completed=True,canaries_intact=True,
                          completion=receipt.raw.hex(),host_submit_wait_ns=time.monotonic_ns()-start)
            if folder and capture_buffers:
                for b in program.bindings:(folder/f"slot-{b.slot}-after.bin").write_bytes(buffers[b.slot].read())
        except BaseException as error:
            self.poisoned=True;record.update(status="FAIL",error=str(error));raise
        finally:
            if folder:(folder/"launch.json").write_text(json.dumps(record,indent=2)+"\n")
        return record
