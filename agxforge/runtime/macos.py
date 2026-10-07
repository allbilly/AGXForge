"""Persistent base-M1 G13 executor using authored Metal binary archives."""
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from agxforge.g13.abi import G13Program
from agxforge.g13.decode import disassemble
from agxforge.g13.metal import carrier_source, code_range, replace_code

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_NAME = "G13 machine code through Metal"


def acquire_gpu_locks(streams, timeout=30):
    deadline = time.monotonic()+timeout
    for path in (Path.home()/"gpu.lock", Path("/tmp/m1-gpu.lock"),
                 Path.home()/".cache/agxforge/g17-execution.lock"):
        # flock does not need a writable descriptor for an existing lock file.
        # Keep the same locks when running with read-only access to the home directory.
        if path.exists(): stream = path.open("r")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a")
        streams.append(stream)
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() >= deadline: raise TimeoutError("GPU lock held by another process")
                time.sleep(.1)


def native_library():
    if platform.system() != "Darwin": raise RuntimeError("macOS is required")
    path = ROOT / "build/macos/libagxforge_macos.dylib"
    if not path.is_file(): raise RuntimeError("build the macOS helper first: make macos-tools")
    lib = C.CDLL(str(path))
    signatures = {
        "am_open": (C.c_void_p, []), "am_close": (C.c_int, [C.c_void_p]),
        "am_error": (C.c_char_p, []), "am_name": (C.c_char_p, [C.c_void_p]),
        "am_alloc": (C.c_void_p, [C.c_void_p, C.c_uint64]),
        "am_map": (C.c_void_p, [C.c_void_p]), "am_address": (C.c_uint64, [C.c_void_p]),
        "am_carrier": (C.c_int, [C.c_void_p, C.c_char_p, C.c_char_p]),
        "am_pipeline": (C.c_void_p, [C.c_void_p, C.c_char_p, C.c_char_p]),
        "am_submit": (C.c_int, [C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p)] + [C.c_uint32] * 5)
    }
    for name, (restype, argtypes) in signatures.items():
        method = getattr(lib, name); method.restype = restype; method.argtypes = argtypes
    return lib


def source_identity():
    paths = list((ROOT / "agxforge/g13").rglob("*.py"))
    paths += list((ROOT / "agxforge/runtime").glob("macos.*"))
    paths += list((ROOT / "agxforge/runtime").glob("iogpu*"))
    paths += list((ROOT / "examples/asahi").glob("*.py"))
    paths += list((ROOT / "examples/macos").glob("*.py"))
    paths += [ROOT / "Makefile", ROOT / "requirements-asahi.txt", ROOT / "agxforge/g17/ir.py",
              ROOT / "build/macos/libagxforge_macos.dylib",
              ROOT / "build/macos/libagxforge_iogpu.dylib",
              ROOT / "build/macos/iogpu-support.bin", ROOT / "build/macos/iogpu-support.json",
              ROOT / "tools/macos_support.c", ROOT / "tools/macos_support.py",
              ROOT / "build/macos/libagxforge_support.dylib",
              ROOT / "tools/asahi_evidence.py", ROOT / "tools/macos_capture.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


class Buffer:
    def __init__(self, executor, size, access="read_write", data=None):
        executor._require_open()
        if type(size) is not int or size <= 0 or access not in ("read", "write", "read_write"):
            raise ValueError("positive extent and a tensor access mode required")
        self.executor, self.size, self.access = executor, size, access
        self.offset, self.allocated_size = 64, size + 128
        self.handle = executor.lib.am_alloc(executor.handle, self.allocated_size)
        if not self.handle: executor.raise_error()
        executor.buffers.append(self)
        self.cpu = executor.lib.am_map(self.handle)
        self.raw_address = executor.lib.am_address(self.handle)
        self.address = self.raw_address + self.offset
        C.memset(self.cpu, 0xa5, self.allocated_size)
        if data is not None: self.write(data)

    def write(self, data, offset=0):
        self.executor._require_open()
        if offset < 0 or offset + len(data) > self.size: raise ValueError("CPU upload exceeds buffer extent")
        C.memmove(self.cpu + self.offset + offset, data, len(data))

    def read(self, size=None, offset=0):
        self.executor._require_open()
        size = self.size - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > self.size: raise ValueError("CPU read exceeds buffer extent")
        return C.string_at(self.cpu + self.offset + offset, size)

    def canaries_ok(self):
        self.executor._require_open()
        return (C.string_at(self.cpu, self.offset) == b"\xa5" * self.offset and
                C.string_at(self.cpu + self.offset + self.size, 64) == b"\xa5" * 64)

    def record(self):
        self.executor._require_open()
        return dict(gpu_address=self.address, mapping_address=self.raw_address,
                    payload_offset=self.offset, extent=self.size, allocation_bytes=self.allocated_size, access=self.access)


class Executor:
    def __init__(self, device=None, va_slot=0, timeout_ms=5000):
        if device is not None: raise ValueError("only the default base M1 device is supported")
        if type(va_slot) is not int or not 0 <= va_slot < 16: raise ValueError("va_slot must be 0..15")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000: raise ValueError("timeout must be 1..60000 ms")
        self.lib, self.buffers, self.programs, self.locks = native_library(), [], {}, []
        self.handle, self.poisoned, self.timeout_ms = None, False, timeout_ms
        self.temp = tempfile.TemporaryDirectory(prefix="agxforge-macos-")
        self.directory = Path(self.temp.name)
        self.template = None
        try:
            acquire_gpu_locks(self.locks)
            self.handle = self.lib.am_open()
            if not self.handle: self.raise_error()
            # Vary allocation placement without claiming control over Metal's VA allocator.
            self.padding = [self.buffer(16384) for _ in range(va_slot)]
            self.dummy = self.buffer(4)
        except BaseException:
            self.close(); raise

    def _require_open(self):
        if not self.handle: raise RuntimeError("executor is closed")
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")

    def raise_error(self): raise RuntimeError(self.lib.am_error().decode())
    def __enter__(self): return self
    def __exit__(self, kind, error, traceback): self.close()

    def close(self):
        if self.handle:
            self.lib.am_close(self.handle); self.handle = None
        for stream in self.locks: stream.close()
        self.locks.clear()
        self.temp.cleanup()

    def buffer(self, size, access="read_write", data=None):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        return Buffer(self, size, access, data)

    def platform(self):
        self._require_open()
        return dict(system=platform.system(), kernel=platform.uname()._asdict(),
                    macos=platform.mac_ver()[0], build=subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
                    gpu=self.lib.am_name(self.handle).decode(), driver="Metal", executor=EXECUTOR_NAME,
                    page_bytes=os.sysconf("SC_PAGE_SIZE"), python=platform.python_version())

    def carrier(self):
        self._require_open()
        if self.template is None:
            source = self.directory/"carrier.metal"; source.write_text(carrier_source())
            air, library, archive = (self.directory/name for name in ("carrier.air", "carrier.lib.metallib", "carrier.arc.metallib"))
            subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(source), "-o", str(air)], check=True, capture_output=True)
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(library)], check=True, capture_output=True)
            if self.lib.am_carrier(self.handle, os.fsencode(library), os.fsencode(archive)): self.raise_error()
            self.template = archive.read_bytes()
            code_range(self.template)
        return self.template

    def prepare(self, program):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        if not isinstance(program, G13Program): raise ValueError("validated G13Program required")
        if len(program.bindings) > 8 or program.register_halfs > 80:
            raise ValueError("program exceeds the 8-buffer / 80-halfword Metal carrier profile")
        if program.code_hash not in self.programs:
            template = self.carrier()
            authored, offset, capacity = replace_code(template, program.code)
            archive = self.directory/(program.code_hash+".arc.metallib")
            library = self.directory/(program.code_hash+".lib.metallib")
            shutil.copyfile(self.directory/"carrier.lib.metallib", library)
            archive.write_bytes(authored)
            pipeline = self.lib.am_pipeline(self.handle, os.fsencode(library), os.fsencode(archive))
            if not pipeline: self.raise_error()
            self.programs[program.code_hash] = (pipeline, authored, dict(code_offset=offset, code_capacity=capacity))
        return self.programs[program.code_hash]

    def dispatch(self, program: G13Program, buffers, threads, evidence=None, capture_buffers=True):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        if type(threads) is not int or threads != program.logical_threads or len(buffers) != len(program.bindings):
            raise ValueError("launch extent or binding count disagrees with program")
        if program.bounds_policy == "complete_groups" and threads % program.workgroup_size:
            raise ValueError("unmasked shader requires complete groups")
        for binding in program.bindings:
            buf = buffers[binding.slot]
            if buf.executor is not self or buf.size < binding.min_bytes or buf.address % binding.alignment:
                raise ValueError(f"binding {binding.name} violates ownership, extent or alignment")
            needs = {"read": {"read"}, "write": {"write"}, "read_write": {"read", "write"}}
            if not needs[binding.access] <= needs[buf.access]: raise ValueError("buffer access contract")
        pipeline, archive, placement = self.prepare(program)
        global_x = (threads + program.workgroup_size - 1)//program.workgroup_size*program.workgroup_size
        record = dict(status="PREPARED", executor=EXECUTOR_NAME, program=program.descriptor(),
                      logical_threads=threads, global_threads=[global_x, 1, 1],
                      workgroup_size=[program.workgroup_size, 1, 1], group_count=[global_x//program.workgroup_size, 1, 1],
                      buffers=[dict(binding=b.name, **buffers[b.slot].record()) for b in program.bindings],
                      **placement)
        folder = Path(evidence) if evidence else None
        if folder:
            folder.mkdir(parents=True, exist_ok=False)
            for name, data in (("shader.bin", program.code), ("metal-archive.bin", archive)):
                (folder/name).write_bytes(data)
                record[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            (folder/"shader.txt").write_text(disassemble(program.code))
            if capture_buffers:
                for binding in program.bindings: (folder/f"slot-{binding.slot}-before.bin").write_bytes(buffers[binding.slot].read())
            (folder/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
        bound = list(buffers) + [self.dummy] * (8 - len(buffers))
        native_buffers = (C.c_void_p * 8)(*(buf.handle for buf in bound))
        start = time.monotonic_ns()
        try:
            if self.lib.am_submit(self.handle, pipeline, native_buffers, 8, 64, global_x, program.workgroup_size, self.timeout_ms):
                self.raise_error()
            if not all(buf.canaries_ok() for buf in bound): raise RuntimeError("GPU buffer canary changed")
            record.update(status="COMPLETED", fence_completed=True, canaries_intact=True,
                          host_submit_wait_ns=time.monotonic_ns()-start)
            if folder and capture_buffers:
                for binding in program.bindings: (folder/f"slot-{binding.slot}-after.bin").write_bytes(buffers[binding.slot].read())
        except BaseException as error:
            self.poisoned = True; record.update(status="FAIL", error=str(error)); raise
        finally:
            if folder: (folder/"launch.json").write_text(json.dumps(record, indent=2)+"\n")
        return record
