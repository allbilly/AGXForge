"""Small Python executor over opaque native C objects, never ioctl ctypes structs."""
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import time
from agxforge.g13.abi import G13Program
from agxforge.g13.decode import disassemble

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ("features", "generation", "variant", "revision", "chip_id", "vm_start", "vm_end",
          "kernel_min_bytes", "max_commands", "max_attachments", "timestamp_hz", "dies",
          "clusters", "cores_per_cluster", "max_khz", "drm_major", "drm_minor", "drm_patch",
          "usc_exec_base", "kernel_start", "vm_id", "queue_id")


def native_library(path=None):
    path = Path(path) if path else ROOT / "build/asahi/libagxforge_asahi.so"
    if not path.exists():
        raise RuntimeError("build the native helper first: make asahi-tools")
    lib = C.CDLL(str(path))
    signatures = {
        "af_open": (C.c_void_p, [C.c_char_p, C.c_uint32]),
        "af_close": (C.c_int, [C.c_void_p]),
        "af_error": (C.c_char_p, [C.c_void_p]), "af_node": (C.c_char_p, [C.c_void_p]),
        "af_param": (C.c_uint64, [C.c_void_p, C.c_uint32]),
        "af_alloc": (C.c_void_p, [C.c_void_p, C.c_uint64, C.c_uint32]),
        "af_free": (C.c_int, [C.c_void_p]), "af_map": (C.c_void_p, [C.c_void_p]),
        "af_address": (C.c_uint64, [C.c_void_p]), "af_size": (C.c_uint64, [C.c_void_p]),
        "af_handle": (C.c_uint32, [C.c_void_p]),
        "af_prepare": (C.c_int, [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p] + [C.c_uint32] * 5),
        "af_submit": (C.c_int, [C.c_void_p, C.c_uint32]),
        "af_state": (C.c_void_p, [C.c_void_p, C.c_uint32]),
        "af_state_size": (C.c_uint32, [C.c_void_p, C.c_uint32]),
        "af_state_address": (C.c_uint64, [C.c_void_p, C.c_uint32])
    }
    for name, (restype, argtypes) in signatures.items():
        method = getattr(lib, name); method.restype = restype; method.argtypes = argtypes
    return lib


def _command(argv):
    child = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    return child.stdout.strip() if child.returncode == 0 else None


def source_identity():
    paths = list((ROOT / "agxforge/g13").rglob("*.py"))
    paths += list((ROOT / "agxforge/runtime").glob("asahi*"))
    paths += list((ROOT / "examples/asahi").glob("*.py"))
    paths += [ROOT / "Makefile", ROOT / "requirements-asahi.txt", ROOT / "agxforge/g17/ir.py",
              ROOT / "examples/asahi/00_probe.c", ROOT / "build/asahi/00_probe",
              ROOT / "build/asahi/libagxforge_asahi.so"]
    paths += [Path("/usr/include/drm") / n for n in ("asahi_drm.h", "drm.h")]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file()}


class Buffer:
    def __init__(self, executor, size, access="read_write", data=None):
        executor._require_open()
        if type(size) is not int or size <= 0:
            raise ValueError("positive buffer extent required")
        self.executor, self.size, self.access = executor, size, access
        flags = {"read": 1, "write": 2, "read_write": 3, "code": 5}[access]
        self.handle = executor.lib.af_alloc(executor.handle, size + (128 if access != "code" else 0), flags)
        if not self.handle:
            executor.raise_error()
        executor.buffers.append(self)
        self.raw_address = executor.lib.af_address(self.handle)
        self.offset = 64 if access != "code" else 0
        self.address = self.raw_address + self.offset
        self.allocated_size = executor.lib.af_size(self.handle)
        self.cpu = executor.lib.af_map(self.handle)
        C.memset(self.cpu, 0xa5, self.allocated_size)
        if data is not None:
            self.write(data)

    def write(self, data, offset=0):
        self.executor._require_open()
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("CPU upload exceeds buffer extent")
        C.memmove(self.cpu + self.offset + offset, data, len(data))

    def read(self, size=None, offset=0):
        self.executor._require_open()
        size = self.size - offset if size is None else size
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ValueError("CPU read exceeds buffer extent")
        return C.string_at(self.cpu + self.offset + offset, size)

    def canaries_ok(self):
        self.executor._require_open()
        return (C.string_at(self.cpu, self.offset) == b"\xa5" * self.offset and
                C.string_at(self.cpu + self.offset + self.size, self.allocated_size - self.offset - self.size) ==
                b"\xa5" * (self.allocated_size - self.offset - self.size))

    def record(self):
        self.executor._require_open()
        return dict(gem_handle=self.executor.lib.af_handle(self.handle),
                    gpu_address=self.address, mapping_address=self.raw_address,
                    payload_offset=self.offset, extent=self.size, allocation_bytes=self.allocated_size,
                    access=self.access)


class Executor:
    def __init__(self, device=None, va_slot=0, timeout_ms=5000):
        if not 1 <= timeout_ms <= 60000:
            raise ValueError("timeout must be 1..60000 ms")
        self.lib, self.buffers, self.programs, self.locks = native_library(), [], {}, []
        self.handle, self.poisoned, self.timeout_ms = None, False, timeout_ms
        try:
            for path in (Path.home() / "gpu.lock", Path("/tmp/m1-gpu.lock")):
                stream = path.open("a")
                self.locks.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle = self.lib.af_open(os.fsencode(device) if device else None, va_slot)
            if not self.handle: self.raise_error()
            self.uniforms = self.buffer(512, "read")
        except BaseException:
            self.close(); raise

    def raise_error(self):
        raise RuntimeError(self.lib.af_error(self.handle).decode())

    def _require_open(self):
        if not self.handle:
            raise RuntimeError("executor is closed")
        if self.poisoned:
            raise RuntimeError("executor stopped after a failed dispatch")

    def __enter__(self): return self
    def __exit__(self, kind, error, traceback):
        try: self.close()
        except RuntimeError:
            if error is None: raise

    def close(self):
        result = 0
        if self.handle:
            result = self.lib.af_close(self.handle); self.handle = None
        for stream in self.locks: stream.close()
        self.locks.clear()
        if result: self.raise_error()

    def buffer(self, size, access="read_write", data=None):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        return Buffer(self, size, access, data)

    def platform(self):
        self._require_open()
        params = {name: self.lib.af_param(self.handle, i) for i, name in enumerate(PARAMS)}
        params["core_masks"] = [self.lib.af_param(self.handle, 64 + i) for i in range(params["clusters"])]
        model = Path("/proc/device-tree/model")
        return dict(system=platform.system(), kernel=platform.uname()._asdict(),
                    model=model.read_bytes().rstrip(b"\0").decode() if model.exists() else None,
                    page_bytes=os.sysconf("SC_PAGE_SIZE"), python=platform.python_version(),
                    driver="asahi", render_node=self.lib.af_node(self.handle).decode(), params=params,
                    packages=_command(["rpm", "-q", "mesa-dri-drivers", "mesa-vulkan-drivers", "kernel-headers", "libdrm"]),
                    compiler=_command(["cc", "--version"]), repository_revision=_command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]))

    def dispatch(self, program: G13Program, buffers, threads, evidence=None, capture_buffers=True):
        self._require_open()
        if self.poisoned: raise RuntimeError("executor stopped after a failed dispatch")
        if type(threads) is not int or threads <= 0 or len(buffers) != len(program.bindings):
            raise ValueError("invalid launch dimensions or binding count")
        if threads != program.logical_threads:
            raise ValueError("launch extent disagrees with compiled bounds")
        if program.bounds_policy == "complete_groups" and threads % program.workgroup_size:
            raise ValueError("unmasked shader requires complete groups")
        for binding in program.bindings:
            buf = buffers[binding.slot]
            if buf.executor is not self or buf.size < binding.min_bytes or buf.address % binding.alignment:
                raise ValueError(f"binding {binding.name} violates ownership, extent or alignment")
            if binding.access in ("write", "read_write") and buf.access not in ("write", "read_write"):
                raise ValueError(f"binding {binding.name} requires GPU write permission")
            if binding.access in ("read", "read_write") and buf.access not in ("read", "read_write"):
                raise ValueError(f"binding {binding.name} requires GPU read permission")
        shader = self.programs.get(program.code_hash)
        if shader is None:
            shader = self.buffer(len(program.code), "code", program.code)
            self.programs[program.code_hash] = shader
        pointers = struct.pack("<" + "Q" * len(buffers), *(b.address for b in buffers))
        self.uniforms.write(pointers)
        global_x = (threads + program.workgroup_size - 1) // program.workgroup_size * program.workgroup_size
        if self.lib.af_prepare(self.handle, shader.handle, program.entry_offset,
                self.uniforms.handle, self.uniforms.offset, program.uniform_halfs, program.register_halfs, global_x, program.workgroup_size):
            self.raise_error()
        record = dict(status="PREPARED", program=program.descriptor(), logical_threads=threads,
                      global_threads=[global_x, 1, 1], workgroup_size=[program.workgroup_size, 1, 1],
                      group_count=[global_x // program.workgroup_size, 1, 1],
                      buffers=[dict(binding=binding.name, **buffers[binding.slot].record()) for binding in program.bindings],
                      shader=shader.record(), uniforms_gpu_address=self.uniforms.address,
                      usc_exec_base=self.lib.af_param(self.handle, 18))
        folder = Path(evidence) if evidence else None
        if folder:
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "shader.bin").write_bytes(program.code)
            (folder / "shader.txt").write_text(disassemble(program.code))
            for i, name in enumerate(("usc.bin", "cdm.bin", "drm-command.bin")):
                data = C.string_at(self.lib.af_state(self.handle, i), self.lib.af_state_size(self.handle, i))
                (folder / name).write_bytes(data)
                record[name] = dict(gpu_address=self.lib.af_state_address(self.handle, i), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            (folder / "uniforms.bin").write_bytes(pointers)
            if capture_buffers:
                for binding in program.bindings:
                    (folder / f"slot-{binding.slot}-before.bin").write_bytes(buffers[binding.slot].read())
            (folder / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        start = time.monotonic_ns()
        try:
            if self.lib.af_submit(self.handle, self.timeout_ms): self.raise_error()
            record.update(status="COMPLETED", fence_completed=True, host_submit_wait_ns=time.monotonic_ns() - start)
            if folder and capture_buffers:
                for binding in program.bindings:
                    (folder / f"slot-{binding.slot}-after.bin").write_bytes(buffers[binding.slot].read())
            # Uniform/code internal BOs aren't tensor buffers. Check caller allocations.
            if not all(b.canaries_ok() for b in buffers):
                raise RuntimeError("GPU buffer canary changed")
            record["canaries_intact"] = True
        except BaseException as error:
            self.poisoned = True
            record.update(status="FAIL", error=str(error))
            raise
        finally:
            if folder: (folder / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
