"""Hardware-free macOS boundary checks; GPU receipts are recorded separately."""
from dataclasses import replace
import ctypes as C
import json
from pathlib import Path
import platform
import struct
import unittest
from unittest.mock import Mock, patch

from agxforge.g13 import kernels, smoke
from agxforge.g13.metal import code_range, replace_code
from agxforge.runtime import asahi, macos, iogpu
from tools.macos_capture import read_capture
import test_review_regressions as regressions


def shader_object():
    """Minimal Mach-O with one native text section and _agc.main symbol."""
    obj = bytearray(260)
    struct.pack_into("<8I", obj, 0, 0xfeedfacf, 0x1000013, 0, 1, 2, 176, 0, 0)
    struct.pack_into("<II", obj, 32, 0x19, 152)
    struct.pack_into("<I", obj, 96, 1)
    obj[104:110] = b"__text"; obj[120:126] = b"__TEXT"
    struct.pack_into("<QQI", obj, 136, 0, 16, 208)
    struct.pack_into("<6I", obj, 184, 2, 24, 224, 1, 240, 20)
    struct.pack_into("<IBBHQ", obj, 224, 1, 0x0e, 1, 0, 0)
    obj[241:251] = b"_agc.main\0"
    return bytes(obj)


class ArchiveBoundary(unittest.TestCase):
    def test_exact_replacement_preserves_container_and_metadata(self):
        archive = b"outer" + shader_object()
        code = smoke.program("store", 32).code[:8]
        authored, offset, capacity = replace_code(archive, code)
        self.assertEqual((offset, capacity), (213, 16))
        self.assertEqual(authored[:offset], archive[:offset])
        self.assertEqual(authored[offset:offset+len(code)], code)
        self.assertEqual(authored[offset+len(code):], archive[offset+len(code):])

    def test_missing_ambiguous_truncated_or_oversized_archives_refused(self):
        obj = shader_object()
        for archive in (b"", obj[:200], obj+obj):
            with self.subTest(archive=len(archive)), self.assertRaises(ValueError): code_range(archive)
        with self.assertRaisesRegex(ValueError, "capacity"): replace_code(obj, bytes(17))

    def test_bad_command_section_and_symbol_bounds_refused(self):
        for offset, value in ((36, 0), (152, 9999), (232, 1000), (192, 9999), (228, 0x20e)):
            obj = bytearray(shader_object()); struct.pack_into("<I", obj, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError): code_range(obj)


class LifetimeAndProfile(unittest.TestCase):
    def test_closed_buffers_and_executors_never_reach_native_calls(self):
        for module in (macos, iogpu):
            gpu = module.Executor.__new__(module.Executor)
            gpu.handle, gpu.poisoned, gpu.lib = None, False, Mock()
            buffer = module.Buffer.__new__(module.Buffer); buffer.executor = gpu
            for call in (lambda: gpu.buffer(4), gpu.platform, lambda: gpu.prepare(smoke.program("copy", 32)),
                         lambda: gpu.dispatch(smoke.program("copy", 32), [], 32),
                         buffer.read, lambda: buffer.write(b"x"), buffer.canaries_ok, buffer.record):
                with self.subTest(module=module.__name__), self.assertRaisesRegex(RuntimeError, "closed"): call()
            self.assertEqual(gpu.lib.mock_calls, [])

    def test_failed_dispatch_prevents_followup_allocation_prepare_and_submit(self):
        for module in (macos, iogpu):
            gpu = module.Executor.__new__(module.Executor)
            gpu.handle, gpu.poisoned, gpu.lib = 1, True, Mock()
            for call in (lambda: gpu.buffer(4), lambda: gpu.prepare(smoke.program("copy", 32)),
                         lambda: gpu.dispatch(smoke.program("copy", 32), [], 32)):
                with self.subTest(module=module.__name__), self.assertRaisesRegex(RuntimeError, "stopped"): call()
            self.assertEqual(gpu.lib.mock_calls, [])

    def test_poisoned_buffers_never_touch_shared_memory(self):
        for module in (asahi, macos, iogpu):
            gpu = module.Executor.__new__(module.Executor)
            gpu.handle, gpu.poisoned = 1, True
            buffer = module.Buffer.__new__(module.Buffer); buffer.executor = gpu
            with patch.object(macos.C, "memmove") as write, patch.object(macos.C, "string_at") as read:
                for call in (buffer.read, lambda: buffer.write(b"x"), buffer.canaries_ok, buffer.record):
                    with self.subTest(module=module.__name__, call=call), self.assertRaisesRegex(RuntimeError, "stopped"):
                        call()
                write.assert_not_called(); read.assert_not_called()

    def test_healthy_buffers_still_read_write_and_check_guards(self):
        for module in (asahi, macos, iogpu):
            gpu = module.Executor.__new__(module.Executor)
            gpu.handle, gpu.poisoned = 1, False
            storage = C.create_string_buffer(b"\xa5"*132)
            buffer = module.Buffer.__new__(module.Buffer)
            buffer.executor, buffer.cpu, buffer.offset, buffer.size = gpu, C.addressof(storage), 64, 4
            buffer.allocated_size = 132
            buffer.write(b"data")
            self.assertEqual(buffer.read(), b"data")
            self.assertTrue(buffer.canaries_ok())
            buffer.write(b"X", offset=1)
            self.assertEqual(buffer.read(2, offset=1), b"Xt")

    def test_native_capacity_refused_before_allocation_or_submit(self):
        gpu = iogpu.Executor.__new__(iogpu.Executor)
        gpu.handle, gpu.poisoned, gpu.lib = 1, False, Mock()
        for program in (replace(smoke.program("copy", 32), register_halfs=82),
                        replace(smoke.program("copy", 32), workgroup_size=64)):
            with self.subTest(program=program.name), self.assertRaisesRegex(ValueError, "IOGPU profile"):
                gpu.dispatch(program, [], program.logical_threads)
        self.assertEqual(gpu.lib.mock_calls, [])

    def test_helper_template_and_capture_sources_are_identified(self):
        identity = macos.source_identity()
        for path in ("agxforge/runtime/iogpu.py", "agxforge/runtime/iogpu.c",
                     "agxforge/runtime/iogpu-26A434.json", "tools/macos_capture.py"):
            self.assertIn(path, identity)


class NativeEnvelope(unittest.TestCase):
    def test_launches_get_fresh_command_and_encoder_ids_even_after_range_exhaustion(self):
        gpu = iogpu.Executor.__new__(iogpu.Executor)
        gpu.handle, gpu.trace_cursor, gpu.trace_end = 1,0,0
        gpu.pages = [(1,bytes(16384)),(2,bytes(16384))]
        ranges = iter(((123,125),(201,205)))
        def allocate(handle,selector,scalars,count,request,size,output,length):
            self.assertEqual((selector,length),(6,16))
            C.memmove(output,struct.pack("<QQ",*next(ranges)),16)
            return 0
        gpu.lib = Mock(); gpu.lib.an_call.side_effect = allocate
        for expected in ((123,124),(201,202),(203,204)):
            ids,pages = gpu.launch_pages()
            self.assertEqual(ids,expected)
            for offset in (0,0x18):self.assertEqual(struct.unpack_from("<Q",pages[0],offset)[0],ids[0])
            self.assertEqual(struct.unpack_from("<Q",pages[0],0x28)[0],ids[1])
            self.assertEqual(struct.unpack_from("<I",pages[1],0x23c)[0],ids[1])
        self.assertEqual(gpu.lib.an_call.call_count,2)
        self.assertEqual(gpu.pages,[(1,bytes(16384)),(2,bytes(16384))])

    def test_tensor_arena_has_one_residency_entry_without_changing_captured_views(self):
        import base64,zlib
        template = json.loads(iogpu.TEMPLATE.read_text())
        address = next(struct.unpack_from("<Q",bytes.fromhex(e["output"]))[0] for e in template["events"]
                       if e["kind"] == "call" and e["selector"] == 14)
        event = next(e for e in template["events"] if e["kind"] == "memory" and e["address"] == address)
        original = zlib.decompress(base64.b64decode(event["data"]))
        page = iogpu.append_arena_resource(original,27)
        self.assertEqual(page[0x48:0x108],original[0x48:0x108])
        self.assertEqual(struct.unpack_from("<II",page,0x40),(17,4))
        self.assertEqual(struct.unpack_from("<I",page,0x108)[0],27)
        self.assertEqual(struct.unpack_from("<H",page,0x146)[0],1)
        self.assertEqual(page[0x148:],original[0x148:])
        with self.assertRaisesRegex(ValueError,"residency list profile changed"):
            iogpu.append_arena_resource(page,27)

    def test_compiled_entries_preserve_driver_helpers_and_never_overwrite_other_code(self):
        gpu = iogpu.Executor.__new__(iogpu.Executor)
        storage = C.create_string_buffer(bytes([0xa5])*iogpu.CODE_BYTES)
        gpu.handle, gpu.poisoned = 1, False
        gpu.code_cpu, gpu.code_cursor, gpu.programs = C.addressof(storage), iogpu.CODE_START, {}
        first = kernels.qmv4(33, 16).compile()
        second = smoke.program("copy",32)
        a, b = gpu.prepare(first), gpu.prepare(second)
        self.assertEqual(storage.raw[:iogpu.CODE_START],bytes([0xa5])*iogpu.CODE_START)
        self.assertEqual(storage.raw[a:a+len(first.code)],first.code)
        self.assertGreaterEqual(b,a+len(first.code))
        self.assertEqual(gpu.prepare(first),a)
        self.assertEqual(len(gpu.programs),2)
        gpu.code_cursor = iogpu.CODE_BYTES
        with self.assertRaisesRegex(ValueError,"code heap exhausted"):
            gpu.prepare(smoke.program("store",32))

    def test_missing_corrupt_or_wrong_profile_support_refused(self):
        import tempfile
        import hashlib
        with tempfile.TemporaryDirectory() as directory, patch.object(iogpu,"SUPPORT",Path(directory)/"support.bin"):
            with self.assertRaisesRegex(RuntimeError,"macos-support"): iogpu.load_support()
            data = bytes([0xa5])*iogpu.CODE_START
            receipt = dict(schema_version=1,build="26A434",gpu="Apple M1",code_start=iogpu.CODE_START,
                           bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                           template_sha256=hashlib.sha256(iogpu.TEMPLATE.read_bytes()).hexdigest())
            iogpu.SUPPORT.write_bytes(data)
            manifest = iogpu.SUPPORT.with_suffix(".json")
            manifest.write_text(json.dumps(receipt))
            self.assertEqual(iogpu.load_support(),(data,receipt))
            for key,value in (("build","wrong"),("code_start",64),("sha256","0"),("template_sha256","0")):
                manifest.write_text(json.dumps(dict(receipt,**{key:value})))
                with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,"incompatible"):
                    iogpu.load_support()
            manifest.write_text(json.dumps(receipt))
            iogpu.SUPPORT.write_bytes(bytes(iogpu.CODE_START))
            with self.assertRaisesRegex(RuntimeError,"incompatible"): iogpu.load_support()

    def test_capture_parser_refuses_truncated_or_unknown_records(self):
        header = struct.pack("<4I", 0x43584741, 2, 0, 0)
        for raw in (header+b"\xff", header+b"\x02", bytes(16)):
            with self.subTest(raw=raw), self.assertRaises((ValueError, struct.error)): read_capture(raw)

    def test_relocations_use_original_bytes_and_preserve_nonpointers(self):
        reloc = iogpu.Relocations(); reloc.add(0x1000, 0x9000, 64)
        raw = struct.pack("<3Q", 0x1010, 0x2000, 0x1000)
        self.assertEqual(struct.unpack("<3Q", reloc.patch(raw)), (0x9010, 0x2000, 0x9000))
        reloc = iogpu.Relocations(); reloc.add(0x100000000, 0x200000001)
        reloc.add(0x200000000, 0xdeadbeef)
        raw = struct.pack("<QI", 0x100000000, 2)
        self.assertEqual(reloc.patch(raw), struct.pack("<QI", 0x200000001, 2))


class NativeEvidence(regressions.ScalarEvidence):
    def setUp(self):
        super().setUp()
        import hashlib
        for folder in self.folders:
            path = folder/"launch.json"; launch = json.loads(path.read_text())
            launch.update(executor="direct IOGPU G13", notification_tokens=[100, 200],
                          completion=(struct.pack("<5Q",100,1,2,0,0)+struct.pack("<5Q",200,2,3,0,0)).hex(),
                          buffers=[dict(binding="output", gpu_address=64)])
            launch["program"]["bindings"] = [dict(name="output", slot=0)]
            (folder/"drm-command.bin").rename(folder/"iogpu-submit.bin")
            launch["iogpu-submit.bin"] = launch.pop("drm-command.bin")
            for name, data in (("iogpu-pages.bin", b"native-pages"), ("uniforms.bin", struct.pack("<Q",64))):
                (folder/name).write_bytes(data); launch[name] = dict(sha256=hashlib.sha256(data).hexdigest())
            path.write_text(json.dumps(launch))

    def test_error_or_missing_notification_cannot_pass(self):
        from tools.asahi_evidence import audit
        path = self.folders[0]/"launch.json"; launch = json.loads(path.read_text())
        launch["completion"] = (struct.pack("<5Q",100,1,2,0,0)+struct.pack("<5Q",200,2,3,0xb,0)).hex()
        path.write_text(json.dumps(launch))
        with self.assertRaisesRegex(ValueError, "failed native completion"): audit(self.root)
        launch["completion"] = ""; path.write_text(json.dumps(launch))
        with self.assertRaisesRegex(ValueError, "missing native completion"): audit(self.root)

    def test_relocated_uniforms_must_agree_with_binding(self):
        from tools.asahi_evidence import audit
        path = self.folders[0]/"launch.json"; launch = json.loads(path.read_text())
        launch["buffers"][0]["gpu_address"] = 128; path.write_text(json.dumps(launch))
        with self.assertRaisesRegex(ValueError, "uniform pointers"): audit(self.root)


@unittest.skipUnless(platform.system() == "Darwin" and
                     (macos.ROOT/"build/macos/libagxforge_iogpu.dylib").is_file(),
                     "build the macOS helper to test the CPU notification parser")
class NativeNotificationParser(unittest.TestCase):
    def setUp(self):
        self.lib = C.CDLL(str(macos.ROOT/"build/macos/libagxforge_iogpu.dylib"))
        self.lib.an_wait.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.c_uint64, C.c_void_p, C.c_uint32]
        self.lib.an_wait.restype = C.c_int
        self.lib.an_error.restype = C.c_char_p
        # CPU-only context/ring. No user-client open, submission or GPU calls.
        self.context = (C.c_uint32*2)(0, 0)
        self.ring, self.receipt = C.create_string_buffer(16384), C.create_string_buffer(80)

    def wait(self, messages, *, head=0, tail=None):
        entries = b"".join(struct.pack("<I",40)+struct.pack("<5Q",*m) for m in messages)
        C.memmove(self.ring, struct.pack("<3I",4096,head,len(entries) if tail is None else tail)+entries, 12+len(entries))
        return self.lib.an_wait(self.context,self.ring,100,200,self.receipt,1)

    def test_two_success_notifications_complete(self):
        messages = [(100,1,2,0,0),(200,2,3,0,0)]
        self.assertEqual(self.wait(messages),0)
        self.assertEqual(self.receipt.raw,b"".join(struct.pack("<5Q",*m) for m in messages))

    def test_fault_duplicate_and_foreign_notifications_stop(self):
        for message in ((200,2,3,0xb,0),(100,2,3,0,0),(300,2,3,0,0)):
            self.context[1] = 0
            with self.subTest(message=message):
                self.assertEqual(self.wait([(100,1,2,0,0),message]),-1)
                self.assertEqual(self.context[1],1)

    def test_invalid_ring_indices_stop_without_dequeue(self):
        self.assertEqual(self.wait([],head=4097),-1)
        self.assertIn(b"outside mapped page",self.lib.an_error())

    def test_exact_end_head_wraps_and_exact_end_tail_completes(self):
        messages = [(100,1,2,0,0),(200,2,3,0,0)]
        self.assertEqual(self.wait(messages,head=4096),0)
        self.assertEqual(struct.unpack_from("<I",self.ring.raw,4)[0],88)
        entries = b"".join(struct.pack("<I",40)+struct.pack("<5Q",*m) for m in messages)
        C.memmove(self.ring,struct.pack("<3I",4096,4008,4096),12)
        C.memmove(C.addressof(self.ring)+12+4008,entries,len(entries))
        self.assertEqual(self.lib.an_wait(self.context,self.ring,100,200,self.receipt,1),0)
        self.assertEqual(struct.unpack_from("<I",self.ring.raw,4)[0],4096)

    def test_empty_ring_has_bounded_timeout(self):
        self.assertEqual(self.wait([]),-1)
        self.assertEqual(self.context[1],1)
        self.assertIn(b"timeout",self.lib.an_error())


if __name__ == "__main__": unittest.main()
