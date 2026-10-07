"""Regression checks for loop overflow, closed ownership and incomplete evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from agxforge.g13 import cc, ir, smoke
from agxforge.runtime.asahi import Buffer, Executor
from tools.asahi_evidence import audit


def counted_loop(width, bound, *, dynamic=False, seed_width=None, increment_width=None):
    out = ir.Buffer("out", 0, ir.I32)
    fn = ir.Function("counted_loop", [out])
    b = ir.Builder(fn, fn.block("entry"))
    zero = b.const(0, type=seed_width or width)
    runtime_bound = b.const(bound) if dynamic else None
    body, after = fn.block("loop"), fn.block("done")
    b.br(body); b.at(body)
    counter = b.phi(zero, type=width)
    increment = b.add(counter, ir.Imm(1), type=increment_width or width)
    b.phi_latch(counter, increment)
    pred = b.cmp(increment, runtime_bound, cap=bound) if dynamic else b.cmp(increment, bound)
    b.br_cond(pred, body, after); b.at(after)
    b.store_at(out, ir.Imm(0), b.const(1)); b.ret()
    return fn


class LoopSafety(unittest.TestCase):
    def compile(self, fn):
        return cc.compile_function(fn, threads=1, extents={"out": 4})

    def test_reject_16_bit_counter_overflow(self):
        for bound in (65536, 10000000):
            for dynamic in (False, True):
                with self.subTest(bound=bound, dynamic=dynamic), self.assertRaisesRegex(cc.Unsupported, "overflow"):
                    self.compile(counted_loop(ir.I16, bound, dynamic=dynamic))

    def test_accept_reachable_counter_caps(self):
        for width, bound in ((ir.I16, 65535), (ir.I32, 65536), (ir.I32, 10000000)):
            for dynamic in (False, True):
                with self.subTest(width=width, bound=bound, dynamic=dynamic):
                    self.assertEqual(self.compile(counted_loop(width, bound, dynamic=dynamic)).target, "g13g")

    def test_reject_narrowing_counter_transfers(self):
        for kwargs in (dict(seed_width=ir.I32), dict(increment_width=ir.I32)):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(cc.Unsupported, "preserve width"):
                self.compile(counted_loop(ir.I16, 7, **kwargs))
        with self.assertRaisesRegex(cc.Unsupported, "preserve width"):
            self.compile(counted_loop(ir.I32, 65536, increment_width=ir.I16))


class ClosedExecutor(unittest.TestCase):
    def setUp(self):
        self.gpu = Executor.__new__(Executor)
        self.gpu.handle, self.gpu.poisoned, self.gpu.locks = None, False, []
        self.gpu.lib = Mock()

    def test_closed_executor_rejects_native_operations(self):
        program = smoke.program("copy", 32)
        for call in (lambda: self.gpu.buffer(4), lambda: Buffer(self.gpu, 4),
                     lambda: self.gpu.platform(), lambda: self.gpu.dispatch(program, [], 32)):
            with self.subTest(call=call), self.assertRaisesRegex(RuntimeError, "closed"):
                call()
        self.assertEqual(self.gpu.lib.mock_calls, [])

    def test_closed_buffer_rejects_access_and_native_record(self):
        buffer = Buffer.__new__(Buffer); buffer.executor = self.gpu
        for call in (lambda: buffer.write(b"x"), buffer.read, buffer.canaries_ok, buffer.record):
            with self.subTest(call=call), self.assertRaisesRegex(RuntimeError, "closed"):
                call()
        self.assertEqual(self.gpu.lib.mock_calls, [])

    def test_close_is_idempotent_and_prevents_reallocation(self):
        self.gpu.handle = 1; self.gpu.lib.af_close.return_value = 0
        self.gpu.close(); self.gpu.close()
        with self.assertRaisesRegex(RuntimeError, "closed"): self.gpu.buffer(4)
        self.gpu.lib.af_close.assert_called_once_with(1)
        self.gpu.lib.af_alloc.assert_not_called()


class ScalarEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        checks = []
        self.folders = []
        for index in range(2):
            folder = self.root / str(index); folder.mkdir(); self.folders.append(folder)
            row = dict(name=f"case-{index}", status="PASS")
            checks.append(row)
            (folder/"check.json").write_text(json.dumps(row))
            launch = dict(status="COMPLETED", fence_completed=True, canaries_intact=True,
                          logical_threads=1, program=dict(logical_threads=1))
            for name in ("shader.bin", "usc.bin", "cdm.bin", "drm-command.bin"):
                data = b"fixture-" + name.encode()
                (folder/name).write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                if name == "shader.bin": launch["program"]["code_sha256"] = digest
                else: launch[name] = dict(sha256=digest)
            (folder/"launch.json").write_text(json.dumps(launch))
            for name in ("slot-0-after.bin", "expected.bin"):
                (folder/name).write_bytes(bytes([index, 0, 0, 0]))
        (self.root/"summary.json").write_text(json.dumps(dict(status="PASS", checks=checks)))

    def test_complete_bundle_passes(self):
        self.assertEqual(audit(self.root)["launches"], 2)

    def test_missing_all_or_one_result_checks_fails(self):
        (self.folders[0]/"check.json").unlink()
        with self.assertRaisesRegex(ValueError, "exactly one result check"): audit(self.root)
        (self.folders[1]/"check.json").unlink()
        with self.assertRaisesRegex(ValueError, "exactly one result check"): audit(self.root)

    def test_result_check_in_wrong_directory_fails(self):
        (self.folders[0]/"check.json").rename(self.root/"check.json")
        with self.assertRaisesRegex(ValueError, "exactly one result check"): audit(self.root)

    def test_result_check_must_match_summary(self):
        (self.folders[0]/"check.json").write_text(json.dumps(dict(name="other-case", status="PASS")))
        with self.assertRaisesRegex(ValueError, "disagree with receipt"): audit(self.root)

    def test_missing_output_artifacts_fail(self):
        for name in ("slot-0-after.bin", "expected.bin"):
            path = self.folders[0]/name; data = path.read_bytes(); path.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing scalar output"):
                audit(self.root)
            path.write_bytes(data)

    def test_empty_output_cannot_pass(self):
        for name in ("slot-0-after.bin", "expected.bin"):
            (self.folders[0]/name).write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty scalar output"): audit(self.root)

    def test_wrong_output_cannot_pass(self):
        (self.folders[0]/"slot-0-after.bin").write_bytes(b"bad!")
        with self.assertRaisesRegex(ValueError, "scalar output bits"): audit(self.root)


if __name__ == "__main__": unittest.main()
