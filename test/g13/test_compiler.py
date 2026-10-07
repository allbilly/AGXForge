"""Hardware-free contract and encoding checks; native results live separately."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from agxforge.g13 import cc, ir, kernels, smoke
from agxforge.g13.encode import instruction, assemble
from agxforge.g13.decode import decode, validate
from agxforge.runtime.asahi import Executor


class Encoding(unittest.TestCase):
    def test_independent_mesa_golden_bytes(self):
        # Existing Mesa 25.3.6 scalar shaders, independent of this encoder.
        forms = {
            "get_sr r0, sr80": "72011004",
            "device_load 0, i32, x, r1, u4_u5, r0, unsigned, lsl 0": "0509080e00c01200",
            "device_load 1, i32, x, r2, u2_u3, r0, unsigned, lsl 0": "0511044e00c01200",
            "fadd32 r1, r1.discard, r2.discard": "2a85c2422c00",
            "fmul32 r1, r1.discard, r2.discard": "1a85c2422c00",
            "device_store 0, i32, x, r1, u0_u1, r0, unsigned, lsl 0, 0": "4509000e00c01200",
            "wait 0": "3800", "stop": "8800", "trap": "0800",
        }
        for assembly, golden in forms.items():
            with self.subTest(assembly=assembly):
                self.assertEqual(instruction(assembly), bytes.fromhex(golden))

    def test_reject_silent_modifier_or_immediate_loss(self):
        for text in ("iadd r3, r4, r5.neg", "wait 256", "mov_imm r3, 4294967296, 0", "mov r128, r1"):
            with self.subTest(text=text), self.assertRaises(ValueError): instruction(text)

    def test_branch_boundaries_and_truncation(self):
        valid = assemble(["jmp_exec_none @end", "mov_imm r1, 1, 0", "end:", "stop"])
        validate(valid, 4, 4)
        with self.assertRaises(ValueError): validate(assemble(["jmp_exec_none pc+1", "stop"]), 4, 4)
        with self.assertRaises(ValueError): decode(instruction("mov_imm r1, 1, 0")[:-1])
        with self.assertRaises(ValueError): validate(instruction("trap")+instruction("stop"), 4, 4)

    def test_register_views_and_uniform_extent(self):
        for text, general, uniform in (("mov r2h, r0l", 5, 4), ("mov r3, r0", 6, 4),
                                      ("mov r1, u2", 4, 4)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate(instruction(text)+instruction("stop"), general, uniform)


class Compiler(unittest.TestCase):
    def test_bits_select_float_arithmetic_without_apple_tools(self):
        program = kernels.vector("mul", 33).compile()
        self.assertIn("fmul32", [i.mnemonic for i in decode(program.code)])
        self.assertEqual(program.logical_threads, 33)
        self.assertEqual(program.bounds_policy, "masked")

    def test_integer_subtraction_selects_negation_bit(self):
        spec, b, t, p = kernels.setup("isub", {"out": 132, "a": 132, "b": 132}, 33)
        value = b.sub(b.load(p["a"], t), b.load(p["b"], t))
        b.store_at(p["out"], t, value); b.ret()
        form, = [i for i in decode(spec.compile().code) if i.mnemonic == "iadd"]
        self.assertEqual(form.fields["N"], 1)

    def test_every_selected_kernel_checks_capabilities(self):
        with self.assertRaisesRegex(cc.Unsupported, "bounded_loops"):
            kernels.gemv(3, 7).compile({"scalar", "integer", "bf16_storage"})

    def test_unsupported_ir_and_attributes_fail_before_execution(self):
        for mutate in (lambda s: setattr(s.function.blocks[0].ops[1], "kind", "tensor_gemm"),
                       lambda s: s.function.blocks[0].ops[1].attrs.update(saturate=True)):
            spec = kernels.vector("add", 33); mutate(spec)
            with self.assertRaises(cc.Unsupported): spec.compile()

    def test_spills_and_shared_memory_are_rejected(self):
        spec, b, t, p = kernels.setup("pressure", {"out": 4}, 1)
        for i in range(125): b.const(i)
        b.ret()
        with self.assertRaisesRegex(cc.Unsupported, "spilling"): spec.compile()
        spec = kernels.vector("copy", 32); spec.function.declare_threadgroup(32)
        with self.assertRaisesRegex(cc.Unsupported, "threadgroup"): spec.compile()

    def test_ssa_and_terminators_are_checked(self):
        spec = kernels.vector("add", 32)
        ops = spec.function.blocks[0].ops
        ops[1], ops[3] = ops[3], ops[1]
        with self.assertRaisesRegex(cc.Unsupported, "before definition"): spec.compile()
        spec = kernels.vector("add", 32)
        spec.function.blocks[0].ops.insert(0, ir.Op("ret"))
        with self.assertRaisesRegex(cc.Unsupported, "terminator"): spec.compile()

    def test_model_kernel_builders_have_linux_compilation(self):
        specs = [kernels.qmv4(33, 16), kernels.rope(14, 64), kernels.scores(14, 2, 64, 32),
                 kernels.attend(14, 2, 64, 32), kernels.argmax(129)]
        for spec in specs:
            with self.subTest(name=spec.function.name): self.assertEqual(spec.compile().target, "g13g")


class Contract(unittest.TestCase):
    def test_shader_declarations_are_checked(self):
        p = smoke.program("copy", 32)
        for attrs in (dict(register_halfs=2), dict(uniform_halfs=4), dict(builtins=()),
                      dict(scratch_bytes=16), dict(threadgroup_bytes=32), dict(target="g17"),
                      dict(code=bytearray(p.code)), dict(logical_threads=0)):
            with self.subTest(attrs=attrs), self.assertRaises(ValueError): replace(p, **attrs)

    def test_bad_bindings_and_launch_fail_before_native_calls(self):
        gpu = Executor.__new__(Executor); gpu.poisoned = False; gpu.handle = 1
        p = smoke.program("copy", 32)
        with self.assertRaises(ValueError): gpu.dispatch(p, [], 32)
        with self.assertRaises(ValueError): gpu.dispatch(p, [None]*3, 64)
        with patch.object(Executor, "buffer", side_effect=AssertionError("allocation reached")):
            class FakeBuffer:
                executor=gpu; size=4; address=64; access="read_write"
            with self.assertRaisesRegex(ValueError, "extent"): gpu.dispatch(p, [FakeBuffer()]*3, 32)


if __name__ == "__main__": unittest.main()
