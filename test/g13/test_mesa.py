"""Hardware-free boundaries for the optional Mesa userspace compiler adapter."""
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import shutil
import tempfile
from agxforge.g13 import cc, kernels, gpt2_kernels, mesa


# Captured from the real Mesa 26.2.4 compiler and executed on base M1.
ADD = bytes.fromhex(
    "72051004620000005228421202000511240e00c012000519284e00c01200"
    "380038012a89446224004511200e00c012003800520e000000008800"
    "08000800080008000800080008000800")


def metadata():
    return dict(compiler="Mesa 26.2.4 AGX", register_halfs=8, uniform_halfs=12,
                binary_size=len(ADD), main_size=len(ADD), main_offset=0,
                has_preamble=False, scratch_bytes=0, threadgroup_bytes=0,
                rodata_halfs=0, workgroup_size=[32, 1, 1])


class MesaBoundary(unittest.TestCase):
    def test_real_mesa_bytes_keep_compiler_identity_and_masked_tail(self):
        admitted = kernels.vector("add", 33).compile()
        program = mesa.from_binary(admitted, ADD, metadata())
        self.assertEqual(program.code_hash, "8ac9b4cbf5c29652dbe94e8ca316b7fd8503d3e4433fb0bd65c8261d33d06537")
        self.assertEqual(program.bindings, admitted.bindings)
        self.assertEqual(program.origin, "Mesa 26.2.4 NIR -> AGX")
        self.assertEqual((program.register_halfs, program.builtins, program.bounds_policy), (8, (80,), "masked"))

    def test_launch_requirements_cannot_be_silently_discarded(self):
        admitted = kernels.vector("add", 33).compile()
        for key, value in (("has_preamble", True), ("scratch_bytes", 16),
                           ("threadgroup_bytes", 16), ("rodata_halfs", 4), ("main_offset", 64),
                           ("main_size", len(ADD)-2), ("binary_size", len(ADD)+2),
                           ("uniform_halfs", 16), ("register_halfs", 81),
                           ("register_halfs", True), ("workgroup_size", [64, 1, 1]),
                           ("compiler", "Apple Metal")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                mesa.from_binary(admitted, ADD, dict(metadata(), **{key: value}))

    def test_binary_validation_still_rejects_truncation_and_register_understatement(self):
        admitted = kernels.vector("add", 33).compile()
        with self.assertRaises(ValueError):
            mesa.from_binary(admitted, ADD[:-1], dict(metadata(), main_size=len(ADD)-1, binary_size=len(ADD)-1))
        with self.assertRaisesRegex(ValueError, "footprint"):
            mesa.from_binary(admitted, ADD, dict(metadata(), register_halfs=6))

    def test_loop_and_fp32_constant_bits_survive_lowering(self):
        protocol, _ = mesa.lower(gpt2_kernels.conv1d(65, 31))
        self.assertEqual(protocol.count("loop_begin "), 1)
        self.assertEqual(protocol.count("loop_end "), 1)
        self.assertEqual(protocol.count("phi_seed "), 2)
        self.assertEqual(protocol.count("phi_store "), 2)
        self.assertIn("fma ", protocol)
        gelu, _ = mesa.lower(gpt2_kernels.gelu(33))
        self.assertIn("1027024659", gelu)  # IEEE binary32 0.044715, not an integer conversion.

    def test_refuses_unimplemented_operations_and_non_x_builtins(self):
        spec = kernels.vector("add", 33)
        spec.function.blocks[0].ops[1].kind = "tensor_gemm"
        with self.assertRaisesRegex(cc.Unsupported, "tensor_gemm"): mesa.lower(spec)
        spec = kernels.vector("add", 33)
        spec.function.blocks[0].ops[0].attrs["axis"] = "y"
        with self.assertRaisesRegex(cc.Unsupported, "builtin"): mesa.lower(spec)

    def test_half_loads_widen_as_bits_without_authored_code_generation(self):
        for dtype in ("bfloat", "f16", "f32"):
            with patch.object(cc, "compile_function", side_effect=AssertionError("authored codegen reached")):
                protocol, contract = mesa.lower(kernels.gemv(3, 7, dtype))
            self.assertEqual(contract.bindings[2].min_bytes, 3*7*(4 if dtype == "f32" else 2))
            if dtype == "bfloat": self.assertIn("shl ", protocol)
            if dtype == "f16": self.assertIn("f16_to_f32 ", protocol)

    def test_displaced_loads_have_a_valid_address_addition(self):
        from agxforge.g13 import ir
        for element in (ir.I16, ir.I32):
            width = "half" if element == ir.I16 else "word"
            for offset, disp in ((1, 0), (0, 2), (1, 2)):
                with self.subTest(element=element, offset=offset, disp=disp):
                    spec, b, t, p = kernels.setup("displaced_load", {"out": 4, "x": 16}, 1,
                                                  {"x": element})
                    loaded = b.load(p["x"], t, width=width, offset=offset, disp=disp)
                    b.store_at(p["out"], t, b._def("or", [loaded, ir.Imm(0)])); b.ret()
                    protocol, _ = mesa.lower(spec)
                    adds = [line.split() for line in protocol.splitlines() if line.startswith("add ")]
                    self.assertEqual(len(adds), 1)
                    self.assertEqual(int(adds[0][-1]), 32)
                    self.assertIn(f"imm {adds[0][3]} {offset+disp} 0 0 32", protocol)

    def test_displaced_stores_have_a_valid_address_addition(self):
        for offset, disp in ((1, 0), (0, 2), (1, 2)):
            with self.subTest(offset=offset, disp=disp):
                spec, b, t, p = kernels.setup("displaced_store", {"out": 16}, 1)
                b.store_at(p["out"], t, b.const(9))
                b.b.ops[-1].attrs.update(offset=offset, disp=disp); b.ret()
                protocol, _ = mesa.lower(spec)
                adds = [line.split() for line in protocol.splitlines() if line.startswith("add ")]
                self.assertEqual(len(adds), 1)
                self.assertEqual(int(adds[0][-1]), 32)
                store = next(line.split() for line in protocol.splitlines() if line.startswith("store "))
                self.assertEqual(store[3], adds[0][1])

    def test_mesa_admission_does_not_inherit_distinct_register_allocator_limit(self):
        spec, b, t, p = kernels.setup("pressure", {"out": 4}, 1)
        for i in range(125): b.const(i)
        b.store_at(p["out"], t, b.const(3)); b.ret()
        with self.assertRaisesRegex(cc.Unsupported, "spilling"): spec.compile()
        protocol, _ = mesa.lower(spec)
        self.assertIn("store ", protocol)

    def test_required_capabilities_and_loop_overflow_are_admitted_independently(self):
        import test_review_regressions as regressions
        for dynamic in (False, True):
            fn = regressions.counted_loop("i16", 65536, dynamic=dynamic)
            spec = kernels.Kernel(fn, 1, {"out": 4})
            with self.assertRaisesRegex(cc.Unsupported, "overflow"): mesa.lower(spec)
        spec = kernels.vector("add", 33)
        spec.requirements = frozenset({"tensor"})
        with self.assertRaisesRegex(cc.Unsupported, "tensor"): mesa.lower(spec)

    def test_forward_fallthrough_without_phis_is_not_a_loop(self):
        spec, b, t, buffers = kernels.setup("fallthrough", {"out": 132, "x": 132}, 33)
        value = b.load(buffers["x"], t)
        after = b.fn.block("after")
        b.br(after); b.at(after)
        b.store_at(buffers["out"], t, value); b.ret()
        protocol, _ = mesa.lower(spec)
        self.assertNotIn("loop_begin", protocol)


class MesaProvenance(unittest.TestCase):
    """Corrupt real captured compiler artifacts; this never claims GPU execution."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        evidence = Path(__file__).resolve().parents[2]/"evidence/macos-mesa-v1"
        shutil.copytree(evidence/"010-add-33/compiler", self.root/"compiler")
        shutil.copyfile(evidence/"mesa-build-identity.json", self.root/"compiler/build-identity.json")
        shutil.copyfile(evidence/"source-sha256.json", self.root/"source-sha256.json")

    def test_real_compiler_output_has_a_matching_descriptor(self):
        from tools.asahi_evidence import mesa_compilation_records
        records = mesa_compilation_records(self.root)
        self.assertEqual(len(records), 1)
        self.assertIn("8ac9b4cbf5c29652dbe94e8ca316b7fd8503d3e4433fb0bd65c8261d33d06537", records)

    def test_changed_lowered_ir_or_adapter_cannot_keep_a_valid_receipt(self):
        from tools.asahi_evidence import mesa_compilation_records
        ir_path = self.root/"compiler/input.ir"
        original = ir_path.read_bytes(); ir_path.write_bytes(original+b"\n")
        with self.assertRaisesRegex(ValueError, "IR/code"): mesa_compilation_records(self.root)
        ir_path.write_bytes(original)
        source_path = self.root/"source-sha256.json"
        sources = json.loads(source_path.read_text()); sources["tools/mesa_agx/bridge.c"] = "changed"
        source_path.write_text(json.dumps(sources))
        with self.assertRaisesRegex(ValueError, "adapter/source"): mesa_compilation_records(self.root)

    def test_unrecorded_scratch_is_rejected_even_when_code_hash_still_matches(self):
        from tools.asahi_evidence import mesa_compilation_records
        path = self.root/"compiler/metadata.json"
        meta = json.loads(path.read_text()); meta["scratch_bytes"] = 16
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "macOS ABI"): mesa_compilation_records(self.root)


class CompilerSelection(unittest.TestCase):
    def test_macos_compiler_choice_is_opt_in_and_keeps_model_arguments(self):
        from examples.macos.backend import select_compiler
        factory, rest = select_compiler(["--output", "run", "--verify"])
        self.assertIsNone(factory); self.assertEqual(rest, ["--output", "run", "--verify"])
        factory, rest = select_compiler(["--compiler", "mesa", "--output", "run", "--verify"])
        self.assertIsNotNone(factory); self.assertEqual(rest, ["--output", "run", "--verify"])

    def test_model_plans_use_the_selected_compiler_for_every_kernel(self):
        from agxforge.g13.gpt2 import GPT2Plan
        from test_gpt2 import Fixture
        class Selected:
            name = "test compiler"
            def __init__(self): self.calls = []
            def __call__(self, spec, name):
                self.calls.append(name); return spec.compile()
            def descriptor(self): return dict(name=self.name)
        provider = Selected()
        plan = GPT2Plan(Fixture(), 4, compiler=provider)
        self.assertEqual(set(provider.calls), set(plan.programs))
        self.assertEqual(len(provider.calls), 21)
        self.assertEqual(plan.descriptor()["compiler"], provider.name)


if __name__ == "__main__": unittest.main()
