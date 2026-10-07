"""Hardware-free rejection checks for the GPT-2 checkpoint and measured ABI."""
import unittest
import hashlib
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from agxforge.g13.gpt2 import GPT2Plan
from examples.macos import benchmark_gpt2 as benchmark
import test_review_regressions as regressions


class Fixture:
    def __init__(self):
        self.config = dict(model_type="gpt2", activation_function="gelu_new", n_embd=8,
                           n_head=2, n_layer=1, n_positions=16, vocab_size=33, layer_norm_epsilon=1e-5)
        shapes = {"wte.weight": [33, 8], "wpe.weight": [16, 8], "ln_f.weight": [8], "ln_f.bias": [8],
                  "h.0.ln_1.weight": [8], "h.0.ln_1.bias": [8], "h.0.ln_2.weight": [8], "h.0.ln_2.bias": [8],
                  "h.0.attn.c_attn.weight": [8, 24], "h.0.attn.c_attn.bias": [24],
                  "h.0.attn.c_proj.weight": [8, 8], "h.0.attn.c_proj.bias": [8],
                  "h.0.mlp.c_fc.weight": [8, 32], "h.0.mlp.c_fc.bias": [32],
                  "h.0.mlp.c_proj.weight": [32, 8], "h.0.mlp.c_proj.bias": [8]}
        self.tensors = {n: dict(shape=s, dtype="F32") for n, s in shapes.items()}


class GPT2Contracts(unittest.TestCase):
    def test_reject_transposed_conv1d_checkpoint(self):
        c = Fixture(); c.tensors["h.0.attn.c_attn.weight"]["shape"] = [24, 8]
        with self.assertRaisesRegex(ValueError, "c_attn.weight"): GPT2Plan(c, 8)

    def test_reject_unimplemented_architectures_and_storage(self):
        for field, value in (("activation_function", "relu"), ("add_cross_attention", True),
                             ("tie_word_embeddings", False), ("scale_attn_weights", False),
                             ("scale_attn_by_inverse_layer_idx", True), ("reorder_and_upcast_attn", True)):
            c = Fixture(); c.config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): GPT2Plan(c, 8)
        c = Fixture(); c.tensors["wte.weight"]["dtype"] = "F16"
        with self.assertRaisesRegex(ValueError, "FP32"): GPT2Plan(c, 8)

    def test_position_limits_and_head_geometry(self):
        for capacity in (0, 3, 32, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError): GPT2Plan(Fixture(), capacity)
        c = Fixture(); c.config["n_head"] = 3
        with self.assertRaisesRegex(ValueError, "head dimension"): GPT2Plan(c, 8)

    def test_both_checkpoint_prefixes_compile_within_profile(self):
        for prefix in ("", "transformer."):
            c = Fixture(); c.tensors = {prefix+n: t for n, t in c.tensors.items()}
            p = GPT2Plan(c, 8)
            self.assertEqual(p.prefix, prefix)
            self.assertEqual(p.descriptor()["architecture"], "gpt2")
            self.assertTrue(all(x.register_halfs <= 80 and len(x.bindings) <= 8 for x in p.programs.values()))


class AppleBaselineEvidence(regressions.ScalarEvidence):
    def setUp(self):
        super().setUp()
        for folder in self.folders:
            path = folder/"launch.json"; launch = json.loads(path.read_text())
            launch.update(executor="Apple-compiled scalar Metal",
                          program_role="logical graph descriptor; G13 bytes are not executed")
            for name in ("shader.metal", "metal-library.bin", "metal-archive.bin"):
                data = name.encode(); (folder/name).write_bytes(data)
                launch[name] = dict(sha256=hashlib.sha256(data).hexdigest())
            (folder/"shader.bin").unlink()
            path.write_text(json.dumps(launch))

    def test_msl_artifacts_must_match_receipt(self):
        from tools.asahi_evidence import audit
        (self.folders[0]/"shader.metal").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "state hash"): audit(self.root)

    def test_baseline_cannot_claim_to_execute_g13(self):
        from tools.asahi_evidence import audit
        path = self.folders[0]/"launch.json"; launch = json.loads(path.read_text())
        launch.pop("program_role"); path.write_text(json.dumps(launch))
        with self.assertRaisesRegex(ValueError, "distinguish MSL"): audit(self.root)

    def test_expected_backend_checks_summary_platform_and_every_launch(self):
        from tools.asahi_evidence import audit
        executor = benchmark.EXECUTORS["apple"]
        summary_path = self.root/"summary.json"
        summary = json.loads(summary_path.read_text()); summary["executor"] = executor
        summary_path.write_text(json.dumps(summary))
        platform_path = self.root/"platform.json"
        platform_path.write_text(json.dumps(dict(executor=executor)))
        self.assertEqual(audit(self.root, expected_executor=executor)["status"], "PASS")
        for path in (summary_path, platform_path, self.folders[-1]/"launch.json"):
            original = path.read_text(); doc = json.loads(original)
            for wrong in (benchmark.EXECUTORS["metal"], None):
                doc["executor"] = wrong; path.write_text(json.dumps(doc))
                with self.subTest(path=path.name, executor=wrong), self.assertRaisesRegex(ValueError, "executor differs"):
                    audit(self.root, expected_executor=executor)
            path.write_text(original)


class BenchmarkContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(checkpoint=self.root, output=self.root/"output", capacity=32,
                                    metal=self.root/"metal", apple=self.root/"apple", iogpu=self.root/"iogpu",
                                    rounds=1, warmup=1)
        self.summaries = {}
        for backend in benchmark.EXECUTORS:
            folder = getattr(self.args, backend); folder.mkdir()
            summary = dict(status="PASS", verified=True, executor=benchmark.EXECUTORS[backend],
                           input_tokens=[0, 1], checks=[dict(token=i, next_token=i+1) for i in range(4)])
            self.summaries[backend] = summary
            for name, doc in (("summary.json", summary), ("checkpoint.json", {}),
                              ("source-sha256.json", {}), ("plan.json", dict(architecture="gpt2", capacity=32))):
                (folder/name).write_text(json.dumps(doc))
        self.audit = self.enterContext(patch.object(benchmark, "audit", return_value=dict(launches=1, tensor_checks=1)))
        self.enterContext(patch.object(benchmark, "source_identity", return_value={}))
        checkpoint = self.enterContext(patch.object(benchmark, "Checkpoint")); checkpoint.return_value.receipt.return_value = {}
        self.run_worker = self.enterContext(patch.object(benchmark.subprocess, "run", side_effect=self.worker))

    def worker(self, command, **kwargs):
        backend = command[command.index("--backend")+1]
        output = Path(command[command.index("--output")+1])
        sample = dict(decode_tokens_per_second=10 if backend == "metal" else 20,
                      prefill_seconds=.1, total_seconds=.3)
        output.write_text(json.dumps(dict(status="PASS", samples=[sample])))

    def write_summary(self, backend):
        (getattr(self.args, backend)/"summary.json").write_text(json.dumps(self.summaries[backend]))

    def test_different_prompt_boundaries_rejected_before_workers(self):
        self.summaries["apple"]["input_tokens"] = [0]
        self.write_summary("apple")
        with self.assertRaisesRegex(ValueError, "prompts differ"): benchmark.compare(self.args)
        self.run_worker.assert_not_called()

    def test_wrong_backend_identity_rejected_before_workers(self):
        self.summaries["apple"]["executor"] = benchmark.EXECUTORS["metal"]
        self.write_summary("apple")
        with self.assertRaisesRegex(ValueError, "executor differs"): benchmark.compare(self.args)
        self.run_worker.assert_not_called()

    def test_invalid_prompt_or_unverified_bundle_rejected(self):
        for field, value, error in (("input_tokens", [], "prompt differs"),
                                    ("input_tokens", [1, 0], "prompt differs"),
                                    ("input_tokens", [0, 1, 2, 3], "decode position"),
                                    ("verified", False, "verified baseline")):
            summary = dict(self.summaries["metal"], **{field: value})
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, error):
                benchmark.validate_baseline(summary, "metal")

    def test_matching_three_backend_comparison_reaches_workers(self):
        with redirect_stdout(io.StringIO()): benchmark.compare(self.args)
        self.assertEqual(self.run_worker.call_count, 3)
        for call in self.audit.call_args_list:
            folder = call.args[0]
            self.assertEqual(call.kwargs["expected_executor"], benchmark.EXECUTORS[folder.name])
        report = json.loads((self.args.output/"summary.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["apple_over_g13_metal_decode_ratio"], 2)
        self.assertEqual(report["iogpu_over_metal_decode_ratio"], 2)


if __name__ == "__main__": unittest.main()
