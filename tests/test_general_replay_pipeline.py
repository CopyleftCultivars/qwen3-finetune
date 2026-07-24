import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneralReplayPipelineTests(unittest.TestCase):
    def test_source_normalization_preserves_context_and_provenance(self):
        module = load_module("general_source", "scripts/prepare_general_instruction_source.py")
        row = module.normalize_record(
            {"instruction": "Explain this", "context": "A short passage.", "response": "Explanation", "category": "open_qa"},
            7,
            dataset_id="example/source",
            revision="deadbeef",
        )
        self.assertIn("Context:", row["instruction"])
        self.assertEqual(row["output"], "Explanation")
        self.assertEqual(row["_source_id"], "dolly15k:00007")
        self.assertEqual(row["_contamination_status"], "unverified")

    def test_audit_rejects_benchmark_match_but_not_generic_base_overlap(self):
        module = load_module("general_audit", "scripts/audit_general_instruction_source.py")
        reference = [("benchmark:0", "what is the benchmark answer", frozenset("what is the benchmark answer".split()))]
        rows = [
            {"_source_id": "clean", "language": "en", "instruction": "Tell me a joke", "output": "A joke."},
            {"_source_id": "leak", "language": "en", "instruction": "What is the benchmark answer", "output": "B"},
        ]
        clean, rejected = module.audit_rows(rows, reference, 0.9)
        self.assertEqual([row["_source_id"] for row in clean], ["clean"])
        self.assertEqual(rejected[0]["_source_id"], "leak")

    def test_token_matched_selection_is_deterministic_and_bounded(self):
        module = load_module("general_builder", "scripts/build_general_replay_datasets.py")
        rows = [{"_source_id": f"s{i}", "instruction": "q", "output": "a"} for i in range(4)]
        selected_a, total_a = module.choose(rows, [10, 20, 30, 40], 50)
        selected_b, total_b = module.choose(rows, [10, 20, 30, 40], 50)
        self.assertEqual([row["_source_id"] for row in selected_a], [row["_source_id"] for row in selected_b])
        self.assertEqual(total_a, total_b)
        self.assertLessEqual(abs(total_a - 50), 10)

    def test_audited_source_manifest_is_training_allowed_when_present(self):
        manifest_path = ROOT / "data/sources/general_instruction/dolly15k/audited/manifest.json"
        if not manifest_path.exists():
            self.skipTest("audited Dolly source has not been materialized")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["training_allowed"])
        self.assertEqual(manifest["audit"]["rejected_count"], 0)
        self.assertEqual(manifest["audit"]["train_validation_source_overlap"], [])


if __name__ == "__main__":
    unittest.main()
