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


class MultiturnReflectionPipelineTests(unittest.TestCase):
    def test_thread_targets_preserve_structure_and_controls_same_source(self):
        module = load_module("multiturn_builder", "scripts/build_multiturn_reflection_datasets.py")
        thread = {
            "source_id": "conversation:test",
            "source_line": 1,
            "source_file": "/tmp/source.jsonl",
            "source_sha256": "file",
            "source_record_sha256": "record",
            "all_pair_count": 2,
            "window_pair_count": 2,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ],
        }
        preserved = module.target_row("multiturn", thread)
        flattened = module.target_row("flattened_control", thread)
        self.assertEqual(len(preserved["messages"]), 4)
        self.assertEqual(len(flattened["messages"]), 2)
        self.assertEqual(preserved["source_id"], flattened["source_id"])
        self.assertEqual(preserved["target_output"], flattened["target_output"])

    def test_reflection_rows_have_corrected_answer_targets(self):
        module = load_module("multiturn_builder_reflection", "scripts/build_multiturn_reflection_datasets.py")
        thread = {
            "source_id": "conversation:reflection",
            "source_line": 2,
            "source_file": "/tmp/source.jsonl",
            "source_sha256": "file",
            "source_record_sha256": "record",
            "all_pair_count": 2,
            "window_pair_count": 2,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "A complete answer with useful context."},
            ],
        }
        for task in ("interrupted_counterfactual", "answer_audit"):
            row = module.target_row(task, thread)
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertIn("Corrected answer:", row["output"])
            self.assertEqual(row["provenance"]["reflection_error_type"] in {"unsupported_universal_claim", "incomplete_draft"}, True)

    def test_generated_manifests_have_equal_clean_doses(self):
        root = ROOT / "data" / "multiturn_reflection"
        if not (root / "manifest.json").exists():
            self.skipTest("generated structural datasets are not present")
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["eligible_thread_count"], 180)
        self.assertEqual(manifest["pilot_dose"]["train_count"], 140)
        self.assertEqual(manifest["pilot_dose"]["validation_count"], 40)
        for task in manifest["tasks"]:
            task_manifest = json.loads((root / task / "manifest.json").read_text())
            self.assertTrue(task_manifest["training_allowed"])
            self.assertEqual(task_manifest["contamination"]["rejected_count"], 0)
            self.assertEqual(task_manifest["train_validation_source_overlap"], [])

    def test_structured_composer_accepts_messages_and_rejects_spanish(self):
        module = load_module("structured_composer", "scripts/build_structured_lever_dataset.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "instruction": "q", "output": "a"}
            for name in ("base.jsonl", "source.jsonl"):
                (root / name).write_text(json.dumps(row) + "\n")
            loaded = module.load(root / "base.jsonl")
            self.assertEqual(loaded[0]["messages"][-1]["role"], "assistant")
        with self.assertRaises(ValueError):
            module.reject_path(Path("/tmp/spanish-benchmark-workspace/train.jsonl"))


if __name__ == "__main__":
    unittest.main()
