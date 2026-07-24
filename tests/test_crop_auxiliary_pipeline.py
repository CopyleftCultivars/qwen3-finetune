import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CropAuxiliaryPipelineTests(unittest.TestCase):
    def test_auxiliary_task_manifests_have_equal_pilot_dose_and_clean_splits(self):
        tasks = ("summarization", "ner", "structured_extraction", "open_book_qa")
        summaries = []
        for task in tasks:
            manifest = json.loads((ROOT / "data" / "auxiliary" / task / "manifest.json").read_text())
            self.assertEqual(manifest["language"], "en")
            self.assertEqual(manifest["contamination"]["rejected_count"], 0)
            self.assertTrue(manifest["training_allowed"])
            self.assertEqual(manifest["train"]["count"], 256)
            self.assertEqual(manifest["validation"]["count"], 64)
            self.assertEqual(manifest["train_validation_source_overlap"], [])
            summaries.append((manifest["train"]["count"], manifest["validation"]["count"]))
        self.assertEqual(len(set(summaries)), 1)


    def test_auxiliary_rows_preserve_task_provenance(self):
        for task in ("summarization", "ner", "structured_extraction", "open_book_qa"):
            path = ROOT / "data" / "auxiliary" / task / "train.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 256)
            for row in rows[:5]:
                self.assertEqual(row["auxiliary_task"], task)
                self.assertEqual(row["language"], "en")
                self.assertEqual(row["_contamination_status"], "passed")
                self.assertEqual(row["provenance"]["authoring_method"], "deterministic_template_v1")
                self.assertTrue(row["provenance"]["source_record_sha256"])


    def test_auxiliary_builder_rejects_sealed_or_spanish_paths(self):
        module = load_module("auxiliary_builder", "scripts/build_auxiliary_lever_dataset.py")
        for path in (
            Path("/tmp/spanish-benchmark-workspace/benchmarks/transfer.jsonl"),
            Path("/tmp/eval_freeform/test.jsonl"),
        ):
            with self.assertRaises(ValueError):
                module.assert_training_safe(path)


    def test_auxiliary_ner_and_structured_outputs_are_json(self):
        module = load_module("crop_auxiliary_builder", "scripts/build_crop_auxiliary_datasets.py")
        documents = module.build_documents(ROOT / "data/sources/gemma4/spaces/data")
        self.assertTrue(json.loads(module.ner_output(documents[0]))["entities"])
        self.assertEqual(json.loads(module.structured_output(documents[0]))["record_type"], "recipe")
