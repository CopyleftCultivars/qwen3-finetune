from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenBookGroundedPipelineTests(unittest.TestCase):
    def test_context_bundle_has_one_evidence_record_and_two_negatives(self):
        module = load_module(
            "scripts/build_open_book_grounded_source.py", "grounded_source"
        )
        documents = [
            {"source_id": "recipe:0001:aaaa", "source_kind": "recipe", "record": {}, "source_path": "a", "source_sha256": "a", "source_record_sha256": "a"},
            {"source_id": "recipe:0002:bbbb", "source_kind": "recipe", "record": {}, "source_path": "b", "source_sha256": "b", "source_record_sha256": "b"},
            {"source_id": "recipe:0003:cccc", "source_kind": "recipe", "record": {}, "source_path": "c", "source_sha256": "c", "source_record_sha256": "c"},
        ]
        candidate = {"document": documents[0], "variant": "answer", "question": "Which?", "answer": "FAA"}
        row = module.make_row(candidate, documents, split="train")
        self.assertEqual(len(row["context_source_ids"]), 3)
        self.assertIn(row["evidence_source_id"], row["context_source_ids"])
        self.assertIn("Answer: FAA", row["output"])
        self.assertIn("Evidence: recipe:0001:aaaa", row["output"])

    def test_audit_rejects_high_overlap_and_marks_clean_rows(self):
        module = load_module(
            "scripts/build_open_book_grounded_source.py", "grounded_audit"
        )
        row = {
            "id": "clean",
            "instruction": "Question: Which pH range is listed for Rice?",
            "output": "Answer: 5.5-6.8\nEvidence: crop_requirement:0001:abcd",
            "language": "en",
            "grounded_task": module.TASK,
            "split": "train",
            "context_source_ids": ["crop_requirement:0001:abcd", "x", "y"],
            "evidence_source_id": "crop_requirement:0001:abcd",
        }
        clean, report = module.audit_rows(
            [row],
            [set(module.normalize(row["instruction"]).split())],
            source_path=ROOT / "data/sources/gemma4/spaces/data",
            benchmark_path=ROOT / "data/val_v3.jsonl",
        )
        self.assertEqual(len(clean), 0)
        self.assertEqual(report["rejected_count"], 1)

    def test_generated_manifest_is_balanced_by_source_document(self):
        manifest_path = ROOT / "data/sources/open_book_grounded/manifest.json"
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text())
        self.assertTrue(manifest["training_allowed"])
        self.assertEqual(manifest["train"]["count"], 256)
        self.assertEqual(manifest["validation"]["count"], 64)
        self.assertEqual(manifest["split"]["train_validation_source_overlap"], [])
        train = [json.loads(line) for line in (ROOT / "data/sources/open_book_grounded/train.jsonl").read_text().splitlines()]
        validation = [json.loads(line) for line in (ROOT / "data/sources/open_book_grounded/validation.jsonl").read_text().splitlines()]
        train_context = {source_id for row in train for source_id in row["context_source_ids"]}
        validation_context = {source_id for row in validation for source_id in row["context_source_ids"]}
        self.assertFalse(train_context & validation_context)
        self.assertTrue(all(row["_contamination_status"] == "passed" for row in train + validation))


class OpenBookGroundedScorerTests(unittest.TestCase):
    def test_prediction_parser_requires_both_lines(self):
        module = load_module(
            "scripts/score_open_book_grounded.py", "grounded_scorer"
        )
        parsed = module.parse_prediction("Answer: 5.5-6.8\nEvidence: crop_requirement:0001:abcd")
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["answer"], "5.5-6.8")
        self.assertFalse(module.parse_prediction("5.5-6.8")["parse_valid"])


if __name__ == "__main__":
    unittest.main()
