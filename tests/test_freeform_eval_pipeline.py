from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autoresearch.run_freeform_evaluation import select_subset, summarize_rubric
from eval_freeform.guards import assert_not_sealed_path, load_dev_ids, validate_dev_ids


ROOT = Path(__file__).resolve().parents[1]


class FreeformEvaluationPipelineTests(unittest.TestCase):
    def test_sealed_and_spanish_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            assert_not_sealed_path(ROOT / "eval_freeform" / "test.jsonl")
        with self.assertRaises(ValueError):
            assert_not_sealed_path(ROOT.parent / "spanish-benchmark-workspace" / "dev.jsonl")

    def test_dev_id_guard_rejects_unknown_ids_without_reading_test_file(self):
        dev_ids = load_dev_ids(ROOT / "eval_freeform" / "split_manifest.json")
        self.assertGreater(len(dev_ids), 0)
        validate_dev_ids([next(iter(dev_ids))], ROOT / "eval_freeform" / "split_manifest.json")
        with self.assertRaises(ValueError):
            validate_dev_ids(["not-a-canonical-dev-id"], ROOT / "eval_freeform" / "split_manifest.json")

    def test_subset_is_deterministic_and_balanced(self):
        rows = [
            {"id": "a1", "type": "a"},
            {"id": "b1", "type": "b"},
            {"id": "a2", "type": "a"},
            {"id": "b2", "type": "b"},
        ]
        selected = select_subset(rows, 1)
        self.assertEqual([row["id"] for row in selected], ["a1", "b1"])

    def test_rubric_summary_requires_exact_selected_ids(self):
        dev_rows = [
            json.loads(line)
            for line in (ROOT / "eval_freeform" / "dev.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rubric.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": row["id"],
                            "type": row["type"],
                            "weighted_score": 0.5,
                            "grader_role": "routine",
                            "grader_model": "mock",
                            "mock_grader": True,
                        }
                    )
                    + "\n"
                    for row in dev_rows
                ),
                encoding="utf-8",
            )
            summary = summarize_rubric(path, {row["id"] for row in dev_rows})
            self.assertEqual(summary["item_count"], 2)
            self.assertEqual(summary["percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
