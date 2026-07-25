from __future__ import annotations

import importlib.util
import json
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistractorQualityPipelineTests(unittest.TestCase):
    def test_rotation_preserves_options_and_balances_target_letters(self):
        module = load_module("scripts/build_distractor_quality_source.py", "distractor_quality_source")
        row = {
            "question_id": "DQ-TEST-0001",
            "question": "Which option is supported?",
            "A": "First plausible answer",
            "B": "Second plausible answer",
            "C": "Third plausible answer",
            "D": "Fourth plausible answer",
            "answer": "C",
            "source_passage": "The third answer is supported by the passage.",
            "judge_model": "test",
            "quality_scores": {"composite": 4.6},
        }
        records = [module.rotate(row, shift) for shift in range(4)]
        letters = [re.search(r"</think>\s*([ABCD])$", record["output"]).group(1) for record in records]
        self.assertEqual(letters, list("CDA B".replace(" ", "")))
        self.assertTrue(all(len(set(re.findall(r"^[ABCD]\. (.+)$", record["instruction"], re.MULTILINE))) == 4 for record in records))
        self.assertTrue(all(record["language"] == "en" for record in records))

    def test_frozen_source_is_balanced_and_disjoint(self):
        manifest = json.loads((ROOT / "data/sources/qmd_mcq_distractor_quality_manifest.json").read_text())
        self.assertTrue(manifest["training_allowed"])
        self.assertEqual(manifest["expanded_train_letter_counts"], {letter: 53 for letter in "ABCD"})
        self.assertEqual(manifest["expanded_validation_letter_counts"], {letter: 9 for letter in "ABCD"})
        train = [json.loads(line) for line in (ROOT / "data/sources/qmd_mcq_distractor_quality_train.jsonl").read_text().splitlines()]
        validation = [json.loads(line) for line in (ROOT / "data/sources/qmd_mcq_distractor_quality_validation.jsonl").read_text().splitlines()]
        train_ids = {row["_source_question_id"] for row in train}
        validation_ids = {row["_source_question_id"] for row in validation}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(Counter(row["judge_verdict"] for row in train + validation), {"keep": 248})
        self.assertTrue(all(row["_contamination_status"] == "passed" for row in train + validation))


if __name__ == "__main__":
    unittest.main()
