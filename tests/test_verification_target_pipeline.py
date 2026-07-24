from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerificationTargetTests(unittest.TestCase):
    def test_each_source_question_emits_balanced_proposals(self):
        module = load_module("scripts/build_verification_target.py", "verification_target")
        row = {
            "question_id": "T-1",
            "question": "Which option is correct?",
            "A": "First",
            "B": "Second",
            "C": "Third",
            "D": "Fourth",
            "answer": "C",
            "judge_verdict": "keep",
            "source_passage": "The third option is supported.",
        }
        records = [module.make_record(row, letter, "train") for letter in module.LETTERS]
        self.assertEqual([r["proposed_answer"] for r in records], ["A", "B", "C", "D"])
        self.assertEqual([r["expected_verdict"] for r in records], ["REJECT", "REJECT", "ACCEPT", "REJECT"])
        self.assertTrue(all(r["output"].endswith("\nC") for r in records))
        self.assertTrue(all("<verification>" in r["output"] for r in records))

    def test_exact_benchmark_overlap_is_rejected(self):
        module = load_module("scripts/build_verification_target.py", "verification_target_audit")
        row = {
            "question_id": "T-1",
            "question": "Which option is correct?",
            "A": "First",
            "B": "Second",
            "C": "Third",
            "D": "Fourth",
            "answer": "C",
            "judge_verdict": "keep",
        }
        clean, report = module.audit_source([row], [{"question": row["question"]}], [], [])
        self.assertEqual(clean, [])
        self.assertEqual(report["rejected_count"], 1)

    def test_prediction_parser_requires_verification_and_final_letter(self):
        module = load_module("scripts/score_verification_target.py", "verification_score")
        parsed = module.parse_prediction(
            "<verification>\nproposed_answer: A\nverdict: REJECT\nverified_answer: C\n</verification>\nC"
        )
        self.assertTrue(parsed["parse_valid"])
        self.assertEqual(parsed["verdict"], "REJECT")
        self.assertEqual(parsed["verified_answer"], "C")
        self.assertEqual(parsed["final_answer"], "C")
        self.assertFalse(module.parse_prediction("C")["parse_valid"])

    def test_composition_paths_are_reproducible(self):
        manifest = ROOT / "data/verification_target/proposed_answer_audit/manifest.json"
        self.assertTrue(manifest.exists())
        source_train = ROOT / "data/verification_target/proposed_answer_audit/train.jsonl"
        source_validation = ROOT / "data/verification_target/proposed_answer_audit/validation.jsonl"
        self.assertTrue(source_train.exists())
        self.assertTrue(source_validation.exists())


if __name__ == "__main__":
    unittest.main()
