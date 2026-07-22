import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_cloze_contamination import audit
from scripts.generate_cloze_from_conversations import build_items, make_cloze_pair


class ClozePipelineTests(unittest.TestCase):
    def test_masks_a_domain_phrase_and_preserves_exact_target(self) -> None:
        item = make_cloze_pair(
            "How should compacted soil be improved?",
            "Incorporate organic matter to improve soil structure and resilience.",
            source_line=7,
        )
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["output"], "organic matter")
        self.assertIn("[BLANK]", item["instruction"])
        self.assertEqual(item["_contamination_status"], "unverified")

    def test_skips_system_prompt_pair_and_caps_repeated_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "chat.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": "You are a Natural Farming assistant."},
                            {"from": "gpt", "value": "Nitrogen is important for growth."},
                            {"from": "human", "value": "Question one"},
                            {"from": "gpt", "value": "Nitrogen supports leaf growth and soil health."},
                            {"from": "human", "value": "Question two"},
                            {"from": "gpt", "value": "Nitrogen supports root growth and soil health."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            items = build_items(source, None, 42, 8, 700, 1)
        self.assertEqual(len(items), 1)
        self.assertNotIn("You are a Natural Farming", items[0]["_source_instruction"])

    def test_audit_marks_safe_items_and_rejects_exact_question_reuse(self) -> None:
        safe = {
            "_cloze_id": "safe",
            "_source_instruction": "How can mulch conserve water in a garden?",
            "instruction": "Complete [BLANK]",
            "output": "mulch",
        }
        contaminated = {
            "_cloze_id": "bad",
            "_source_instruction": "What is the primary purpose of compost?",
            "instruction": "Complete [BLANK]",
            "output": "soil health",
        }
        kept, report = audit(
            [safe, contaminated],
            [{"question": "What is the primary purpose of compost?"}],
        )
        self.assertEqual([item["_cloze_id"] for item in kept], ["safe"])
        self.assertEqual(report["rejected_count"], 1)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
