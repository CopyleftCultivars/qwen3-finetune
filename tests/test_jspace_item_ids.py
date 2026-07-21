import json
import tempfile
import unittest
from pathlib import Path

from autoresearch.jspace_item_ids import build_report


class JspaceItemIdJoinTests(unittest.TestCase):
    def test_resolves_one_based_source_rows_and_limits_to_qmd_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "val.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": "row_000001",
                                "question": "  Which crop is   hardy? ",
                                "A": "A crop",
                                "B": "B crop",
                                "C": "C crop",
                                "D": "D crop",
                                "answer": "A",
                            }
                        ),
                        json.dumps(
                            {
                                "question": "Only in source, not in the map",
                                "A": "A",
                                "B": "B",
                                "C": "C",
                                "D": "D",
                                "answer": "B",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            category_dir = root / "output"
            category_dir.mkdir()
            (category_dir / "organic_farming.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "OF-TRAIN-0001",
                        "question": "Which crop is hardy?",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (category_dir / "other.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "OTHER-TRAIN-0001",
                        "question": "Only in source, not in the map",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            qmd = root / "qmd.jsonl"
            qmd.write_text(
                json.dumps(
                    {
                        "_source_question_id": "OF-TRAIN-0001",
                        "instruction": "x",
                        "output": "y",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            signals = root / "signals.jsonl"
            signals.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "item_id": "row_000001",
                                "source_split": str(source),
                                "bucket": "position_unstable",
                                "category": "organic_farming",
                            }
                        ),
                        json.dumps(
                            {
                                "item_id": "row_000002",
                                "source_split": str(source),
                                "bucket": "position_unstable",
                                "category": "conversations",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_report(
                signals_path=signals,
                qmd_path=qmd,
                category_map_dir=category_dir,
                buckets=["position_unstable"],
                root=root,
            )

        self.assertEqual(report["signal_count"], 2)
        self.assertEqual(report["question_join_count"], 2)
        self.assertEqual(report["selected_qmd_record_count"], 1)
        self.assertEqual(report["selected_question_ids"], ["OF-TRAIN-0001"])
        self.assertEqual(report["category_agreement_count"], 1)
        self.assertEqual(report["category_disagreement_count"], 1)
        self.assertEqual(report["status_counts"]["selected"], 1)
        self.assertEqual(report["status_counts"]["resolved_not_in_qmd"], 1)


if __name__ == "__main__":
    unittest.main()
