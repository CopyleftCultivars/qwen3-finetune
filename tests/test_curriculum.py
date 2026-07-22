import unittest

from autoresearch.curriculum import order_pairs, score_pair


class CurriculumTests(unittest.TestCase):
    def test_signal_is_explainable_and_bounded(self) -> None:
        signal = score_pair({"instruction": "Short question?", "output": "Answer."})
        self.assertEqual(signal["signal_version"], "length-structure-v1")
        self.assertGreaterEqual(signal["difficulty_score"], 0)
        self.assertLessEqual(signal["difficulty_score"], 100)
        self.assertIn("features", signal)

    def test_easy_to_hard_is_deterministic(self) -> None:
        pairs = [
            {"instruction": "A short question?", "output": "A short answer."},
            {
                "instruction": (
                    "Which option is best and why, unless the soil is already wet?\n\n"
                    "A. One\nB. Two\nC. Three\nD. Four"
                ),
                "output": "<think>Because the conditions matter.</think> C",
            },
        ]
        ordered, report = order_pairs(pairs, "easy-to-hard")
        self.assertEqual(report["item_count"], 2)
        self.assertEqual(ordered[0]["instruction"], pairs[0]["instruction"])
        again, _ = order_pairs(pairs, "easy-to-hard")
        self.assertEqual(ordered, again)


if __name__ == "__main__":
    unittest.main()
