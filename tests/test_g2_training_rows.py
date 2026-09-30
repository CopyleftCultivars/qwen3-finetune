#!/usr/bin/env python3
"""#238 G2 unit tests: shared training-row selection (CPU-only, hermetic)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from g2_training_rows import load_examples  # noqa: E402


class LoadExamplesTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="g2rows_")
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)

    def _write(self, rows: list[dict]) -> Path:
        path = self.tmp / "train.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_first_n_usable_rows_in_order(self):
        path = self._write([
            {"instruction": "i1", "output": "o1"},
            {"instruction": "", "output": "oX"},
            {"instruction": "i2"},
            {"instruction": "i3", "output": "o3"},
        ])
        rows = load_examples(path, 2)
        self.assertEqual([r["instruction"] for r in rows], ["i1", "i3"])

    def test_short_file_is_a_hard_error(self):
        path = self._write([{"instruction": "i1", "output": "o1"}])
        with self.assertRaises(SystemExit):
            load_examples(path, 2)

    def test_values_coerced_to_str(self):
        path = self._write([{"instruction": 7, "output": "o"}])
        rows = load_examples(path, 1)
        self.assertEqual(rows[0]["instruction"], "7")


if __name__ == "__main__":
    unittest.main()
