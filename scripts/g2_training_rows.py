#!/usr/bin/env python3
"""#238 G2: shared selection rule for the tiny-adaptation training rows.

Both the trainer (scripts/g2_qwen35_tiny_adapt.py) and the benchmark-overlap
check (scripts/g2_benchmark_overlap.py) must pick exactly the same rows, so the
overlap result describes the examples that were actually trained on.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_examples(path: Path, count: int) -> list[dict[str, str]]:
    """First `count` rows of an instruction/output JSONL that carry both fields."""
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("instruction") and row.get("output"):
                rows.append({"instruction": str(row["instruction"]),
                             "output": str(row["output"])})
            if len(rows) >= count:
                break
    if len(rows) < count:
        raise SystemExit(f"need {count} usable rows in {path}, found {len(rows)}")
    return rows
