#!/usr/bin/env python3
"""#238 G2: show the tiny-adaptation training rows are not benchmark rows.

The adaptation steps in scripts/g2_qwen35_tiny_adapt.py must run on
NON-benchmark examples, so this reuses the F2 (#139) contamination signals
against the frozen 210-item ladder: normalized exact match and max 8-gram
Jaccard over (instruction, output) text. Any hit >= the alarm threshold stops
the adaptation from being called "non-benchmark".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g2_training_rows import load_examples  # noqa: E402
from ladder_contamination import (  # noqa: E402
    BENCH,
    jaccard,
    norm,
    shingles,
    split_text,
)

ALARM_JACCARD = 0.6


def ladder_texts(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = split_text(row)
            items.append({"text": text, "shingles": shingles(text),
                          "norm": " ".join(norm(text).split())})
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bench", default=str(BENCH))
    args = parser.parse_args(argv)

    examples = load_examples(Path(args.data), args.steps * args.batch_size)
    ladder = ladder_texts(Path(args.bench))

    exact_hits, max_jaccard, alarm = [], 0.0, []
    for index, example in enumerate(examples):
        example_norm = " ".join(norm(split_text(example)).split())
        example_shingles = shingles(split_text(example))
        for item in ladder:
            if item["norm"] == example_norm:
                exact_hits.append(index)
            score = jaccard(example_shingles, item["shingles"])
            max_jaccard = max(max_jaccard, score)
            if score >= ALARM_JACCARD:
                alarm.append({"example": index, "jaccard": round(score, 4),
                              "ladder_text": item["text"][:160]})

    report = {
        "issue": 238,
        "stage": "g2_benchmark_overlap",
        "train_file": args.data,
        "examples_checked": len(examples),
        "ladder_file": str(Path(args.bench).name),
        "ladder_items": len(ladder),
        "exact_normalized_hits": exact_hits,
        "max_8gram_jaccard": round(max_jaccard, 4),
        "alarm_threshold": ALARM_JACCARD,
        "alarms": alarm,
        "verdict": "NON_BENCHMARK" if not exact_hits and not alarm else "OVERLAP",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "NON_BENCHMARK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
