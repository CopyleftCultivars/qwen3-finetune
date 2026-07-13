#!/usr/bin/env python3
"""Flatten generate_mcqs_rag.py's per-topic JSON output into a single flat JSONL,
in the canonical schema quality/llm_judge.py expects (question_id, question, A-D,
answer, topic, category, source_passage).

Mirrors quality/merge_regen_round.py's flattening logic, but reads from any input
directory (not the hardcoded regenerated_mcqs/{category}_round{N}/ path) and writes
to any output file (not benchmarks/{category}/...) — this is training data, not a
benchmark file, so it doesn't belong in that directory structure.

Usage:
    python flatten_to_jsonl.py \\
        --input-dir output/permaculture_pilot \\
        --category "Permaculture Benchmark" \\
        --prefix PC-TRAIN \\
        --output output/permaculture_pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_LETTERS = ["A", "B", "C", "D"]


def flatten_one(q: dict, category: str, question_id: str) -> dict | None:
    opts = q.get("options", {})
    if not all(letter in opts for letter in _LETTERS):
        return None
    ca = q.get("correct_answer", "")
    if ca not in _LETTERS:
        return None
    question = q.get("question", "").strip()
    if not question:
        return None
    return {
        "question_id": question_id,
        "question": question,
        "A": opts["A"],
        "B": opts["B"],
        "C": opts["C"],
        "D": opts["D"],
        "answer": ca,
        "topic": q.get("topic", ""),
        "category": category,
        "source_passage": q.get("source_passage", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--category", required=True, help="Category label, e.g. 'Permaculture Benchmark'")
    ap.add_argument("--prefix", required=True, help="Question ID prefix, e.g. PC-TRAIN")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        print(f"error: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    flat: list[dict] = []
    skipped = 0
    for path in sorted(args.input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            idx = len(flat) + skipped + 1
            qid = f"{args.prefix}-{idx:04d}"
            rec = flatten_one(q, args.category, qid)
            if rec is None:
                skipped += 1
                continue
            flat.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for rec in flat:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(flat)} questions to {args.output} ({skipped} skipped, malformed)", file=sys.stderr)


if __name__ == "__main__":
    main()
