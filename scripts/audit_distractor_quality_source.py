#!/usr/bin/env python3
"""Audit a locally authored distractor-repair MCQ source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


LETTERS = ("A", "B", "C", "D")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(TOKEN_RE.findall(value))


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def benchmark_text(path: Path) -> set[str]:
    values: set[str] = set()
    for row in load(path):
        for field in ("question", *LETTERS):
            if row.get(field):
                values.add(normalize(row[field]))
        instruction = str(row.get("instruction", ""))
        if instruction:
            values.add(normalize(instruction))
    return values


def base_questions(paths: list[Path]) -> set[str]:
    values: set[str] = set()
    for path in paths:
        for row in load(path):
            if row.get("question"):
                values.add(normalize(row["question"]))
            instruction = str(row.get("instruction", ""))
            if instruction:
                values.add(normalize(instruction.split("\n\nA.", 1)[0]))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--base", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lexical-threshold", type=float, default=0.85)
    args = parser.parse_args()
    rows = load(args.input)
    benchmark = benchmark_text(args.benchmark)
    base = base_questions(args.base)
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows):
        question_id = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip().upper()
        reason: str | None = None
        if not question_id or question_id in seen_ids:
            reason = "missing_or_duplicate_question_id"
        elif not question:
            reason = "missing_question"
        elif answer not in LETTERS:
            reason = "invalid_answer_letter"
        elif any(not str(row.get(letter, "")).strip() for letter in LETTERS):
            reason = "missing_option"
        elif len({normalize(row[letter]) for letter in LETTERS}) != 4:
            reason = "duplicate_options"
        elif any(normalize(row[letter]) in {"a", "b", "c", "d"} for letter in LETTERS):
            reason = "bare_letter_option"
        elif normalize(question) in benchmark:
            reason = "exact_benchmark_question_overlap"
        elif normalize(question) in base:
            reason = "exact_base_question_overlap"
        if reason:
            rejected.append({"index": index, "question_id": question_id, "reason": reason})
            continue
        seen_ids.add(question_id)
        clean_row = dict(row)
        clean_row.update(
            {
                "judge_verdict": "pending_quality_judge",
                "_contamination_status": "passed",
                "_contamination_audit": "distractor_quality_source_audit_v1",
            }
        )
        clean.append(clean_row)

    report = {
        "format": "distractor_quality_source_audit_v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "benchmark": str(args.benchmark.resolve()),
        "base_paths": [str(path.resolve()) for path in args.base],
        "input_count": len(rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "answer_counts": {letter: Counter(row["answer"] for row in clean).get(letter, 0) for letter in LETTERS},
        "base_exact_overlap_count": sum(item["reason"] == "exact_base_question_overlap" for item in rejected),
        "benchmark_exact_overlap_count": sum(item["reason"] == "exact_benchmark_question_overlap" for item in rejected),
        "training_allowed": bool(clean) and not any(
            item["reason"] == "exact_benchmark_question_overlap" for item in rejected
        ),
        "sealed_test_paths_read": [],
        "notes": [
            "This is a source-level structural and exact-overlap audit; semantic quality is judged separately.",
            "Final target-letter balance is supplied by the existing cyclic-shift training builder.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in clean:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("input_count", "clean_count", "rejected_count", "answer_counts", "training_allowed")}, indent=2, sort_keys=True))
    return 0 if report["training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
