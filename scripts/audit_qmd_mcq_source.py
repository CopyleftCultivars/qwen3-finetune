#!/usr/bin/env python3
"""Audit and package flat QMD-generated MCQs for training use.

This is intentionally a question-level boundary check.  It rejects exact
normalized question-stem reuse and high lexical overlap against the English
v5clean anchor, validates the flat QMD record shape, and emits an auditable
clean copy.  It does not claim that QMD source documents are independent of
the benchmark's source documents; that limitation is recorded in the report.

The sealed benchmark is never an accepted input to this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+")
LETTERS = ("A", "B", "C", "D")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_RE.findall(text))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(value)
    return rows


def _benchmark_fields(path: Path) -> dict[str, tuple[str, str]]:
    fields: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(_load_jsonl(path)):
        question_id = str(row.get("question_id", index))
        for field in ("question", *LETTERS):
            value = str(row.get(field, "")).strip()
            if value:
                fields.setdefault(normalize(value), (question_id, field))
    return fields


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def audit(
    source_rows: list[dict[str, Any]],
    benchmark_fields: dict[str, tuple[str, str]],
    *,
    source_path: Path,
    benchmark_path: Path,
    lexical_threshold: float,
    excluded_source_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_sets = [
        (question_id, field, set(value.split()))
        for value, (question_id, field) in benchmark_fields.items()
    ]
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    categories: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    max_overlap = {
        "score": 0.0,
        "source_question_id": None,
        "benchmark_question_id": None,
    }

    for index, row in enumerate(source_rows):
        source_id = str(row.get("question_id", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip().upper()
        reason: str | None = None
        overlap_source_field: str | None = None
        overlap_benchmark_id: str | None = None
        overlap_benchmark_field: str | None = None

        if not source_id:
            reason = "missing_question_id"
        elif excluded_source_ids and source_id in excluded_source_ids:
            reason = "external_spanish_transfer_overlap"
        elif source_id in seen_ids:
            reason = "duplicate_question_id"
        elif not question or any(not str(row.get(letter, "")).strip() for letter in LETTERS):
            reason = "missing_question_or_option"
        elif answer not in LETTERS:
            reason = "invalid_answer_letter"
        elif str(row.get("judge_verdict", "")).strip().lower() != "keep":
            reason = "not_judge_kept"
        else:
            for source_field in ("question", *LETTERS):
                source_value = str(row.get(source_field, "")).strip()
                normalized = normalize(source_value)
                if normalized in benchmark_fields:
                    benchmark_id, benchmark_field = benchmark_fields[normalized]
                    overlap_source_field = source_field
                    overlap_benchmark_id = benchmark_id
                    overlap_benchmark_field = benchmark_field
                    reason = (
                        "exact_benchmark_question_overlap"
                        if source_field == "question" and benchmark_field == "question"
                        else "exact_benchmark_text_overlap"
                    )
                    break
                source_tokens = set(normalized.split())
                best_score = 0.0
                best_id = ""
                best_field = ""
                for benchmark_id, benchmark_field, benchmark_tokens in benchmark_sets:
                    score = _jaccard(source_tokens, benchmark_tokens)
                    if score > best_score:
                        best_score, best_id, best_field = score, benchmark_id, benchmark_field
                if best_score > max_overlap["score"]:
                    max_overlap = {
                        "score": round(best_score, 6),
                        "source_question_id": source_id,
                        "source_field": source_field,
                        "benchmark_question_id": best_id,
                        "benchmark_field": best_field,
                    }
                if best_score >= lexical_threshold:
                    overlap_source_field = source_field
                    overlap_benchmark_id = best_id
                    overlap_benchmark_field = best_field
                    reason = "high_lexical_benchmark_overlap"
                    break

        if reason:
            rejected.append(
                {
                    "source_index": index,
                    "source_question_id": source_id or None,
                    "reason": reason,
                    "question": question,
                    "benchmark_question_id": (
                        benchmark_fields.get(normalize(question), (None, None))[0]
                    ),
                    "source_field": overlap_source_field,
                    "benchmark_question_id_matched": overlap_benchmark_id,
                    "benchmark_field": overlap_benchmark_field,
                }
            )
            continue

        seen_ids.add(source_id)
        category = str(row.get("category", "unknown")).strip() or "unknown"
        categories[category] += 1
        answers[answer] += 1
        clean_row = dict(row)
        clean_row.update(
            {
                "_source": "qmd_mcq_generated_clean",
                "_source_question_id": source_id,
                "_contamination_status": "passed",
                "_contamination_audit": "qmd_mcq_source_contamination_audit_v1",
                "_benchmark_question_text_overlap": "not_detected",
            }
        )
        clean.append(clean_row)

    report = {
        "format": "qmd_mcq_source_contamination_audit_v1",
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "benchmark": str(benchmark_path),
        "source_count": len(source_rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "benchmark_field_count": len(benchmark_fields),
        "benchmark_question_count": len({item[0] for item in benchmark_fields.values()}),
        "exact_question_overlap_count": sum(
            item["reason"] == "exact_benchmark_question_overlap" for item in rejected
        ),
        "exact_text_overlap_count": sum(
            item["reason"] == "exact_benchmark_text_overlap" for item in rejected
        ),
        "high_lexical_overlap_count": sum(
            item["reason"] == "high_lexical_benchmark_overlap" for item in rejected
        ),
        "lexical_overlap_threshold": lexical_threshold,
        "maximum_nonmatching_token_jaccard": max_overlap,
        "source_categories": dict(sorted(categories.items())),
        "answer_counts": {letter: answers.get(letter, 0) for letter in LETTERS},
        "rejected": rejected,
        "passed": not rejected,
        "clean_output_ready": bool(clean),
        "training_allowed_for_clean_output": bool(clean),
        "selection_status": "clean_subset_emitted" if clean else "no_clean_items",
        "sealed_test_paths_read": [],
        "notes": [
            "Question-level audit; semantic similarity is not claimed absent.",
            "The QMD source documents are not asserted to be independent of benchmark source documents.",
            "Only the clean emitted copy may be translated into the Spanish training package.",
        ],
        "external_excluded_source_ids": sorted(excluded_source_ids or set()),
    }
    return clean, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lexical-threshold", type=float, default=0.85)
    parser.add_argument("--exclude-source-id", action="append", default=[])
    args = parser.parse_args()

    for path in (args.input, args.benchmark):
        if not path.exists():
            parser.error(f"file not found: {path}")
    if not 0.0 < args.lexical_threshold <= 1.0:
        parser.error("--lexical-threshold must be in (0, 1]")

    clean, report = audit(
        _load_jsonl(args.input),
        _benchmark_fields(args.benchmark),
        source_path=args.input,
        benchmark_path=args.benchmark,
        lexical_threshold=args.lexical_threshold,
        excluded_source_ids=set(args.exclude_source_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in clean:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "source_count", "clean_count", "rejected_count", "exact_question_overlap_count",
        "high_lexical_overlap_count", "answer_counts", "passed",
    )}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
