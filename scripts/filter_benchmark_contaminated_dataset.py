#!/usr/bin/env python3
"""Remove rows with exact or high-overlap English benchmark text.

This is the write-producing companion to
check_spanish_transfer_contamination.py. It is used only for the candidate
training base, never for sealed evaluation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MIN_EXACT_CHARS = 48
MIN_OVERLAP_TOKENS = 6


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize(text)))


def load_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return values


def benchmark_fields(path: Path) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for line_number, row in enumerate(load_jsonl(path), 1):
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_question_id") or row.get("id") or line_number)
        for field in ("question", "A", "B", "C", "D"):
            value = row.get(field)
            if not isinstance(value, str) or len(normalize(value)) < MIN_EXACT_CHARS:
                continue
            fields.append({"source_question_id": source_id, "field": field, "normalized": normalize(value), "tokens": tokens(value)})
    return fields


def row_match(row: Any, fields: list[dict[str, Any]], token_index: dict[str, set[int]], threshold: float) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return {"reason": "invalid_json_object"}
    for value in string_values(row):
        normalized = normalize(value)
        row_tokens = tokens(value)
        if len(row_tokens) < MIN_OVERLAP_TOKENS:
            continue
        ranked_tokens = sorted(row_tokens, key=lambda token: len(token_index.get(token, set())))[:8]
        candidates = {candidate for token in ranked_tokens for candidate in token_index.get(token, set())}
        for index in candidates:
            field = fields[index]
            if len(field["normalized"]) >= MIN_EXACT_CHARS and field["normalized"] in normalized:
                return {"reason": "exact_normalized_substring", "benchmark_question_id": field["source_question_id"], "benchmark_field": field["field"]}
            intersection = len(row_tokens & field["tokens"])
            union = len(row_tokens | field["tokens"])
            score = intersection / union if union else 0.0
            if intersection >= MIN_OVERLAP_TOKENS and score >= threshold:
                return {"reason": "high_token_jaccard", "benchmark_question_id": field["source_question_id"], "benchmark_field": field["field"], "token_jaccard": score}
    return None


def string_values(value: Any):
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from string_values(child)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lexical-threshold", type=float, default=0.85)
    args = parser.parse_args()
    fields = benchmark_fields(args.benchmark.resolve())
    token_index: dict[str, set[int]] = defaultdict(set)
    for index, field in enumerate(fields):
        for token in field["tokens"]:
            token_index[token].add(index)
    source_rows = load_jsonl(args.input.resolve())
    clean: list[Any] = []
    rejected: list[dict[str, Any]] = []
    for line_number, row in enumerate(source_rows, 1):
        match = row_match(row, fields, token_index, args.lexical_threshold)
        if match:
            rejected.append({"line": line_number, **match})
        else:
            clean.append(row)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as handle:
        for row in clean:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "format": "benchmark_contaminated_dataset_filter_v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.resolve().read_bytes()).hexdigest(),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_field_count": len(fields),
        "lexical_threshold": args.lexical_threshold,
        "input_count": len(source_rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "sealed_test_paths_read": [],
        "passed": True,
        "notes": ["Filtered rows are not used in the strict Spanish-data candidate training base."],
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"input_count": len(source_rows), "clean_count": len(clean), "rejected_count": len(rejected), "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
