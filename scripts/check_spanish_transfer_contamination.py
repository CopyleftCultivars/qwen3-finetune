#!/usr/bin/env python3
"""Audit autoresearch JSONL pools for Spanish transfer-benchmark leakage.

The Spanish English-to-Spanish transfer set is evaluation-only. This audit
checks every JSONL file under the current ``data`` and ``eval_freeform`` trees
by exact normalized substring and high token-overlap matching. It does not
modify any dataset and exits nonzero on a match.
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
from typing import Any, Iterable

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MIN_EXACT_CHARS = 48
MIN_OVERLAP_TOKENS = 6
DEFAULT_BENCHMARK = Path(
    "../spanish-benchmark-workspace/benchmarks/transfer/en_to_es/"
    "english_v5clean_benchmark_es.jsonl"
)
DEFAULT_TARGET_ROOTS = (Path("data"), Path("eval_freeform"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize(text)))


def load_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, line


def string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from string_values(child)


def benchmark_fields(path: Path) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for line_number, row in load_jsonl(path):
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_question_id") or row.get("id") or line_number)
        for field in ("question", "A", "B", "C", "D"):
            value = row.get(field)
            if not isinstance(value, str) or len(normalize(value)) < MIN_EXACT_CHARS:
                continue
            norm = normalize(value)
            field_tokens = tokens(value)
            fields.append(
                {
                    "source_question_id": source_id,
                    "field": field,
                    "normalized": norm,
                    "tokens": field_tokens,
                }
            )
    return fields


def audit_target(
    path: Path,
    fields: list[dict[str, Any]],
    token_index: dict[str, set[int]],
    lexical_threshold: float,
) -> dict[str, Any]:
    exact_matches: list[dict[str, Any]] = []
    lexical_matches: list[dict[str, Any]] = []
    max_overlap = {"score": 0.0, "source_question_id": None, "field": None, "line": None}
    records = 0
    for line_number, row in load_jsonl(path):
        records += 1
        text_values = list(string_values(row))
        for text_value in text_values:
            normalized = normalize(text_value)
            row_tokens = tokens(text_value)
            if len(row_tokens) < MIN_OVERLAP_TOKENS:
                continue
            # Use the rarest words in this target string to keep candidate
            # generation bounded. Common Spanish stopwords and numerals can
            # occur in thousands of benchmark fields.
            ranked_tokens = sorted(
                row_tokens,
                key=lambda token: len(token_index.get(token, set())),
            )[:8]
            candidate_ids = set(
                candidate
                for token in ranked_tokens
                for candidate in token_index.get(token, set())
            )
            for index in candidate_ids:
                field = fields[index]
                if field["normalized"] in normalized:
                    exact_matches.append(
                        {
                            "line": line_number,
                            "source_question_id": field["source_question_id"],
                            "field": field["field"],
                            "reason": "exact_normalized_substring",
                        }
                    )
                    continue
                intersection = len(row_tokens & field["tokens"])
                union = len(row_tokens | field["tokens"])
                score = intersection / union if union else 0.0
                if score > max_overlap["score"]:
                    max_overlap = {
                        "score": score,
                        "source_question_id": field["source_question_id"],
                        "field": field["field"],
                        "line": line_number,
                    }
                if intersection >= MIN_OVERLAP_TOKENS and score >= lexical_threshold:
                    lexical_matches.append(
                        {
                            "line": line_number,
                            "source_question_id": field["source_question_id"],
                            "field": field["field"],
                            "reason": "high_token_jaccard",
                            "token_jaccard": score,
                        }
                    )
    return {
        "path": str(path),
        "records_scanned": records,
        "exact_match_count": len(exact_matches),
        "high_lexical_overlap_count": len(lexical_matches),
        "maximum_token_jaccard": max_overlap,
        "exact_matches": exact_matches[:20],
        "high_lexical_overlaps": lexical_matches[:20],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--target", type=Path, action="append", help="JSONL file or directory; repeatable")
    parser.add_argument("--lexical-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=Path("results/spanish_transfer_contamination_audit.json"))
    return parser


def target_files(args: argparse.Namespace) -> list[Path]:
    roots = args.target or list(DEFAULT_TARGET_ROOTS)
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".jsonl":
            paths.add(root)
        elif root.is_dir():
            paths.update(root.rglob("*.jsonl"))
    return sorted(path for path in paths if path.resolve() != args.benchmark.resolve())


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.lexical_threshold <= 1.0:
        parser.error("--lexical-threshold must be in (0, 1]")
    benchmark = args.benchmark.resolve()
    if not benchmark.is_file():
        parser.error(f"benchmark not found: {benchmark}")
    fields = benchmark_fields(benchmark)
    token_index: dict[str, set[int]] = defaultdict(set)
    for index, field in enumerate(fields):
        for token in field["tokens"]:
            token_index[token].add(index)
    audits = [audit_target(path.resolve(), fields, token_index, args.lexical_threshold) for path in target_files(args)]
    report = {
        "format": "spanish_transfer_contamination_audit_v1",
        "benchmark": str(benchmark),
        "benchmark_sha256": sha256(benchmark),
        "benchmark_field_count": len(fields),
        "lexical_threshold": args.lexical_threshold,
        "minimum_exact_chars": MIN_EXACT_CHARS,
        "minimum_overlap_tokens": MIN_OVERLAP_TOKENS,
        "targets": audits,
        "passed": all(
            item["exact_match_count"] == 0 and item["high_lexical_overlap_count"] == 0
            for item in audits
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "target_files": len(audits), "passed": report["passed"]}, indent=2))
    if not report["passed"]:
        print("Spanish transfer contamination detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
