#!/usr/bin/env python3
"""Audit and package a clean MCQ+rationale JSONL source.

The audit is deliberately question-level.  It catches exact normalized stem
reuse and a high lexical-overlap floor against the benchmark, validates the
``<think>...</think>`` plus bare-letter target shape, removes known embedded
persona prefixes, and emits provenance fields that prepare_dataset_v3.py
requires for an opt-in rationale source.

This does not claim semantic independence from the benchmark's source
documents.  It proves the narrower training/evaluation boundary used by this
project: no benchmark question-text collision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


OPTION_RE = re.compile(
    r"^(.+?)\n\nA\. (.+)\nB\. (.+)\nC\. (.+)\nD\. (.+)$",
    re.DOTALL,
)
TARGET_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*([A-D])\s*$", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")
SYSTEM_END = "Respond in user's language unless otherwise specified."
SYSTEM_MARKERS = (
    "You are a Natural Farming",
    "You are an AI assistant",
    "You are a helpful",
)
CATEGORY_BY_PREFIX = {
    "AG": "agroecology",
    "CONV": "conversations",
    "IFS": "integrated_farming_systems",
    "ITK": "indigenous_and_traditional_knowledge",
    "NF": "natural_fertilizers",
    "OF": "organic_farming",
    "PC": "permaculture",
    "RA": "regenerative_agriculture",
    "SS": "soil_science_and_microbiology",
    "SWM": "sustainable_water_management",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_RE.findall(text))


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(normalize(text)))


def _strip_system_prefix(instruction: str) -> tuple[str, bool]:
    stripped = instruction.strip()
    if not any(stripped.startswith(marker) for marker in SYSTEM_MARKERS):
        return stripped, False
    boundary = stripped.find(SYSTEM_END)
    if boundary < 0:
        return "", True
    suffix = stripped[boundary + len(SYSTEM_END):].strip()
    return suffix, True


def _stem(instruction: str) -> str | None:
    match = OPTION_RE.match(instruction.strip())
    return match.group(1).strip() if match else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(value)
    return rows


def _benchmark_stems(path: Path) -> dict[str, str]:
    stems: dict[str, str] = {}
    for index, row in enumerate(_load_jsonl(path)):
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        stems.setdefault(normalize(question), str(row.get("question_id", index)))
    return stems


def _best_token_jaccard(left: set[str], benchmark_sets: list[tuple[str, set[str]]]) -> tuple[float, str]:
    best_score = 0.0
    best_id = ""
    for question_id, right in benchmark_sets:
        union = left | right
        score = len(left & right) / len(union) if union else 0.0
        if score > best_score:
            best_score, best_id = score, question_id
    return best_score, best_id


def audit_rows(
    source_rows: list[dict[str, Any]],
    benchmark_stems: dict[str, str],
    *,
    source_path: Path,
    benchmark_path: Path,
    lexical_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_sets = [
        (question_id, set(stem.split()))
        for stem, question_id in benchmark_stems.items()
    ]
    seen_ids: set[str] = set()
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    dropped_system = 0
    invalid = 0
    max_lexical = {"score": 0.0, "source_question_id": None, "benchmark_question_id": None}
    source_categories: Counter[str] = Counter()

    for index, row in enumerate(source_rows):
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", ""))
        source_id = str(row.get("_source_question_id", row.get("question_id", "")))
        instruction, had_system = _strip_system_prefix(instruction)
        if had_system:
            dropped_system += 1
        stem = _stem(instruction)
        target = TARGET_RE.search(output)
        reason: str | None = None
        if not source_id:
            reason = "missing_source_question_id"
        elif source_id in seen_ids:
            reason = "duplicate_source_question_id"
        elif stem is None or target is None or not target.group(1).strip():
            reason = "invalid_mcq_rationale_target"
        elif not instruction:
            reason = "system_prompt_only"
        else:
            normalized = normalize(stem)
            if normalized in benchmark_stems:
                reason = "exact_benchmark_question_overlap"
            else:
                score, benchmark_id = _best_token_jaccard(
                    set(normalized.split()), benchmark_sets
                )
                if score > max_lexical["score"]:
                    max_lexical = {
                        "score": round(score, 6),
                        "source_question_id": source_id,
                        "benchmark_question_id": benchmark_id,
                    }
                if score >= lexical_threshold:
                    reason = "high_lexical_benchmark_overlap"
        if reason:
            rejected.append(
                {
                    "source_index": index,
                    "source_question_id": source_id or None,
                    "reason": reason,
                    "question": stem,
                    "benchmark_question_id": benchmark_stems.get(normalize(stem or "")),
                }
            )
            invalid += 1
            continue

        seen_ids.add(source_id)
        prefix = source_id.split("-", 1)[0]
        category = CATEGORY_BY_PREFIX.get(prefix, "unknown")
        source_categories[category] += 1
        clean_row = {
                "instruction": instruction,
                "output": output,
                "_source": "mcq_rationale_clean",
                "_source_question_id": source_id,
                "_source_category": category,
                "_contamination_status": "passed",
                "_rationale_status": "validated_think_target",
                "_rationale_audit": "mcq_rationale_contamination_audit_v1",
            }
        # Preserve authoring provenance for opt-in experimental sources.  The
        # canonical fields above remain normalized for prepare_dataset_v3.py;
        # these extra fields make it possible to distinguish novel authoring
        # from historical QMD replay after the contamination audit.
        for key in ("_rationale_origin", "_rationale_replay_comparison"):
            if key in row:
                clean_row[key] = row[key]
        clean.append(clean_row)

    report = {
        "format": "mcq_rationale_contamination_audit_v1",
        "source": str(source_path),
        "benchmark": str(benchmark_path),
        "source_count": len(source_rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "dropped_system_prompt_count": dropped_system,
        "benchmark_question_count": len(benchmark_stems),
        "exact_question_overlap_count": sum(
            item["reason"] == "exact_benchmark_question_overlap" for item in rejected
        ),
        "high_lexical_overlap_count": sum(
            item["reason"] == "high_lexical_benchmark_overlap" for item in rejected
        ),
        "lexical_overlap_threshold": lexical_threshold,
        "maximum_nonmatching_token_jaccard": max_lexical,
        "source_categories": dict(sorted(source_categories.items())),
        "rejected": rejected,
        "passed": not rejected,
        "sealed_test_paths_read": [],
        "notes": [
            "Question-level audit; semantic similarity is not claimed absent.",
            "Known system/persona prefixes are removed before validation and packaging.",
            "The emitted source is opt-in and must be passed with --mcq-rationale.",
        ],
    }
    return clean, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lexical-threshold", type=float, default=0.85)
    args = parser.parse_args()

    for path in (args.input, args.benchmark):
        if not path.exists():
            parser.error(f"file not found: {path}")
    if not 0.0 < args.lexical_threshold <= 1.0:
        parser.error("--lexical-threshold must be in (0, 1]")

    clean, report = audit_rows(
        _load_jsonl(args.input),
        _benchmark_stems(args.benchmark),
        source_path=args.input,
        benchmark_path=args.benchmark,
        lexical_threshold=args.lexical_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in clean:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "source_count", "clean_count", "rejected_count",
        "dropped_system_prompt_count", "exact_question_overlap_count",
        "high_lexical_overlap_count", "source_categories", "passed",
    )}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
