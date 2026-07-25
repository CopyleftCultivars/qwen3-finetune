#!/usr/bin/env python3
"""Freeze judge-kept MCQs into a balanced, provenance-tracked train/val source.

The source is deliberately separate from the strict v5clean prepared dataset.
Each kept item is rendered into four cyclic option rotations (one per target
letter), so this new stream is exactly balanced even though the available
high-quality source items have a skewed original answer distribution.
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


LETTERS = ("A", "B", "C", "D")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def benchmark_questions(path: Path) -> set[str]:
    values: set[str] = set()
    for row in load(path):
        if row.get("question"):
            values.add(normalize(row["question"]))
        instruction = str(row.get("instruction", ""))
        if instruction:
            values.add(normalize(instruction.split("\n\nA.", 1)[0]))
    return values


def render_instruction(question: str, options: dict[str, str]) -> str:
    block = "\n".join(f"{letter}. {options[letter]}" for letter in LETTERS)
    return f"{question.strip()}\n\n{block}"


def rotate(row: dict[str, Any], shift: int) -> dict[str, Any]:
    original = [str(row[letter]).strip() for letter in LETTERS]
    original_answer = LETTERS.index(str(row["answer"]).upper())
    shifted = original[-shift:] + original[:-shift] if shift else original
    new_answer = LETTERS[(original_answer + shift) % 4]
    correct_text = shifted[LETTERS.index(new_answer)]
    source_passage = str(row.get("source_passage", "")).strip()
    evidence = source_passage[:1200]
    rationale = (
        f"The source-backed answer is {correct_text}."
        + (f" The supporting passage states: {evidence}" if evidence else "")
        + " The other options are plausible alternatives but do not match the supplied evidence."
    )
    result = {
        "instruction": render_instruction(
            str(row["question"]),
            {letter: shifted[index] for index, letter in enumerate(LETTERS)},
        ),
        "output": f"<think>\n{rationale}\n</think>\n{new_answer}",
        "_source": "qmd_mcq_distractor_quality",
        "_source_question_id": str(row["question_id"]),
        "_source_rotation": shift,
        "_contamination_status": "passed",
        "_contamination_audit": "distractor_quality_source_audit_v1",
        "distractor_quality_task": "mcq_distractor_quality",
        "judge_verdict": "keep",
        "judge_model": str(row.get("judge_model", "")),
        "quality_composite": row.get("quality_scores", {}).get("composite"),
        "language": "en",
    }
    return result


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["question_id"]).encode("utf-8")
        # Stable 80/20 item-level split; all rotations stay together.
        if int(hashlib.sha256(key).hexdigest()[:8], 16) % 10 < 8:
            train.append(row)
        else:
            validation.append(row)
    if not train or not validation:
        raise ValueError("deterministic split produced an empty train or validation set")
    return train, validation


def write(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest(path)


def letter_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        match = re.search(r"</think>\s*([ABCD])\s*$", str(row["output"]))
        if not match:
            raise ValueError("formatted row has no final answer letter")
        counts[match.group(1)] += 1
    return {letter: counts[letter] for letter in LETTERS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-validation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    rows = load(args.judged)
    benchmark = benchmark_questions(args.benchmark)
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if audit.get("training_allowed") is not True:
        parser.error("source audit is not training_allowed")

    keeps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("quality_scores", {}).get("verdict") != "keep":
            continue
        question_id = str(row.get("question_id", "")).strip()
        if not question_id or question_id in seen:
            parser.error(f"missing or duplicate keep question_id: {question_id!r}")
        if row.get("_contamination_status") != "passed":
            parser.error(f"keep row is not contamination-audited: {question_id}")
        if row.get("language") not in (None, "en"):
            parser.error(f"non-English keep row: {question_id}")
        if normalize(str(row.get("question", ""))) in benchmark:
            parser.error(f"benchmark overlap in keep row: {question_id}")
        if any(not str(row.get(letter, "")).strip() for letter in LETTERS):
            parser.error(f"incomplete keep row: {question_id}")
        seen.add(question_id)
        keeps.append(row)

    keeps.sort(key=lambda row: str(row["question_id"]))
    if len(keeps) < 20:
        parser.error(f"quality keep pool is too small: {len(keeps)}")
    source_train, source_validation = split_rows(keeps)
    train = [rotated for row in source_train for rotated in (rotate(row, shift) for shift in range(4))]
    validation = [rotated for row in source_validation for rotated in (rotate(row, shift) for shift in range(4))]

    train_ids = {row["_source_question_id"] for row in train}
    validation_ids = {row["_source_question_id"] for row in validation}
    if train_ids & validation_ids:
        raise AssertionError("source train/validation question IDs overlap")
    train_sha = write(args.output_train, train)
    validation_sha = write(args.output_validation, validation)
    manifest = {
        "format": "distractor_quality_source_manifest_v1",
        "task": "mcq_distractor_quality",
        "training_allowed": True,
        "language": "en",
        "quality_gate": {
            "judge_verdict": "keep",
            "minimum_composite": 4.0,
            "judge_model": "anthropic/claude-haiku-4.5",
            "judged_count": len(rows),
            "kept_count": len(keeps),
        },
        "source_answer_counts": {letter: sum(row["answer"] == letter for row in keeps) for letter in LETTERS},
        "source_train_question_count": len(source_train),
        "source_validation_question_count": len(source_validation),
        "source_train_validation_overlap": [],
        "expanded_train_count": len(train),
        "expanded_validation_count": len(validation),
        "expanded_train_letter_counts": letter_counts(train),
        "expanded_validation_letter_counts": letter_counts(validation),
        "train_path": str(args.output_train.resolve()),
        "train_sha256": train_sha,
        "validation_path": str(args.output_validation.resolve()),
        "validation_sha256": validation_sha,
        "judged_path": str(args.judged.resolve()),
        "judged_sha256": digest(args.judged),
        "benchmark_path": str(args.benchmark.resolve()),
        "audit_report_path": str(args.audit_report.resolve()),
        "audit_report_sha256": digest(args.audit_report),
        "sealed_test_paths_read": [],
        "notes": [
            "Only OpenRouter-judge keep rows are included; review and reject rows are excluded.",
            "Each source question receives four equal cyclic rotations, making this added stream exactly A/B/C/D balanced.",
            "The strict English v5clean base remains a separate, unchanged input to the composed training dataset.",
            "No Spanish benchmark or sealed evaluation path was read.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "kept_questions": len(keeps),
        "source_train_questions": len(source_train),
        "source_validation_questions": len(source_validation),
        "expanded_train": len(train),
        "expanded_validation": len(validation),
        "train_letters": manifest["expanded_train_letter_counts"],
        "validation_letters": manifest["expanded_validation_letter_counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
