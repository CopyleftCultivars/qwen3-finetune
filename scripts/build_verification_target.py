#!/usr/bin/env python3
"""Build the proposed-answer verification supervision source.

Each audited English QMD MCQ produces four records: the model receives one
proposed option and must accept it when correct or reject it while returning
the verified answer. This is a verification task, not rationale replay or
reflection-data replay. The final answer remains a bare letter, while the
verification ledger is a distinct supervision surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LETTERS = ("A", "B", "C", "D")
TASK = "proposed_answer_audit"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.casefold()))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def reject_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    normalized = path.resolve().as_posix().casefold()
    if "spanish-benchmark-workspace" in parts:
        raise ValueError(f"Spanish benchmark path is evaluation-only: {path}")
    if "sealed" in parts or "sealed" in path.name.casefold():
        raise ValueError(f"sealed path is forbidden: {path}")
    if "/eval_freeform/test.jsonl" in normalized:
        raise ValueError(f"sealed free-form test path is forbidden: {path}")


def option_block(row: dict[str, Any]) -> str:
    return "\n".join(f"{letter}. {str(row[letter]).strip()}" for letter in LETTERS)


def build_instruction(row: dict[str, Any], proposed: str) -> str:
    evidence = str(row.get("source_passage", "")).strip()
    evidence_block = f"\n\nEvidence excerpt:\n{evidence[:1800]}" if evidence else ""
    return (
        "Audit the proposed answer to this natural-farming multiple-choice question. "
        "Do not guess from the proposal alone: compare it with the question, all options, "
        "and the evidence excerpt. Return the verified answer letter in the final line.\n\n"
        f"Question:\n{str(row['question']).strip()}\n\n"
        f"Options:\n{option_block(row)}\n\n"
        f"Proposed answer: {proposed}{evidence_block}"
    )


def build_output(row: dict[str, Any], proposed: str) -> str:
    answer = str(row["answer"]).strip().upper()
    verdict = "ACCEPT" if proposed == answer else "REJECT"
    correct_text = str(row[answer]).strip()
    return (
        "<verification>\n"
        f"proposed_answer: {proposed}\n"
        f"verdict: {verdict}\n"
        f"verified_answer: {answer}\n"
        f"verification_basis: The verified option states: {correct_text}\n"
        "</verification>\n"
        f"{answer}"
    )


def make_record(row: dict[str, Any], proposed: str, split: str) -> dict[str, Any]:
    source_id = str(row["question_id"])
    answer = str(row["answer"]).strip().upper()
    return {
        "id": f"verify_{source_id}_{proposed}",
        "instruction": build_instruction(row, proposed),
        "output": build_output(row, proposed),
        "verification_task": TASK,
        "language": "en",
        "split": split,
        "source_id": source_id,
        "category": row.get("category"),
        "topic": row.get("topic"),
        "proposed_answer": proposed,
        "expected_verdict": "ACCEPT" if proposed == answer else "REJECT",
        "correct_answer": answer,
        "_contamination_status": "passed",
        "_contamination_audit": "verification_target_contamination_audit_v1",
        "_authoring_method": "deterministic_proposed_answer_verification_v1",
    }


def audit_source(
    rows: list[dict[str, Any]], benchmark: list[dict[str, Any]], base_train: list[dict[str, Any]], base_validation: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_questions = {normalize(str(row.get("question", ""))) for row in benchmark}
    benchmark_full = {
        normalize("\n".join(str(row.get(key, "")) for key in ("question", *LETTERS)))
        for row in benchmark
    }
    base_instructions = {normalize(str(row.get("instruction", ""))) for row in base_train + base_validation}
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        source_id = str(row.get("question_id", ""))
        reason = None
        if not source_id or source_id in seen:
            reason = "duplicate_or_missing_question_id"
        elif str(row.get("answer", "")).upper() not in LETTERS:
            reason = "invalid_answer_letter"
        elif any(not str(row.get(letter, "")).strip() for letter in ("question", *LETTERS)):
            reason = "missing_question_or_option"
        elif str(row.get("judge_verdict", "keep")) != "keep":
            reason = "source_not_judge_approved"
        else:
            question_key = normalize(str(row["question"]))
            full_key = normalize("\n".join(str(row.get(key, "")) for key in ("question", *LETTERS)))
            if question_key in benchmark_questions or full_key in benchmark_full:
                reason = "exact_english_benchmark_overlap"
        if reason:
            rejected.append({"source_id": source_id, "reason": reason})
            continue
        seen.add(source_id)
        clean.append(row)
    base_overlap = sum(
        normalize(str(row["question"])) in base_instructions
        for row in clean
    )
    report = {
        "input_count": len(rows),
        "clean_count": len(clean),
        "rejected_count": len(rejected),
        "rejected_sample": rejected[:20],
        "base_instruction_exact_overlap_count": base_overlap,
        "benchmark_exact_overlap_count": len(rejected),
    }
    return clean, report


def split_source(rows: list[dict[str, Any]], split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0 and 1")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['question_id']}".encode()).hexdigest(),
    )
    train_count = int(len(ordered) * split)
    train_ids = {str(row["question_id"]) for row in ordered[:train_count]}
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in ordered:
        target = train if str(row["question_id"]) in train_ids else validation
        split_name = "train" if target is train else "validation"
        target.extend(make_record(row, proposed, split_name) for proposed in LETTERS)
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--base-validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.source, args.benchmark, args.base_train, args.base_validation):
        if not path.exists():
            parser.error(f"file not found: {path}")
        reject_path(path)

    raw = load_jsonl(args.source.resolve())
    benchmark = load_jsonl(args.benchmark.resolve())
    base_train = load_jsonl(args.base_train.resolve())
    base_validation = load_jsonl(args.base_validation.resolve())
    clean, audit = audit_source(raw, benchmark, base_train, base_validation)
    if not clean or audit["rejected_count"]:
        raise SystemExit(f"verification source audit failed: {json.dumps(audit, sort_keys=True)}")
    source_train, source_validation = split_source(clean, args.split, args.seed)
    output_dir = args.output_dir.resolve()
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    train_sha = write_jsonl(train_path, source_train)
    validation_sha = write_jsonl(validation_path, source_validation)
    manifest = {
        "format": "verification_target_source_manifest_v1",
        "task": TASK,
        "language": "en",
        "training_allowed": True,
        "source": str(args.source.resolve()),
        "source_sha256": digest(args.source.resolve()),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": digest(args.benchmark.resolve()),
        "base_paths": [str(args.base_train.resolve()), str(args.base_validation.resolve())],
        "split": args.split,
        "seed": args.seed,
        "audit": audit,
        "source_question_count": len(clean),
        "source_train_question_count": len(source_train) // 4,
        "source_validation_question_count": len(source_validation) // 4,
        "source_train_count": len(source_train),
        "source_validation_count": len(source_validation),
        "source_train_sha256": train_sha,
        "source_validation_sha256": validation_sha,
        "proposed_answer_distribution": dict(sorted(Counter(row["proposed_answer"] for row in source_train + source_validation).items())),
        "verdict_distribution": dict(sorted(Counter(row["expected_verdict"] for row in source_train + source_validation).items())),
        "source_train_validation_overlap": [],
        "sealed_test_paths_read": [],
        "notes": [
            "This is a proposed-answer verification task, not QMD rationale replay or reflection replay.",
            "Each source question emits exactly one proposed-answer record for A, B, C, and D.",
            "The final answer remains a bare letter to preserve the settled MCQ interface.",
            "Benchmark exact question and full option-block overlaps are rejected.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"task": TASK, "source_questions": len(clean), "train_rows": len(source_train), "validation_rows": len(source_validation), "audit": audit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
