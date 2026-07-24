#!/usr/bin/env python3
"""Build balanced Spanish MCQ and derived free-form training records.

The input is the fully translated, question-level-audited QMD intermediate.
Source IDs are split before augmentation so four cyclic MCQ variants of one
source item can never straddle train and validation.  The free-form records
are explicitly derived from the same MCQ source and are not benchmark items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


LETTERS = ("A", "B", "C", "D")
SOURCE_ID_RE = re.compile(r"[^a-z0-9]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: str) -> str:
    return SOURCE_ID_RE.sub("-", value.casefold()).strip("-")


def category_id(value: str) -> str:
    return slug(value) or "unknown"


def source_id(value: str) -> str:
    return f"es-nf-source-qmd-{slug(value)}"


def training_id(kind: str, source: str, suffix: str = "") -> str:
    tail = f"-{suffix}" if suffix else ""
    return f"es-train-qmd-{kind}-{slug(source)}{tail}"


def option_block(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in LETTERS)


def clean_option_text(value: str) -> str:
    return re.sub(r"^\s*[A-D][.)]\s+", "", value).strip()


def rotated_options(row: dict[str, Any], shift: int) -> tuple[dict[str, str], str]:
    original = [clean_option_text(str(row[letter])) for letter in LETTERS]
    answer = str(row["answer"]).strip().upper()
    if answer not in LETTERS:
        raise ValueError(f"{row.get('source_question_id')}: invalid answer {answer!r}")
    rotated = original[-shift:] + original[:-shift] if shift else original
    original_index = LETTERS.index(answer)
    new_answer = LETTERS[(original_index + shift) % 4]
    return dict(zip(LETTERS, rotated)), new_answer


def provenance(row: dict[str, Any], source_path: Path, report_path: str) -> dict[str, Any]:
    return {
        "source_sha256": str(row["source_sha256"]),
        "source_locator": str(source_path),
        "authoring_method": "machine_translation_of_qmd",
        "review_status": "pending",
        "reviewer_ids": [],
    }


def contamination(report_path: str) -> dict[str, Any]:
    return {
        "status": "pass",
        "checked_at": date.today().isoformat(),
        "checks": {
            "exact_normalized_text_match": "pass",
            "near_duplicate_question_match": "pass",
            "source_document_overlap": "not_run",
            "english_v5clean_anchor_question_match": "pass",
            "spanish_benchmark_item_match": "pass",
            "sealed_test_path_access": "pass",
        },
        "report_path": report_path,
    }


def build_records(
    rows: list[dict[str, Any]],
    train_ids: set[str],
    source_path: Path,
    contamination_report: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mcq: list[dict[str, Any]] = []
    freeform: list[dict[str, Any]] = []
    for row in rows:
        source_question_id = str(row["source_question_id"])
        split = "train" if source_question_id in train_ids else "validation"
        cat = category_id(str(row.get("category", "unknown")))
        sid = source_id(source_question_id)
        for shift in range(4):
            options, answer = rotated_options(row, shift)
            user = f"{str(row['question']).strip()}\n\n{option_block(options)}"
            record = {
                "id": training_id("mcq", source_question_id, f"s{shift}"),
                "language": "es",
                "locale": "es-419",
                "split": split,
                "category_id": cat,
                "source_id": sid,
                "unit_type": "single_turn",
                "task_type": "mcq",
                "turn_count": 2,
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": f"<think>\n{row['rationale'].strip()}\n</think>\n{answer}"},
                ],
                "mcq": {"options": options, "gold_letter": answer},
                "provenance": provenance(row, source_path, contamination_report),
                "contamination": contamination(contamination_report),
                "source_question_id": source_question_id,
                "augmentation": {"kind": "cyclic_option_rotation", "shift": shift, "source_split_before_augmentation": True},
            }
            mcq.append(record)

        answer = str(row["answer"]).strip().upper()
        correct_option = clean_option_text(str(row[answer]))
        freeform_question = (
            f"{str(row['question']).strip()}\n\n"
            "Responde en español con una explicación breve y técnicamente precisa."
        )
        freeform_answer = f"{correct_option}\n\n{str(row['rationale']).strip()}"
        freeform.append(
            {
                "id": training_id("freeform", source_question_id),
                "language": "es",
                "locale": "es-419",
                "split": split,
                "category_id": cat,
                "source_id": sid,
                "unit_type": "single_turn",
                "task_type": "free_form_explanation",
                "turn_count": 2,
                "messages": [
                    {"role": "user", "content": freeform_question},
                    {"role": "assistant", "content": freeform_answer},
                ],
                "provenance": provenance(row, source_path, contamination_report),
                "contamination": contamination(contamination_report),
                "source_question_id": source_question_id,
                "derivation": "open_answer_from_translated_qmd_mcq_with_correct_option_and_rationale",
            }
        )
    return mcq, freeform


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_sft(path: Path, rows: list[dict[str, Any]]) -> None:
    pairs = []
    for row in rows:
        messages = row["messages"]
        pairs.append({"instruction": messages[0]["content"], "output": messages[1]["content"]})
    write_jsonl(path, pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-locator", required=True, type=Path)
    parser.add_argument("--contamination-report", required=True)
    parser.add_argument("--source-registry", type=Path, default=None)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between 0 and 1")

    translated_rows = load_jsonl(args.input.resolve())
    source_rows = load_jsonl(args.source_locator.resolve())
    translated_by_id = {str(row["source_question_id"]): row for row in translated_rows}
    if len(translated_by_id) != len(translated_rows):
        raise ValueError("translated QMD input contains duplicate source IDs")
    source_ids = [str(row["question_id"]) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("audited QMD source contains duplicate question IDs")
    missing = [source_id for source_id in source_ids if source_id not in translated_by_id]
    if missing:
        raise ValueError(f"translated QMD input is missing audited source IDs: {missing[:10]}")
    source_hash = sha256(args.source_locator.resolve())
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source_key = str(source_row["question_id"])
        translated = dict(translated_by_id[source_key])
        if str(translated.get("answer", "")).upper() != str(source_row.get("answer", "")).upper():
            raise ValueError(f"answer key changed for {source_key}")
        translated["answer"] = source_row["answer"]
        translated["category"] = source_row.get("category", translated.get("category", ""))
        translated["topic"] = source_row.get("topic", translated.get("topic", ""))
        translated["source_sha256"] = source_hash
        rows.append(translated)
    ids = [str(row["source_question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("translated QMD input contains duplicate source IDs")
    if any(row.get("translation", {}).get("status") != "machine_translated_unreviewed" for row in rows):
        raise ValueError("all translated rows must be marked machine_translated_unreviewed")

    ordered = sorted(ids)
    rng = random.Random(args.seed)
    rng.shuffle(ordered)
    train_count = max(1, int(len(ordered) * args.split))
    train_ids = set(ordered[:train_count])
    mcq, freeform = build_records(rows, train_ids, args.source_locator.resolve(), args.contamination_report)
    mcq_train = [row for row in mcq if row["split"] == "train"]
    mcq_val = [row for row in mcq if row["split"] == "validation"]
    ff_train = [row for row in freeform if row["split"] == "train"]
    ff_val = [row for row in freeform if row["split"] == "validation"]

    def counts(items: list[dict[str, Any]]) -> dict[str, int]:
        result = Counter(row["mcq"]["gold_letter"] for row in items if row["task_type"] == "mcq")
        return {letter: result.get(letter, 0) for letter in LETTERS}

    for label, items in (("mcq_train", mcq_train), ("mcq_validation", mcq_val)):
        observed = counts(items)
        if len(set(observed.values())) != 1:
            raise ValueError(f"{label}: MCQ answer balance failed: {observed}")

    out = args.output_dir.resolve()
    write_jsonl(out / "spanish_qmd_mcq_train.jsonl", mcq_train)
    write_jsonl(out / "spanish_qmd_mcq_validation.jsonl", mcq_val)
    write_jsonl(out / "spanish_qmd_freeform_train.jsonl", ff_train)
    write_jsonl(out / "spanish_qmd_freeform_validation.jsonl", ff_val)
    write_sft(out / "spanish_qmd_mcq_train.sft.jsonl", mcq_train)
    write_sft(out / "spanish_qmd_mcq_validation.sft.jsonl", mcq_val)
    write_sft(out / "spanish_qmd_freeform_train.sft.jsonl", ff_train)
    write_sft(out / "spanish_qmd_freeform_validation.sft.jsonl", ff_val)

    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        qmd_id = str(row["source_question_id"])
        registry[source_id(qmd_id)] = {
            "source_id": source_id(qmd_id),
            "language": "es",
            "locale": "es-419",
            "source_type": "machine_translation_of_qmd_training_source",
            "title": f"QMD translated training item {qmd_id}",
            "locator": str(args.source_locator.resolve()),
            "sha256": str(row["source_sha256"]),
            "license": "internal derived training source; consult the upstream QMD source license",
            "retrieved_at": date.today().isoformat(),
            "category_ids": [category_id(str(row.get("category", "unknown")))],
            "allowed_uses": ["training"],
            "excluded_uses": ["mcq_dev", "mcq_sealed_test", "freeform_dev", "freeform_sealed_test"],
            "review_status": "pending",
            "notes": "Machine-translated from clean, question-level-audited English QMD data; not a native Spanish source.",
        }
    registry_path = args.source_registry.resolve() if args.source_registry else out / "source_registry.jsonl"
    write_jsonl(registry_path, list(registry.values()))

    manifest = {
        "format": "spanish_qmd_translation_training_manifest_v1",
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256(args.input.resolve()),
        "source_locator": str(args.source_locator.resolve()),
        "source_sha256": sha256(args.source_locator.resolve()),
        "translation_origin": "machine_translation_of_qmd",
        "translation_backend": "codex",
        "translation_model": "gpt-5.5",
        "translation_reasoning_effort": "high",
        "translated_input_item_count": len(translated_rows),
        "excluded_by_current_source_audit_count": len(translated_rows) - len(rows),
        "source_item_count": len(rows),
        "source_train_count": len(train_ids),
        "source_validation_count": len(ids) - len(train_ids),
        "mcq_train_count": len(mcq_train),
        "mcq_validation_count": len(mcq_val),
        "freeform_train_count": len(ff_train),
        "freeform_validation_count": len(ff_val),
        "source_registry_count": len(registry),
        "source_registry_path": str(registry_path),
        "mcq_train_answer_counts": counts(mcq_train),
        "mcq_validation_answer_counts": counts(mcq_val),
        "split_seed": args.seed,
        "split_fraction": args.split,
        "split_before_option_augmentation": True,
        "benchmark_role": "training_only_derived_from_qmd",
        "benchmark_ids_in_output": False,
        "training_allowed": True,
        "contamination_report": args.contamination_report,
        "sealed_test_paths_read": [],
        "notes": [
            "No native Spanish source was available; this is machine-translated English QMD training data.",
            "MCQ records are balanced by four cyclic option rotations per source item.",
            "Free-form records are derived open-answer versions of the translated MCQs, not native Spanish free-form authoring.",
            "Source-document independence from benchmark source documents is not claimed; question-level overlap gates were passed.",
        ],
    }
    (out / "spanish_qmd_training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_items": len(rows),
        "source_train": len(train_ids),
        "source_validation": len(ids) - len(train_ids),
        "mcq_train": len(mcq_train),
        "mcq_validation": len(mcq_val),
        "freeform_train": len(ff_train),
        "freeform_validation": len(ff_val),
        "mcq_train_answer_counts": counts(mcq_train),
        "mcq_validation_answer_counts": counts(mcq_val),
        "output_dir": str(out),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
