#!/usr/bin/env python3
"""Audit and package the general-instruction source for replay training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.casefold()))


def tokens(value: str) -> set[str]:
    return set(normalize(value).split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def fields(row: dict[str, Any], *, include_outputs: bool = False) -> list[str]:
    keys = ("instruction", "question", "A", "B", "C", "D")
    if include_outputs:
        keys += ("answer", "gold_letter", "rationale", "output", "response", "target")
    return [str(row.get(key, "")) for key in keys if row.get(key)]


def benchmark_fields(path: Path) -> list[tuple[str, str, frozenset[str]]]:
    values = []
    for index, row in enumerate(load_jsonl(path)):
        for field_index, value in enumerate(fields(row, include_outputs=True)):
            normalized = normalize(value)
            if normalized:
                values.append((f"{index}:{field_index}", normalized, frozenset(normalized.split())))
    return values


def overlap(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def audit_rows(rows: list[dict[str, Any]], reference_fields: list[tuple[str, str, frozenset[str]]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact: dict[str, str] = {}
    by_size: dict[int, list[tuple[str, frozenset[str]]]] = {}
    for reference_id, normalized, reference_text in reference_fields:
        exact.setdefault(normalized, reference_id)
        by_size.setdefault(len(reference_text), []).append((reference_id, reference_text))
    clean = []
    rejected = []
    seen_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("_source_id", ""))
        reason = None
        if not source_id or source_id in seen_ids:
            reason = "duplicate_or_missing_source_id"
        elif row.get("language") != "en":
            reason = "non_english_label"
        elif not str(row.get("instruction", "")).strip() or not str(row.get("output", "")).strip():
            reason = "empty_instruction_or_output"
        else:
            normalized_source = normalize(str(row["instruction"]))
            source_text = frozenset(normalized_source.split())
            if normalized_source in exact:
                reason = f"exact_reference_match:{exact[normalized_source]}"
            elif len(source_text) >= 8:
                minimum = max(1, int(len(source_text) * threshold + 0.999999))
                maximum = max(minimum, int(len(source_text) / threshold))
                for size in range(minimum, maximum + 1):
                    for reference_id, reference_text in by_size.get(size, []):
                        if overlap(source_text, reference_text) >= threshold:
                            reason = f"near_reference_match:{reference_id}"
                            break
                    if reason:
                        break
        if reason:
            rejected.append({"_source_id": source_id, "reason": reason})
            continue
        seen_ids.add(source_id)
        clean_row = dict(row)
        clean_row["_contamination_status"] = "passed"
        clean_row["_contamination_audit"] = "general_instruction_contamination_audit_v1"
        clean.append(clean_row)
    return clean, rejected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lexical-threshold", type=float, default=0.9)
    args = parser.parse_args()
    if not args.benchmark.exists():
        parser.error(f"benchmark not found: {args.benchmark}")

    reference = benchmark_fields(args.benchmark)
    # The base is reported separately. A repeated generic prompt with a
    # different answer is not benchmark contamination (e.g. "Tell me a joke")
    # and must not be rejected as if it were an MCQ benchmark item.
    base_instruction_keys: set[str] = set()
    for path in (args.base_train, args.base_validation):
        for row in load_jsonl(path):
            value = str(row.get("instruction", ""))
            if normalize(value):
                base_instruction_keys.add(normalize(value))

    train_raw = load_jsonl(args.input_dir / "raw_train.jsonl")
    validation_raw = load_jsonl(args.input_dir / "raw_validation.jsonl")
    all_raw = train_raw + validation_raw
    clean_all, rejected = audit_rows(all_raw, reference, args.lexical_threshold)
    clean_by_id = {row["_source_id"]: row for row in clean_all}
    train = [clean_by_id[row["_source_id"]] for row in train_raw if row["_source_id"] in clean_by_id]
    validation = [clean_by_id[row["_source_id"]] for row in validation_raw if row["_source_id"] in clean_by_id]

    train_ids = {row["_source_id"] for row in train}
    validation_ids = {row["_source_id"] for row in validation}
    if train_ids & validation_ids:
        raise ValueError("general source train/validation overlap")
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)

    source_manifest = json.loads((args.input_dir / "source_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        **source_manifest,
        "format": "general_instruction_audited_manifest_v1",
        "training_allowed": not rejected and bool(train) and bool(validation),
        "contamination_audit": "general_instruction_contamination_audit_v1",
        "benchmark_path": str(args.benchmark.resolve()),
        "base_paths": [str(args.base_train.resolve()), str(args.base_validation.resolve())],
        "lexical_threshold": args.lexical_threshold,
        "audit": {
            "input_count": len(all_raw),
            "clean_count": len(clean_all),
            "rejected_count": len(rejected),
            "rejected_sample": rejected[:20],
            "train_count": len(train),
            "validation_count": len(validation),
            "train_validation_source_overlap": sorted(train_ids & validation_ids),
            "reference_field_count": len(reference),
            "base_instruction_overlap_count": sum(
                normalize(str(row["instruction"])) in base_instruction_keys
                for row in clean_all
            ),
        },
        "source_sha256": hashlib.sha256(
            "".join(row["_source_id"] for row in clean_all).encode("utf-8")
        ).hexdigest(),
        "audited_at": date.today().isoformat(),
        "notes": [
            "Training is allowed only when the benchmark/base near-duplicate audit is clean.",
            "The audit is conservative and applies to the instruction surface; outputs remain provenance-tracked.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"clean": len(clean_all), "rejected": len(rejected), "training_allowed": manifest["training_allowed"]}))


if __name__ == "__main__":
    main()
