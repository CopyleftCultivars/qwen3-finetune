#!/usr/bin/env python3
"""Compose one proposed-answer verification stream with the strict v5clean base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = ("proposed_answer_audit",)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    normalized = path.resolve().as_posix().casefold()
    if "spanish-benchmark-workspace" in parts:
        raise ValueError(f"Spanish benchmark path is evaluation-only: {path}")
    if "sealed" in parts or "sealed" in path.name.casefold():
        raise ValueError(f"sealed path is forbidden: {path}")
    if "/eval_freeform/test.jsonl" in normalized:
        raise ValueError(f"sealed free-form test path is forbidden: {path}")


def load(path: Path) -> list[dict[str, Any]]:
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


def write(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return digest(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--base-validation", required=True, type=Path)
    parser.add_argument("--verification-task", required=True, choices=TASKS)
    parser.add_argument("--source-train", required=True, type=Path)
    parser.add_argument("--source-validation", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.base_train, args.base_validation, args.source_train, args.source_validation, args.source_manifest):
        if not path.exists():
            parser.error(f"file not found: {path}")
        reject_path(path)
    base_train = load(args.base_train.resolve())
    base_validation = load(args.base_validation.resolve())
    source_train = load(args.source_train.resolve())
    source_validation = load(args.source_validation.resolve())
    manifest = json.loads(args.source_manifest.resolve().read_text(encoding="utf-8"))
    if manifest.get("task") != args.verification_task:
        parser.error("source manifest task does not match --verification-task")
    if manifest.get("training_allowed") is not True:
        parser.error("source manifest is not marked training_allowed")
    if manifest.get("source_train_validation_overlap"):
        parser.error("source train/validation overlap is non-empty")
    for row in source_train + source_validation:
        if row.get("verification_task") != args.verification_task:
            parser.error("source row verification_task does not match selected task")
        if row.get("language") != "en":
            parser.error("source row language must be en")
        if row.get("_contamination_status") != "passed":
            parser.error("source row did not pass contamination audit")
    train = base_train + source_train
    validation = base_validation + source_validation
    output_root = args.output_root.resolve()
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"
    train_sha = write(train_path, train)
    validation_sha = write(validation_path, validation)
    composed = {
        "format": "verification_lever_composed_dataset_manifest_v1",
        "verification_task": args.verification_task,
        "base": {
            "train_path": str(args.base_train.resolve()),
            "train_sha256": digest(args.base_train.resolve()),
            "train_count": len(base_train),
            "validation_path": str(args.base_validation.resolve()),
            "validation_sha256": digest(args.base_validation.resolve()),
            "validation_count": len(base_validation),
        },
        "source": {
            "train_path": str(args.source_train.resolve()),
            "train_sha256": digest(args.source_train.resolve()),
            "train_count": len(source_train),
            "validation_path": str(args.source_validation.resolve()),
            "validation_sha256": digest(args.source_validation.resolve()),
            "validation_count": len(source_validation),
            "manifest_path": str(args.source_manifest.resolve()),
            "manifest_sha256": digest(args.source_manifest.resolve()),
        },
        "output": {
            "train_path": str(train_path),
            "train_sha256": train_sha,
            "train_count": len(train),
            "validation_path": str(validation_path),
            "validation_sha256": validation_sha,
            "validation_count": len(validation),
        },
        "source_train_fraction": len(source_train) / len(train),
        "source_validation_fraction": len(source_validation) / len(validation),
        "sealed_test_paths_read": [],
        "notes": [
            "The strict English v5clean base is unchanged.",
            "This composition contains exactly one verification stream.",
            "No prior reflection, rationale, auxiliary, general-replay, or Spanish stream is included.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(composed, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verification_task": args.verification_task, "base_train": len(base_train), "source_train": len(source_train), "composed_train": len(train), "base_validation": len(base_validation), "source_validation": len(source_validation), "composed_validation": len(validation)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
