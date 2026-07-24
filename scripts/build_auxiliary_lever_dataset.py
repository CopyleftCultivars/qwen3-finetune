#!/usr/bin/env python3
"""Compose one fixed-dose auxiliary task with the strict v5clean base.

This is the boundary between dataset authoring and the autoresearch harness.
It keeps the English v5clean train/validation pools intact and adds exactly one
audited auxiliary task.  Run it once per task; the resulting four manifests are
the individual CROP-style lever inputs for serial GPU experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = ("summarization", "ner", "structured_extraction", "open_book_qa")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if "instruction" not in value or "output" not in value:
                raise ValueError(f"{path}:{line_number}: missing instruction/output")
            rows.append(value)
    return rows


def assert_training_safe(path: Path, *, allowed_language: str | None = None) -> None:
    resolved = path.resolve()
    parts = {part.casefold() for part in resolved.parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError(f"evaluation-only Spanish path is forbidden: {path}")
    if "eval_freeform" in parts and resolved.name.casefold() == "test.jsonl":
        raise ValueError(f"sealed free-form test path is forbidden: {path}")
    if "sealed" in resolved.name.casefold():
        raise ValueError(f"sealed path is forbidden: {path}")
    if allowed_language is not None:
        for row in load(path):
            if row.get("language") != allowed_language:
                raise ValueError(f"{path}: expected language={allowed_language}")


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
    parser.add_argument("--auxiliary-task", required=True, choices=TASKS)
    parser.add_argument("--auxiliary-train", required=True, type=Path)
    parser.add_argument("--auxiliary-validation", required=True, type=Path)
    parser.add_argument("--auxiliary-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    for path in (args.base_train, args.base_validation, args.auxiliary_train, args.auxiliary_validation, args.auxiliary_manifest):
        if not path.exists():
            parser.error(f"file not found: {path}")
    for path in (args.base_train, args.base_validation):
        assert_training_safe(path)
    for path in (args.auxiliary_train, args.auxiliary_validation):
        assert_training_safe(path, allowed_language="en")

    base_train = load(args.base_train.resolve())
    base_validation = load(args.base_validation.resolve())
    auxiliary_train = load(args.auxiliary_train.resolve())
    auxiliary_validation = load(args.auxiliary_validation.resolve())
    auxiliary_manifest = json.loads(args.auxiliary_manifest.resolve().read_text(encoding="utf-8"))
    if auxiliary_manifest.get("task") != args.auxiliary_task:
        parser.error("auxiliary manifest task does not match --auxiliary-task")
    if auxiliary_manifest.get("training_allowed") is not True:
        parser.error("auxiliary manifest is not marked training_allowed")
    if auxiliary_manifest.get("train_validation_source_overlap"):
        parser.error("auxiliary train/validation source IDs overlap")
    for row in auxiliary_train + auxiliary_validation:
        if row.get("auxiliary_task") != args.auxiliary_task:
            parser.error("auxiliary row task does not match --auxiliary-task")
        if row.get("_contamination_status") != "passed":
            parser.error("auxiliary row is not contamination-audited")

    train = base_train + auxiliary_train
    validation = base_validation + auxiliary_validation
    output_root = args.output_root.resolve()
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"
    train_sha = write(train_path, train)
    validation_sha = write(validation_path, validation)
    manifest = {
        "format": "auxiliary_lever_composed_dataset_manifest_v1",
        "auxiliary_task": args.auxiliary_task,
        "base": {
            "train_path": str(args.base_train.resolve()),
            "train_sha256": digest(args.base_train.resolve()),
            "train_count": len(base_train),
            "validation_path": str(args.base_validation.resolve()),
            "validation_sha256": digest(args.base_validation.resolve()),
            "validation_count": len(base_validation),
        },
        "auxiliary": {
            "train_path": str(args.auxiliary_train.resolve()),
            "train_sha256": digest(args.auxiliary_train.resolve()),
            "train_count": len(auxiliary_train),
            "validation_path": str(args.auxiliary_validation.resolve()),
            "validation_sha256": digest(args.auxiliary_validation.resolve()),
            "validation_count": len(auxiliary_validation),
            "manifest_path": str(args.auxiliary_manifest.resolve()),
            "manifest_sha256": digest(args.auxiliary_manifest.resolve()),
        },
        "output": {
            "train_path": str(train_path),
            "train_sha256": train_sha,
            "train_count": len(train),
            "validation_path": str(validation_path),
            "validation_sha256": validation_sha,
            "validation_count": len(validation),
        },
        "auxiliary_train_fraction_of_composed_train": len(auxiliary_train) / len(train),
        "auxiliary_validation_fraction_of_composed_validation": len(auxiliary_validation) / len(validation),
        "sealed_test_paths_read": [],
        "notes": [
            "The strict English v5clean base is unchanged; this manifest adds one auxiliary task only.",
            "The four task manifests use the same fixed pilot dose for causal comparison.",
            "No Spanish benchmark or free-form sealed test path is included.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "auxiliary_task": args.auxiliary_task,
        "base_train": len(base_train),
        "auxiliary_train": len(auxiliary_train),
        "composed_train": len(train),
        "base_validation": len(base_validation),
        "auxiliary_validation": len(auxiliary_validation),
        "composed_validation": len(validation),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
