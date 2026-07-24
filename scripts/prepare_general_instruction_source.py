#!/usr/bin/env python3
"""Materialize a pinned general-instruction source for replay experiments.

The first source is the English Databricks Dolly 15k dataset.  This command
does not make the source training-eligible by itself: it writes a raw,
provenance-bearing split that must pass ``audit_general_instruction_source``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


DATASET_ID = "databricks/databricks-dolly-15k"
DATASET_REVISION = "bdd27f4"
DATASET_LICENSE = "CC BY-SA 3.0"
DATASET_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k"


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _text(value: Any, field: str, source_id: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source_id}: {field} must be a string")
    value = " ".join(value.split())
    if not value:
        raise ValueError(f"{source_id}: {field} is empty")
    return value


def normalize_record(row: dict[str, Any], index: int, *, dataset_id: str, revision: str) -> dict[str, Any]:
    source_id = str(row.get("_id") or row.get("id") or f"dolly15k:{index:05d}")
    instruction = _text(row.get("instruction"), "instruction", source_id)
    response = _text(row.get("response"), "response", source_id)
    context = row.get("context", "")
    if not isinstance(context, str):
        raise ValueError(f"{source_id}: context must be a string")
    context = "\n\nContext:\n" + context.strip() if context.strip() else ""
    return {
        "instruction": instruction + context,
        "output": response,
        "language": "en",
        "category": _text(row.get("category", "general"), "category", source_id),
        "_source": "general_instruction",
        "_source_id": source_id,
        "_source_dataset": dataset_id,
        "_source_revision": revision,
        "_contamination_status": "unverified",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_id, split="train", revision=args.revision)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        rows.append(normalize_record(row, index, dataset_id=args.dataset_id, revision=args.revision))
    if len({row["_source_id"] for row in rows}) != len(rows):
        raise ValueError("source IDs are not unique")

    ordered = sorted(rows, key=lambda row: hashlib.sha256(row["_source_id"].encode()).hexdigest())
    validation_count = max(1, round(len(ordered) * args.validation_fraction))
    validation_ids = {row["_source_id"] for row in ordered[:validation_count]}
    validation = [row for row in rows if row["_source_id"] in validation_ids]
    train = [row for row in rows if row["_source_id"] not in validation_ids]
    _write_jsonl(args.output_dir / "raw_train.jsonl", train)
    _write_jsonl(args.output_dir / "raw_validation.jsonl", validation)
    manifest = {
        "format": "general_instruction_source_manifest_v1",
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "license": DATASET_LICENSE,
        "source_url": DATASET_URL,
        "retrieved_at": date.today().isoformat(),
        "language": "en",
        "source_split": "train",
        "total_count": len(rows),
        "train_count": len(train),
        "validation_count": len(validation),
        "split_seed": "sha256_source_id_v1",
        "training_allowed": False,
        "contamination_audit": "pending",
        "notes": [
            "English general-purpose instruction/response records; not natural-farming or QMD data.",
            "This raw source must pass the repository contamination audit before training.",
            "The source is split before any replay-ratio sampling.",
        ],
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"total": len(rows), "train": len(train), "validation": len(validation)}))


if __name__ == "__main__":
    main()
