#!/usr/bin/env python3
"""Build serial Spanish-data debug overlays from the strict v5clean base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sft(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"instruction": row["messages"][0]["content"], "output": row["messages"][1]["content"]}
        for row in rows
    ]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_source_ids(rows: list[dict[str, Any]], fraction: float) -> set[str]:
    ids = sorted({str(row["source_question_id"]) for row in rows})
    ranked = sorted(ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    return set(ranked[: max(1, int(len(ids) * fraction))])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--base-validation", required=True, type=Path)
    parser.add_argument("--mcq-train", required=True, type=Path)
    parser.add_argument("--mcq-validation", required=True, type=Path)
    parser.add_argument("--freeform-train", required=True, type=Path)
    parser.add_argument("--freeform-validation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    base_train = load(args.base_train.resolve())
    base_validation = load(args.base_validation.resolve())
    mcq_train = load(args.mcq_train.resolve())
    mcq_validation = load(args.mcq_validation.resolve())
    ff_train = load(args.freeform_train.resolve())
    ff_validation = load(args.freeform_validation.resolve())

    def filtered(rows: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
        return [row for row in rows if str(row["source_question_id"]) in ids]

    mixed_train_ids = choose_source_ids(mcq_train, 0.10)
    mixed_validation_ids = choose_source_ids(mcq_validation, 0.10)
    variants = {
        "mcq_only": (mcq_train, mcq_validation),
        "freeform_only": ([], ff_validation),
        "mixed_10pct": (
            filtered(mcq_train, mixed_train_ids) + filtered(ff_train, mixed_train_ids),
            filtered(mcq_validation, mixed_validation_ids) + filtered(ff_validation, mixed_validation_ids),
        ),
    }
    # The freeform-only train side is kept separate above so it cannot
    # accidentally include MCQ rows; validation follows the same rule.
    variants["freeform_only"] = (ff_train, ff_validation)

    root = args.output_root.resolve()
    manifest: dict[str, Any] = {
        "format": "spanish_debug_dataset_manifest_v1",
        "base_train": {"path": str(args.base_train.resolve()), "sha256": digest(args.base_train.resolve()), "count": len(base_train)},
        "base_validation": {"path": str(args.base_validation.resolve()), "sha256": digest(args.base_validation.resolve()), "count": len(base_validation)},
        "v5_settings": {"lr": 5e-5, "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.025, "batch_size": 4, "grad_accum": 4, "max_seq_len": 2048, "epochs": 1},
        "variants": {},
        "sealed_test_paths_read": [],
    }
    for name, (spanish_train, spanish_validation) in variants.items():
        train = base_train + sft(spanish_train)
        validation = base_validation + sft(spanish_validation)
        variant_dir = root / name
        write(variant_dir / "train.jsonl", train)
        write(variant_dir / "validation.jsonl", validation)
        train_ids = sorted({str(row["source_question_id"]) for row in spanish_train})
        validation_ids = sorted({str(row["source_question_id"]) for row in spanish_validation})
        if set(train_ids) & set(validation_ids):
            raise ValueError(f"{name}: Spanish source IDs overlap across splits")
        manifest["variants"][name] = {
            "train_path": str((variant_dir / "train.jsonl").resolve()),
            "validation_path": str((variant_dir / "validation.jsonl").resolve()),
            "train_sha256": digest(variant_dir / "train.jsonl"),
            "validation_sha256": digest(variant_dir / "validation.jsonl"),
            "train_count": len(train),
            "validation_count": len(validation),
            "spanish_train_count": len(spanish_train),
            "spanish_validation_count": len(spanish_validation),
            "spanish_train_source_count": len(train_ids),
            "spanish_validation_source_count": len(validation_ids),
            "spanish_train_source_ids": train_ids,
            "spanish_validation_source_ids": validation_ids,
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in data.items() if key.endswith("count")} for name, data in manifest["variants"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
