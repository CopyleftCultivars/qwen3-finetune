#!/usr/bin/env python3
"""Combine the clean v5clean data with Spanish QMD training overlays.

This preserves the existing English training/validation pools and appends only
the Spanish QMD-derived SFT records.  The Spanish source split is already
performed before augmentation by build_spanish_qmd_training.py.
"""

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
            if not isinstance(value, dict) or "instruction" not in value or "output" not in value:
                raise ValueError(f"{path}:{line_number}: expected instruction/output JSON object")
            rows.append(value)
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--base-validation", required=True, type=Path)
    parser.add_argument("--spanish-train", required=True, type=Path, action="append")
    parser.add_argument("--spanish-validation", required=True, type=Path, action="append")
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-validation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    base_train = load(args.base_train.resolve())
    base_validation = load(args.base_validation.resolve())
    spanish_train_paths = [path.resolve() for path in args.spanish_train]
    spanish_validation_paths = [path.resolve() for path in args.spanish_validation]
    spanish_train = [row for path in spanish_train_paths for row in load(path)]
    spanish_validation = [row for path in spanish_validation_paths for row in load(path)]
    output_train = base_train + spanish_train
    output_validation = base_validation + spanish_validation
    write(args.output_train.resolve(), output_train)
    write(args.output_validation.resolve(), output_validation)
    manifest = {
        "format": "spanish_qmd_overlay_baseline_dataset_manifest_v1",
        "base_train": {"path": str(args.base_train.resolve()), "sha256": digest(args.base_train.resolve()), "count": len(base_train)},
        "base_validation": {"path": str(args.base_validation.resolve()), "sha256": digest(args.base_validation.resolve()), "count": len(base_validation)},
        "spanish_train_inputs": [{"path": str(path), "sha256": digest(path), "count": len(load(path))} for path in spanish_train_paths],
        "spanish_validation_inputs": [{"path": str(path), "sha256": digest(path), "count": len(load(path))} for path in spanish_validation_paths],
        "output_train": {"path": str(args.output_train.resolve()), "sha256": digest(args.output_train.resolve()), "count": len(output_train)},
        "output_validation": {"path": str(args.output_validation.resolve()), "sha256": digest(args.output_validation.resolve()), "count": len(output_validation)},
        "spanish_records_training_allowed": True,
        "spanish_benchmark_records_included": False,
        "sealed_test_paths_read": [],
        "notes": [
            "The English v5clean training/validation pools are retained as the regression-preserving base.",
            "Spanish overlays are QMD translations only; no Spanish benchmark transfer item is included.",
        ],
    }
    args.manifest.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.manifest.resolve().write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_train": len(base_train), "base_validation": len(base_validation),
        "spanish_train": len(spanish_train), "spanish_validation": len(spanish_validation),
        "output_train": len(output_train), "output_validation": len(output_validation),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
