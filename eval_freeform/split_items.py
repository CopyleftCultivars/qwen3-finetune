#!/usr/bin/env python3
"""Create the stratified free-form dev/test split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ITEM_TYPES = [
    "multi_hop_factual",
    "diagnostic_reasoning",
    "free_form_explanation",
    "constraint_satisfaction",
    "quantitative",
]
DEFAULT_SEED = 20260715
DEFAULT_DEV_FRACTION = 0.60


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows))


def stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    dev_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    rng = random.Random(seed)
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    manifest_types: dict[str, Any] = {}
    for item_type in ITEM_TYPES:
        bucket = sorted(by_type[item_type], key=lambda row: row["id"])
        rng.shuffle(bucket)
        split_index = round(len(bucket) * dev_fraction)
        dev_rows = sorted(bucket[:split_index], key=lambda row: row["id"])
        test_rows = sorted(bucket[split_index:], key=lambda row: row["id"])
        dev.extend(dev_rows)
        test.extend(test_rows)
        manifest_types[item_type] = {
            "dev_count": len(dev_rows),
            "test_count": len(test_rows),
            "dev_ids": [row["id"] for row in dev_rows],
            "test_ids": [row["id"] for row in test_rows],
        }

    dev.sort(key=lambda row: (ITEM_TYPES.index(row["type"]), row["id"]))
    test.sort(key=lambda row: (ITEM_TYPES.index(row["type"]), row["id"]))
    manifest = {
        "format": "eval_freeform_split_manifest_v1",
        "seed": seed,
        "dev_fraction": dev_fraction,
        "items_count": len(rows),
        "dev_count": len(dev),
        "test_count": len(test),
        "dev_path": "eval_freeform/dev.jsonl",
        "test_path": "eval_freeform/test.jsonl",
        "sealed_test": True,
        "types": manifest_types,
        "all_dev_ids": [row["id"] for row in dev],
        "all_test_ids": [row["id"] for row in test],
    }
    return dev, test, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stratify eval_freeform/items.jsonl into dev/test JSONL files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--items", default="eval_freeform/items.jsonl")
    parser.add_argument("--dev", default="eval_freeform/dev.jsonl")
    parser.add_argument("--test", default="eval_freeform/test.jsonl")
    parser.add_argument("--manifest", default="eval_freeform/split_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    args = parser.parse_args()

    if not 0.0 < args.dev_fraction < 1.0:
        raise SystemExit("--dev-fraction must be between 0 and 1")
    rows = load_jsonl(Path(args.items))
    dev, test, manifest = stratified_split(rows, args.seed, args.dev_fraction)
    write_jsonl(Path(args.dev), dev)
    write_jsonl(Path(args.test), test)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "items": len(rows),
        "dev": len(dev),
        "test": len(test),
        "seed": args.seed,
        "manifest": args.manifest,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
