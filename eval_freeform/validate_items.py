#!/usr/bin/env python3
"""Validate the canonical free-form eval item set and review manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TYPES = {
    "multi_hop_factual",
    "diagnostic_reasoning",
    "free_form_explanation",
    "constraint_satisfaction",
    "quantitative",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def validate_review_manifest(path: Path, item_ids: set[str]) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"review manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("review manifest must be a JSON object")
    if manifest.get("review_type") != "literal_human_review":
        raise ValueError("review manifest review_type must be literal_human_review")
    if not manifest.get("reviewed_by") or not manifest.get("reviewed_at"):
        raise ValueError("review manifest must record reviewed_by and reviewed_at")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("review manifest must include an items list")
    reviewed_ids: set[str] = set()
    for index, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ValueError(f"review manifest item {index} is not an object")
        item_id = entry.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"review manifest item {index} missing id")
        if entry.get("reviewed") is not True:
            raise ValueError(f"review manifest item {item_id} is not reviewed=true")
        if entry.get("reviewed_by") != manifest["reviewed_by"]:
            raise ValueError(f"review manifest item {item_id} reviewed_by mismatch")
        if entry.get("reviewed_at") != manifest["reviewed_at"]:
            raise ValueError(f"review manifest item {item_id} reviewed_at mismatch")
        reviewed_ids.add(item_id)
    missing = item_ids - reviewed_ids
    extra = reviewed_ids - item_ids
    if missing:
        raise ValueError(f"review manifest missing item ids: {sorted(missing)[:10]}")
    if extra:
        raise ValueError(f"review manifest has extra item ids: {sorted(extra)[:10]}")
    return manifest


def validate_items(items_path: Path, manifest_path: Path) -> dict[str, Any]:
    rows = load_jsonl(items_path)
    if not 150 <= len(rows) <= 250:
        raise ValueError(f"expected 150-250 items, found {len(rows)}")

    ids = [str(row.get("id", "")) for row in rows]
    questions = [str(row.get("question", "")) for row in rows]
    sources = [str(row.get("source_doc", "")) for row in rows]

    for label, values in {
        "id": ids,
        "question": questions,
        "source_doc": sources,
    }.items():
        duplicates = duplicate_values(values)
        if duplicates:
            raise ValueError(f"duplicate {label} values: {duplicates[:10]}")

    type_counts: Counter[str] = Counter()
    repo_root = items_path.resolve().parents[1]
    for row in rows:
        item_id = row.get("id", f"line_{row['_line_number']}")
        item_type = row.get("type")
        if item_type not in EXPECTED_TYPES:
            raise ValueError(f"{item_id}: invalid type {item_type!r}")
        type_counts[str(item_type)] += 1
        if row.get("held_out") is not True:
            raise ValueError(f"{item_id}: held_out must be true")
        if row.get("reviewed") is not True:
            raise ValueError(f"{item_id}: reviewed must be true in canonical items")
        if not row.get("question") or not row.get("reference_answer"):
            raise ValueError(f"{item_id}: missing question or reference_answer")
        source_doc = row.get("source_doc")
        if not isinstance(source_doc, str) or not source_doc:
            raise ValueError(f"{item_id}: missing source_doc")
        if not (repo_root / source_doc).resolve().exists():
            raise ValueError(f"{item_id}: source_doc does not exist: {source_doc}")

        rubric = row.get("rubric")
        if not isinstance(rubric, list) or not 3 <= len(rubric) <= 5:
            raise ValueError(f"{item_id}: rubric must have 3-5 points")
        total_weight = 0.0
        for point_index, point in enumerate(rubric):
            if not isinstance(point, dict):
                raise ValueError(f"{item_id}: rubric[{point_index}] is not an object")
            if not isinstance(point.get("point"), str) or not point["point"].strip():
                raise ValueError(f"{item_id}: rubric[{point_index}] missing point text")
            weight = point.get("weight")
            if not isinstance(weight, (int, float)):
                raise ValueError(f"{item_id}: rubric[{point_index}] missing numeric weight")
            total_weight += float(weight)
        if not math.isclose(total_weight, 1.0, abs_tol=0.011):
            raise ValueError(f"{item_id}: rubric weights sum to {total_weight:.6f}")

    missing_types = EXPECTED_TYPES - set(type_counts)
    if missing_types:
        raise ValueError(f"missing item types: {sorted(missing_types)}")
    low_types = {item_type: count for item_type, count in type_counts.items() if count < 30}
    if low_types:
        raise ValueError(f"types below 30 items: {low_types}")

    manifest = validate_review_manifest(manifest_path, set(ids))
    return {
        "items": len(rows),
        "type_counts": dict(sorted(type_counts.items())),
        "unique_source_docs": len(set(sources)),
        "review_manifest": str(manifest_path),
        "reviewed_by": manifest["reviewed_by"],
        "reviewed_at": manifest["reviewed_at"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate eval_freeform/items.jsonl and reviewed_manifest.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--items", default="eval_freeform/items.jsonl")
    parser.add_argument("--review-manifest", default="eval_freeform/reviewed_manifest.json")
    args = parser.parse_args()

    summary = validate_items(Path(args.items), Path(args.review_manifest))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
