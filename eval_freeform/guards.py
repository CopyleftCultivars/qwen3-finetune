"""Shared guards for the English free-form dev-only evaluation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SEALED_ERROR = "sealed free-form evaluation paths are forbidden"


def assert_not_sealed_path(path: str | Path) -> Path:
    """Reject sealed English test, Spanish benchmark, and explicitly sealed paths."""

    resolved = Path(path).expanduser().resolve()
    parts = {part.casefold() for part in resolved.parts}
    normalized = resolved.as_posix().casefold()
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "sealed" in parts or "sealed" in resolved.name.casefold():
        raise ValueError(SEALED_ERROR)
    if "/eval_freeform/test.jsonl" in normalized:
        raise ValueError(SEALED_ERROR)
    return resolved


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = assert_not_sealed_path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def load_dev_ids(split_manifest: str | Path = "eval_freeform/split_manifest.json") -> set[str]:
    """Load only split metadata; the sealed test file itself is never opened."""

    manifest_path = Path(split_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dev_ids = manifest.get("all_dev_ids")
    test_ids = manifest.get("all_test_ids")
    if not isinstance(dev_ids, list) or not isinstance(test_ids, list):
        raise ValueError(f"{manifest_path}: missing all_dev_ids/all_test_ids")
    dev = {str(item) for item in dev_ids}
    test = {str(item) for item in test_ids}
    if dev & test:
        raise ValueError(f"{manifest_path}: dev/test ID overlap")
    return dev


def validate_dev_ids(
    ids: Iterable[str],
    split_manifest: str | Path = "eval_freeform/split_manifest.json",
    *,
    allow_subset: bool = True,
) -> set[str]:
    """Ensure an evaluation artifact contains only fixed English dev IDs."""

    values = [str(item) for item in ids]
    if len(values) != len(set(values)):
        raise ValueError("free-form evaluation artifact contains duplicate item IDs")
    dev_ids = load_dev_ids(split_manifest)
    unknown = set(values) - dev_ids
    if unknown:
        raise ValueError(
            "free-form evaluation artifact contains non-dev or sealed IDs: "
            f"{sorted(unknown)[:5]}"
        )
    if not allow_subset and set(values) != dev_ids:
        raise ValueError("free-form evaluation artifact does not contain the complete dev split")
    return set(values)
