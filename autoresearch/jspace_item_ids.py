#!/usr/bin/env python3
"""Resolve J-space dev rows to QMD training IDs.

The J-space annotator writes ``item_id`` values such as ``row_000009``.  These
are one-based row IDs from the signal's ``source_split``; they are not QMD
training IDs.  This utility follows the same normalization used by
``autoresearch/score_mcq_ll.py``, joins the normalized question text to the
question bank under ``mcq_generation/output``, and writes the resulting QMD
IDs in the JSON shape consumed by ``prepare_dataset_v3.py``'s existing
``--target-permutation-ids`` option.

Example:

    python autoresearch/jspace_item_ids.py \
        --bucket position_unstable \
        --signals results/jspace_item_signals.jsonl \
        --qmd data/sources/qmd_mcq_all_phase3.jsonl \
        --category-map-dir mcq_generation/output \
        --output results/jspace_position_unstable_target_ids.json \
        --report results/jspace_position_unstable_join_report.json

Add ``--dry-run`` to print the coverage and selection counts without writing
the ID list.  Only IDs present in the supplied QMD pool are emitted, so a
successful text join that is absent from the current pool is reported but
cannot accidentally create a zero-overlap target file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from score_mcq_ll import normalize_records
except ImportError:  # pragma: no cover - supports importing as a package module
    from autoresearch.score_mcq_ll import normalize_records


JSPACE_BUCKETS = {
    "robust",
    "absent",
    "late_weak",
    "present_unsurfaced",
    "format_sensitive",
    "position_unstable",
}
SEALED_PATH_FRAGMENTS = ("eval_freeform/test.jsonl",)
ROW_ID_RE = re.compile(r"^row_(?P<number>[0-9]+)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _input_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def _check_not_sealed(path: Path) -> None:
    path_text = path.as_posix()
    if any(fragment in path_text for fragment in SEALED_PATH_FRAGMENTS):
        raise ValueError(f"refusing to read sealed test path: {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _check_not_sealed(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _normalize_question(text: str) -> str:
    return " ".join(text.lower().split())


def _load_question_map(
    category_map_dir: Path,
) -> dict[str, set[tuple[str, str]]]:
    """Return normalized question -> (QMD ID, category slug) candidates."""

    question_map: dict[str, set[tuple[str, str]]] = {}
    for category_file in sorted(category_map_dir.glob("*.jsonl")):
        for record in _load_jsonl(category_file):
            question = record.get("question")
            question_id = record.get("question_id")
            if not isinstance(question, str) or not question:
                continue
            if question_id is None:
                continue
            qid = str(question_id)
            category = category_file.stem
            question_map.setdefault(_normalize_question(question), set()).add(
                (qid, category)
            )
    if not question_map:
        raise ValueError(f"no question records found in {category_map_dir}")
    return question_map


def _load_source_questions(source_path: Path) -> dict[str, dict[str, Any]]:
    """Normalize one signal source exactly as the scorer does."""

    rows = _load_jsonl(source_path)
    by_item_id: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        records, _skipped = normalize_records([row])
        if records:
            # J-space item IDs are positional source-row IDs.  The scorer's
            # row_N convention is one-based, including only non-empty JSONL
            # rows, because that is how its load_benchmark helper presents the
            # source to normalize_records().
            by_item_id[f"row_{row_number:06d}"] = records[0]
    return by_item_id


def _qmd_question_id(record: dict[str, Any]) -> str | None:
    value = record.get("question_id", record.get("_source_question_id"))
    return str(value) if value is not None else None


def _load_qmd_ids(qmd_path: Path) -> list[str]:
    records = _load_jsonl(qmd_path)
    ids: list[str] = []
    for record in records:
        # Match prepare_dataset_v3.py: only records with both fields become
        # training pairs and therefore belong to the denominator.
        if "instruction" not in record or "output" not in record:
            continue
        question_id = _qmd_question_id(record)
        if question_id is not None:
            ids.append(question_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{qmd_path}: duplicate QMD question IDs")
    return ids


def _parse_buckets(value: str) -> list[str]:
    buckets = sorted({part.strip() for part in value.split(",") if part.strip()})
    unknown = sorted(set(buckets) - JSPACE_BUCKETS)
    if unknown:
        raise ValueError(f"unknown J-space bucket(s): {', '.join(unknown)}")
    if not buckets:
        raise ValueError("at least one J-space bucket is required")
    return buckets


def build_report(
    signals_path: Path,
    qmd_path: Path,
    category_map_dir: Path,
    buckets: list[str],
    root: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducible join/selection report without writing output."""

    root = root or _repo_root()
    signals = _load_jsonl(signals_path)
    selected_signals = [row for row in signals if row.get("bucket") in buckets]
    question_map = _load_question_map(category_map_dir)
    qmd_ids = _load_qmd_ids(qmd_path)
    qmd_id_set = set(qmd_ids)

    source_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    matches: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_agreements = 0
    category_disagreements = 0

    for signal in selected_signals:
        item_id = str(signal.get("item_id", ""))
        source_value = signal.get("source_split")
        match: dict[str, Any] = {
            "item_id": item_id,
            "bucket": signal.get("bucket"),
            "signal_category": signal.get("category", "unknown"),
            "source_split": source_value,
        }
        if ROW_ID_RE.fullmatch(item_id) is None:
            match["status"] = "invalid_source_item_id"
            matches.append(match)
            continue
        if not isinstance(source_value, str) or not source_value:
            match["status"] = "missing_source_split"
            matches.append(match)
            continue

        source_path = _input_path(source_value, root)
        if source_path not in source_cache:
            source_cache[source_path] = _load_source_questions(source_path)
        source_record = source_cache[source_path].get(item_id)
        if source_record is None:
            match["status"] = "source_item_not_found"
            matches.append(match)
            continue

        question = str(source_record["question"])
        candidates = question_map.get(_normalize_question(question), set())
        match["question"] = question
        if not candidates:
            match["status"] = "question_not_found"
            matches.append(match)
            continue
        if len(candidates) > 1:
            match["status"] = "ambiguous_question_match"
            match["candidates"] = [
                {"question_id": qid, "category": category}
                for qid, category in sorted(candidates)
            ]
            matches.append(match)
            continue

        question_id, mapped_category = next(iter(candidates))
        match["question_id"] = question_id
        match["mapped_category"] = mapped_category
        if signal.get("category") == mapped_category:
            category_agreements += 1
        else:
            category_disagreements += 1
        if question_id in qmd_id_set:
            match["status"] = "selected"
            selected_ids.add(question_id)
        else:
            match["status"] = "resolved_not_in_qmd"
        matches.append(match)

    status_counts: dict[str, int] = {}
    for match in matches:
        status = str(match["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    resolved = status_counts.get("selected", 0) + status_counts.get(
        "resolved_not_in_qmd", 0
    )
    selected_count = len(selected_ids)
    qmd_count = len(qmd_ids)
    return {
        "signals": signals_path.as_posix(),
        "source_splits": sorted(
            {str(row.get("source_split")) for row in selected_signals}
        ),
        "category_map_dir": category_map_dir.as_posix(),
        "qmd_source": qmd_path.as_posix(),
        "buckets": buckets,
        "signal_count": len(selected_signals),
        "question_join_count": resolved,
        "question_join_fraction": (
            round(resolved / len(selected_signals), 6) if selected_signals else 0.0
        ),
        "category_agreement_count": category_agreements,
        "category_disagreement_count": category_disagreements,
        "status_counts": dict(sorted(status_counts.items())),
        "qmd_record_count": qmd_count,
        "selected_qmd_record_count": selected_count,
        "selected_qmd_fraction": round(selected_count / qmd_count, 6)
        if qmd_count
        else 0.0,
        "selected_question_ids": sorted(selected_ids),
        "matches": matches,
    }


def _write_target_ids(path: Path, question_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "augmentation_seed_set": [
            {"question_id": question_id} for question_id in question_ids
        ]
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _print_report(report: dict[str, Any]) -> None:
    print(
        "J-space item join: "
        f"{report['question_join_count']}/{report['signal_count']} signal rows "
        f"resolved ({report['question_join_fraction']:.2%})"
    )
    print(
        "Category agreement for resolved rows: "
        f"{report['category_agreement_count']} agreement(s), "
        f"{report['category_disagreement_count']} disagreement(s)"
    )
    print(
        "QMD selection: "
        f"{report['selected_qmd_record_count']}/{report['qmd_record_count']} records "
        f"({report['selected_qmd_fraction']:.2%})"
    )
    print(f"Statuses: {report['status_counts']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", default="results/jspace_item_signals.jsonl")
    parser.add_argument("--qmd", default="data/sources/qmd_mcq_all_phase3.jsonl")
    parser.add_argument("--category-map-dir", default="mcq_generation/output")
    parser.add_argument(
        "--bucket",
        default="position_unstable",
        help="Comma-separated signal bucket(s), default: position_unstable",
    )
    parser.add_argument("--output", help="target-permutation JSON path")
    parser.add_argument("--report", help="optional JSON report path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print counts without writing the target ID list",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    signals_path = _input_path(args.signals, root)
    qmd_path = _input_path(args.qmd, root)
    category_map_dir = _input_path(args.category_map_dir, root)
    buckets = _parse_buckets(args.bucket)
    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is used")

    report = build_report(
        signals_path=signals_path,
        qmd_path=qmd_path,
        category_map_dir=category_map_dir,
        buckets=buckets,
        root=root,
    )
    _print_report(report)

    if args.output and not args.dry_run:
        output_path = _input_path(args.output, root)
        _write_target_ids(output_path, report["selected_question_ids"])
        print(f"Wrote target IDs: {output_path}")
    if args.report:
        report_path = _input_path(args.report, root)
        _write_report(report_path, report)
        print(f"Wrote join report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
