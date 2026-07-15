#!/usr/bin/env python3
"""Build the free-form eval contamination report.

The check is intentionally exact and deterministic: normalize question +
reference_answer text, compare 13-grams against current/future training text
sources, and quarantine any item whose exact 13-gram overlap crosses the
documented threshold.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TRAINING_CORPUS = [
    "data/train_v3.jsonl",
    "data/sources/gemma4/hf_dataset/train.jsonl",
    "data/sources/qmd_mcq_generated.jsonl",
    "data/sources/qmd_mcq_all_phase3.jsonl",
]
NGRAM_N = 13
HIGH_OVERLAP_MIN_NGRAMS = 1
HIGH_OVERLAP_MIN_FRACTION = 0.05


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int = NGRAM_N) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def collect_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(collect_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(collect_text(v) for v in value)
    return ""


def load_jsonl_text(path: Path) -> str:
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                chunks.append(collect_text(json.loads(line)))
            except json.JSONDecodeError:
                chunks.append(line)
    return "\n".join(chunks)


def load_items(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_training_index(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    scanned: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        text = load_jsonl_text(path) if path.suffix == ".jsonl" else path.read_text(errors="ignore")
        scanned.append(str(path))
        for gram in ngrams(normalize(text)):
            index.setdefault(gram, {"training_path": str(path)})
    return index, scanned


def report_item(item: dict[str, Any], training_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item_text = f"{item['question']}\n{item['reference_answer']}"
    item_ngrams = ngrams(normalize(item_text))
    overlaps = sorted(gram for gram in item_ngrams if gram in training_index)
    overlap_fraction = len(overlaps) / max(len(item_ngrams), 1)
    high_overlap = (
        len(overlaps) >= HIGH_OVERLAP_MIN_NGRAMS
        and overlap_fraction >= HIGH_OVERLAP_MIN_FRACTION
    )
    decision = "quarantine" if high_overlap else "keep"
    first_overlap = overlaps[0] if overlaps else None
    return {
        "id": item["id"],
        "type": item["type"],
        "source_doc": item["source_doc"],
        "item_13gram_count": len(item_ngrams),
        "overlap_13gram_count": len(overlaps),
        "overlap_fraction": round(overlap_fraction, 6),
        "max_overlap_text": first_overlap,
        "max_overlap_training_path": (
            training_index[first_overlap]["training_path"] if first_overlap else None
        ),
        "high_overlap": high_overlap,
        "decision": decision,
        "source_doc_action": "exclude_from_future_training",
    }


def write_source_exclusions(path: Path, items: list[dict[str, Any]]) -> None:
    source_docs = sorted({str(item["source_doc"]) for item in items})
    payload = {
        "format": "eval_freeform_source_doc_exclusions_v1",
        "reason": "Canonical held-out free-form eval source documents; exclude from future finetuning corpus builds.",
        "source_doc_count": len(source_docs),
        "source_docs": source_docs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit eval_freeform contamination report and source exclusions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--items", default="eval_freeform/items.jsonl")
    parser.add_argument("--output", default="eval_freeform/contamination_report.json")
    parser.add_argument("--source-exclusions", default="eval_freeform/source_doc_exclusions.json")
    parser.add_argument(
        "--training-corpus",
        nargs="*",
        default=DEFAULT_TRAINING_CORPUS,
        help="JSONL/text files to scan as the current training corpus.",
    )
    args = parser.parse_args()

    items = load_items(Path(args.items))
    training_paths = [Path(path) for path in args.training_corpus]
    training_index, scanned_paths = build_training_index(training_paths)

    per_item = [report_item(item, training_index) for item in items]
    decision_counts = Counter(row["decision"] for row in per_item)
    high_overlap_survivors = [
        row["id"] for row in per_item if row["high_overlap"] and row["decision"] == "keep"
    ]

    exclusion_path = Path(args.source_exclusions)
    write_source_exclusions(exclusion_path, items)

    report = {
        "format": "eval_freeform_contamination_report_v1",
        "items_path": args.items,
        "training_corpus_paths": scanned_paths,
        "ngram_n": NGRAM_N,
        "high_overlap_threshold": {
            "min_exact_13gram_overlaps": HIGH_OVERLAP_MIN_NGRAMS,
            "min_overlap_fraction": HIGH_OVERLAP_MIN_FRACTION,
        },
        "item_count": len(items),
        "source_doc_exclusion_count": len({item["source_doc"] for item in items}),
        "source_doc_exclusions_path": str(exclusion_path),
        "decision_counts": dict(sorted(decision_counts.items())),
        "high_overlap_survivor_count": len(high_overlap_survivors),
        "high_overlap_survivor_ids": high_overlap_survivors,
        "per_item": per_item,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "items": len(items),
        "decisions": dict(sorted(decision_counts.items())),
        "high_overlap_survivor_count": len(high_overlap_survivors),
        "source_doc_exclusion_count": report["source_doc_exclusion_count"],
        "output": str(output_path),
        "source_exclusions": str(exclusion_path),
    }, indent=2, sort_keys=True))
    if high_overlap_survivors:
        raise SystemExit("high-overlap items survived into the used eval set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
