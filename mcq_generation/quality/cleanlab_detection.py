#!/usr/bin/env python3
"""
Cleanlab label error detection for MCQ benchmarks (Issue #58).

Uses confident learning (Northcutt et al., 2021) to flag items where the
model-response distribution is inconsistent with the declared answer key.
This is Layer 5 of the quality pipeline.

How it works
------------
The IRT eval step (quality/irt_analysis.py eval) stores each model's chosen
letter (A/B/C/D) for every item.  For each item we build a probability
distribution over the four options: pred_probs[i, j] = fraction of models
that chose option j.  Cleanlab's find_label_issues then flags items where
this distribution is surprising given the declared correct answer.

Expected error rate: ~5–7% (MMLU-Redux found 6.5% in a similar benchmark).

Layer 5 flag sources
--------------------
1. Cleanlab-flagged items    — answer-key possibly wrong
2. IRT floor items           — all models got wrong (from quality/irt/floor_items.jsonl)
3. Gray-zone LLM-judge items — composite score 3.0–3.9 (from quality/scores/)

Usage
-----
    # Run detection (requires IRT eval responses in quality/irt/responses/)
    conda run -n nf-benchmark python quality/cleanlab_detection.py detect

    # Dry run — show data availability without running Cleanlab
    conda run -n nf-benchmark python quality/cleanlab_detection.py detect --dry-run

    # Report — print summary of flagged items (requires detect to have run)
    conda run -n nf-benchmark python quality/cleanlab_detection.py report

Output
------
    quality/cleanlab/flagged_items.jsonl   all Layer 5 flags (all three sources)
    quality/cleanlab/cleanlab_flags.jsonl  Cleanlab-only flags
    quality/cleanlab/detection_report.json summary stats
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
BENCHMARKS_DIR = WORKSPACE / "benchmarks"
IRT_DIR = WORKSPACE / "quality" / "irt"
SCORES_DIR = WORKSPACE / "quality" / "scores"
CLEANLAB_DIR = WORKSPACE / "quality" / "cleanlab"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LETTERS = ("A", "B", "C", "D")
LETTER_TO_IX = {l: i for i, l in enumerate(LETTERS)}
EPSILON = 1e-6          # floor for pred_probs (Cleanlab requires > 0)
MIN_MODELS = 3          # minimum model responses needed to include an item
GRAY_ZONE_LOW = 3.0
GRAY_ZONE_HIGH = 3.9    # LLM-judge composite score range for gray-zone flag


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_benchmarks() -> dict[str, dict]:
    """Return {question_id: item_dict} for all benchmark items."""
    items: dict[str, dict] = {}
    for path in sorted(BENCHMARKS_DIR.rglob("*_benchmark_v1.jsonl")):
        category = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                item = json.loads(line)
                item["_category"] = category
                items[item["question_id"]] = item
    return items


def load_irt_responses() -> dict[str, dict[str, list[str | None]]]:
    """
    Load all IRT eval responses.

    Returns {question_id: {"predicted": [letter_or_None, ...], "correct_answer": str}}
    """
    responses_dir = IRT_DIR / "responses"
    if not responses_dir.exists():
        return {}

    per_item: dict[str, dict] = {}
    for path in sorted(responses_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["question_id"]
            if qid not in per_item:
                per_item[qid] = {
                    "predicted": [],
                    "correct_answer": rec.get("correct_answer"),
                }
            per_item[qid]["predicted"].append(rec.get("predicted"))
    return per_item


def load_floor_items() -> list[dict]:
    """Load IRT floor items (all models wrong → potential mislabeling)."""
    path = IRT_DIR / "floor_items.jsonl"
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_gray_zone_items() -> list[dict]:
    """
    Load LLM-judge items with composite score in [GRAY_ZONE_LOW, GRAY_ZONE_HIGH].
    Reads quality/scores/*_judged.jsonl files.
    """
    gray: list[dict] = []
    for path in sorted(SCORES_DIR.glob("*_judged.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qs = item.get("quality_scores", {})
            composite = qs.get("composite")
            if composite is not None and GRAY_ZONE_LOW <= float(composite) <= GRAY_ZONE_HIGH:
                gray.append(item)
    return gray


# ---------------------------------------------------------------------------
# pred_probs builder
# ---------------------------------------------------------------------------


def build_pred_probs(
    qids: list[str],
    responses: dict[str, dict],
) -> np.ndarray:
    """
    Build (n_items, 4) pred_probs matrix.

    pred_probs[i, j] = fraction of models that chose option j for item i.
    Rows are normalised to sum to 1.  An epsilon floor prevents zero entries
    which would cause Cleanlab to error.
    """
    n = len(qids)
    mat = np.full((n, 4), EPSILON)

    for i, qid in enumerate(qids):
        rec = responses.get(qid, {})
        predictions = [p for p in rec.get("predicted", []) if p in LETTER_TO_IX]
        if not predictions:
            # No usable predictions — uniform distribution (item will be kept)
            mat[i] = np.ones(4)
        else:
            for letter in predictions:
                mat[i, LETTER_TO_IX[letter]] += 1.0

    # Normalise rows
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / row_sums
    return mat


# ---------------------------------------------------------------------------
# detect subcommand
# ---------------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> int:
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        print("ERROR: cleanlab not installed. Run: pip install cleanlab", file=sys.stderr)
        return 1

    all_items = load_benchmarks()
    if not all_items:
        print("ERROR: No benchmark items found.", file=sys.stderr)
        return 1

    responses = load_irt_responses()

    # Filter to items with enough model responses
    eligible_qids = [
        qid for qid, rec in responses.items()
        if sum(1 for p in rec["predicted"] if p is not None) >= MIN_MODELS
        and qid in all_items
    ]

    print(f"Benchmark items total:   {len(all_items)}")
    print(f"Items with IRT responses:{len(responses)}")
    print(f"Eligible (≥{MIN_MODELS} valid responses): {len(eligible_qids)}")

    if args.dry_run:
        print("\n[dry-run] Would run Cleanlab on eligible items.")
        floor_items = load_floor_items()
        gray_items = load_gray_zone_items()
        print(f"IRT floor items:         {len(floor_items)}")
        print(f"LLM-judge gray-zone:     {len(gray_items)}")
        return 0

    if not eligible_qids:
        print(
            "\nNo eligible items — run `quality/irt_analysis.py eval` first to collect "
            "model responses, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    # Build labels and pred_probs
    labels = np.array(
        [LETTER_TO_IX[all_items[qid]["answer"]] for qid in eligible_qids],
        dtype=int,
    )
    pred_probs = build_pred_probs(eligible_qids, responses)

    print("\nRunning Cleanlab confident learning...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by="prune_by_noise_rate",
        verbose=False,
    )

    flagged_qids = {eligible_qids[i] for i in issue_indices}
    print(f"Cleanlab flagged:        {len(flagged_qids)} items "
          f"({len(flagged_qids) / len(eligible_qids) * 100:.1f}% of eligible)")

    # Build Cleanlab flag records
    cleanlab_flags: list[dict] = []
    for i in issue_indices:
        qid = eligible_qids[i]
        item = all_items[qid]
        rec = responses[qid]
        predictions = [p for p in rec["predicted"] if p is not None]
        vote_counts = {letter: predictions.count(letter) for letter in LETTERS}
        majority = max(vote_counts, key=lambda letter: vote_counts[letter])
        cleanlab_flags.append({
            **item,
            "flag_source": "cleanlab",
            "declared_answer": item["answer"],
            "model_vote_counts": vote_counts,
            "model_majority_vote": majority,
            "n_models_evaluated": len(predictions),
            "self_confidence": float(pred_probs[i, labels[i]]),
        })

    # Load other Layer 5 sources
    floor_items = load_floor_items()
    gray_items = load_gray_zone_items()

    # Tag floor items not already in cleanlab flags
    cleanlab_qids = {r["question_id"] for r in cleanlab_flags}
    floor_flags = [
        {**item, "flag_source": "irt_floor"}
        for item in floor_items
        if item["question_id"] not in cleanlab_qids
    ]

    floor_and_cleanlab = cleanlab_qids | {r["question_id"] for r in floor_flags}
    gray_flags = [
        {**item, "flag_source": "llm_judge_gray_zone"}
        for item in gray_items
        if item["question_id"] not in floor_and_cleanlab
    ]

    all_flags = cleanlab_flags + floor_flags + gray_flags

    # Write outputs
    CLEANLAB_DIR.mkdir(parents=True, exist_ok=True)

    cleanlab_path = CLEANLAB_DIR / "cleanlab_flags.jsonl"
    with open(cleanlab_path, "w", encoding="utf-8") as fh:
        for rec in cleanlab_flags:
            fh.write(json.dumps(rec) + "\n")

    all_flags_path = CLEANLAB_DIR / "flagged_items.jsonl"
    with open(all_flags_path, "w", encoding="utf-8") as fh:
        for rec in all_flags:
            fh.write(json.dumps(rec) + "\n")

    # Write report
    report = {
        "total_benchmark_items": len(all_items),
        "items_with_irt_responses": len(responses),
        "eligible_items": len(eligible_qids),
        "cleanlab_flagged": len(cleanlab_flags),
        "cleanlab_flag_rate": round(len(cleanlab_flags) / max(1, len(eligible_qids)), 4),
        "irt_floor_flagged": len(floor_flags),
        "llm_judge_gray_zone_flagged": len(gray_flags),
        "total_layer5_flags": len(all_flags),
        "min_models_threshold": MIN_MODELS,
        "gray_zone_range": [GRAY_ZONE_LOW, GRAY_ZONE_HIGH],
    }
    report_path = CLEANLAB_DIR / "detection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nLayer 5 flags summary:")
    print(f"  Cleanlab:            {len(cleanlab_flags)}")
    print(f"  IRT floor:           {len(floor_flags)}")
    print(f"  LLM-judge gray-zone: {len(gray_flags)}")
    print(f"  Total:               {len(all_flags)}")
    print(f"\n  {all_flags_path.relative_to(WORKSPACE)}")
    print(f"  {report_path.relative_to(WORKSPACE)}")

    return 0


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------


def cmd_report(_args: argparse.Namespace) -> int:
    report_path = CLEANLAB_DIR / "detection_report.json"
    if not report_path.exists():
        print("No detection report found. Run `detect` first.", file=sys.stderr)
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))

    print("Cleanlab Label Error Detection — Report")
    print(f"  Benchmark items:          {report['total_benchmark_items']}")
    print(f"  Items with IRT responses: {report['items_with_irt_responses']}")
    print(f"  Eligible for Cleanlab:    {report['eligible_items']}")
    print()
    flag_rate_pct = report['cleanlab_flag_rate'] * 100
    print(f"  Cleanlab flagged:         {report['cleanlab_flagged']} "
          f"({flag_rate_pct:.1f}% — expected ~5–7%)")
    print(f"  IRT floor flagged:        {report['irt_floor_flagged']}")
    print(f"  LLM-judge gray-zone:      {report['llm_judge_gray_zone_flagged']}")
    print(f"  Total Layer 5 flags:      {report['total_layer5_flags']}")

    flags_path = CLEANLAB_DIR / "flagged_items.jsonl"
    if flags_path.exists():
        print(f"\n  Flagged items: {flags_path.relative_to(WORKSPACE)}")
        by_source: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for line in flags_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            src = item.get("flag_source", "unknown")
            cat = item.get("_category", item.get("topic", "unknown"))
            by_source[src] = by_source.get(src, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1

        print("\n  By source:")
        for src, count in sorted(by_source.items()):
            print(f"    {src:<30s} {count}")

        print("\n  By category:")
        for cat, count in sorted(by_category.items()):
            print(f"    {cat:<45s} {count}")

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cleanlab label error detection for MCQ benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser("detect", help="Run Cleanlab detection + merge Layer 5 flags")
    detect_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show data availability without running Cleanlab",
    )

    sub.add_parser("report", help="Print summary of flagged items")

    args = parser.parse_args()
    return {"detect": cmd_detect, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
