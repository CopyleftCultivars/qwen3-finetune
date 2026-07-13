#!/usr/bin/env python3
"""
Dataset cartography mapping for MCQ benchmarks (Issue #57).

Adapts Swayamdipta et al. (EMNLP 2020) dataset cartography to the benchmark
setting.  Instead of training dynamics across epochs, we use the spread of
correct/incorrect signals across the 12-model IRT evaluation pool.

Metrics per item
----------------
confidence   Mean fraction of models that answered correctly.
             High → item is easy; low → item is hard.
variability  Standard deviation of per-model correctness (0/1) across the pool.
             High → different model tiers disagree; low → unanimous.
correctness  Fraction of models correct (same as confidence when all models evaluated).

Cartography regions (following Swayamdipta et al.)
---------------------------------------------------
easy-to-learn   confidence ≥ 0.7  AND variability < 0.3
                All or most models get it right. Low signal for discrimination.
ambiguous       variability ≥ 0.3
                Models split — item sits at a capability boundary. Most useful
                for benchmark discrimination.
hard-to-learn   confidence < 0.3  AND variability < 0.3
                All or most models get it wrong. Potential label error → flags
                for Layer 5.

Usage
-----
    # Map all items (requires IRT eval responses)
    conda run -n nf-benchmark python quality/cartography.py map

    # Dry run — show data availability
    conda run -n nf-benchmark python quality/cartography.py map --dry-run

    # Report — print cartography summary
    conda run -n nf-benchmark python quality/cartography.py report

Output
------
    quality/cartography/cartography_map.jsonl   per-item cartography data
    quality/cartography/hard_to_learn.jsonl     hard-to-learn items → Layer 5
    quality/cartography/cartography_report.json summary statistics
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
BENCHMARKS_DIR = WORKSPACE / "benchmarks"
IRT_DIR = WORKSPACE / "quality" / "irt"
CARTOGRAPHY_DIR = WORKSPACE / "quality" / "cartography"

# ---------------------------------------------------------------------------
# Thresholds (following Swayamdipta et al.)
# ---------------------------------------------------------------------------
EASY_CONFIDENCE_MIN = 0.7
HARD_CONFIDENCE_MAX = 0.3
AMBIGUOUS_VARIABILITY_MIN = 0.3
MIN_MODELS = 3   # minimum model responses to include an item


# ---------------------------------------------------------------------------
# Data loading
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


def load_irt_responses() -> dict[str, list[int]]:
    """
    Load IRT eval responses.

    Returns {question_id: [1, 0, 1, ...]}  where each value is 1 (correct)
    or 0 (incorrect) for one model.  None (unparseable) responses are excluded.
    """
    responses_dir = IRT_DIR / "responses"
    if not responses_dir.exists():
        return {}

    per_item: dict[str, list[int]] = {}
    for path in sorted(responses_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Skip unparseable responses
            if rec.get("predicted") is None:
                continue
            qid = rec["question_id"]
            if qid not in per_item:
                per_item[qid] = []
            per_item[qid].append(1 if rec.get("correct") else 0)
    return per_item


# ---------------------------------------------------------------------------
# Cartography computation
# ---------------------------------------------------------------------------


def compute_cartography(correctness_vec: list[int]) -> dict:
    """
    Compute cartography metrics for a single item.

    Parameters
    ----------
    correctness_vec : list of 0/1 values (one per model)

    Returns
    -------
    dict with confidence, variability, region
    """
    n = len(correctness_vec)
    confidence = sum(correctness_vec) / n

    # Population std dev
    mean = confidence
    variance = sum((x - mean) ** 2 for x in correctness_vec) / n
    variability = math.sqrt(variance)

    if variability >= AMBIGUOUS_VARIABILITY_MIN:
        region = "ambiguous"
    elif confidence >= EASY_CONFIDENCE_MIN:
        region = "easy-to-learn"
    elif confidence <= HARD_CONFIDENCE_MAX:
        region = "hard-to-learn"
    else:
        region = "ambiguous"  # mid-confidence, low variability → treat as ambiguous

    return {
        "confidence": round(confidence, 4),
        "variability": round(variability, 4),
        "n_models": n,
        "n_correct": sum(correctness_vec),
        "region": region,
    }


# ---------------------------------------------------------------------------
# map subcommand
# ---------------------------------------------------------------------------


def cmd_map(args: argparse.Namespace) -> int:
    all_items = load_benchmarks()
    if not all_items:
        print("ERROR: No benchmark items found.", file=sys.stderr)
        return 1

    responses = load_irt_responses()

    eligible = {
        qid: vec for qid, vec in responses.items()
        if len(vec) >= MIN_MODELS and qid in all_items
    }

    print(f"Benchmark items total:        {len(all_items)}")
    print(f"Items with IRT responses:     {len(responses)}")
    print(f"Eligible (≥{MIN_MODELS} responses):   {len(eligible)}")

    if args.dry_run:
        print("\n[dry-run] Would compute cartography for eligible items.")
        return 0

    if not eligible:
        print(
            "\nNo eligible items — run `quality/irt_analysis.py eval` first.",
            file=sys.stderr,
        )
        return 1

    CARTOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)

    map_records: list[dict] = []
    hard_records: list[dict] = []
    region_counts: dict[str, int] = {"easy-to-learn": 0, "ambiguous": 0, "hard-to-learn": 0}

    for qid, vec in sorted(eligible.items()):
        item = all_items[qid]
        metrics = compute_cartography(vec)

        record = {
            "question_id": qid,
            "category": item.get("_category", ""),
            "topic": item.get("topic", ""),
            **metrics,
        }
        map_records.append(record)
        region_counts[metrics["region"]] = region_counts.get(metrics["region"], 0) + 1

        if metrics["region"] == "hard-to-learn":
            hard_records.append({**item, **metrics, "flag_source": "cartography_hard"})

    # Write cartography map
    map_path = CARTOGRAPHY_DIR / "cartography_map.jsonl"
    with open(map_path, "w", encoding="utf-8") as fh:
        for rec in map_records:
            fh.write(json.dumps(rec) + "\n")

    # Write hard-to-learn items for Layer 5
    hard_path = CARTOGRAPHY_DIR / "hard_to_learn.jsonl"
    with open(hard_path, "w", encoding="utf-8") as fh:
        for rec in hard_records:
            fh.write(json.dumps(rec) + "\n")

    # Write report
    total = len(map_records)
    report = {
        "total_mapped": total,
        "regions": {
            region: {
                "count": count,
                "fraction": round(count / max(1, total), 4),
            }
            for region, count in region_counts.items()
        },
        "thresholds": {
            "easy_confidence_min": EASY_CONFIDENCE_MIN,
            "hard_confidence_max": HARD_CONFIDENCE_MAX,
            "ambiguous_variability_min": AMBIGUOUS_VARIABILITY_MIN,
        },
        "min_models_threshold": MIN_MODELS,
    }
    report_path = CARTOGRAPHY_DIR / "cartography_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    easy = region_counts["easy-to-learn"]
    ambig = region_counts["ambiguous"]
    hard = region_counts["hard-to-learn"]

    print(f"\nCartography map ({total} items):")
    print(f"  easy-to-learn:  {easy:4d}  ({easy / total * 100:.1f}%)")
    print(f"  ambiguous:      {ambig:4d}  ({ambig / total * 100:.1f}%)")
    print(f"  hard-to-learn:  {hard:4d}  ({hard / total * 100:.1f}%)  → Layer 5")
    print(f"\n  {map_path.relative_to(WORKSPACE)}")
    print(f"  {hard_path.relative_to(WORKSPACE)}")
    print(f"  {report_path.relative_to(WORKSPACE)}")

    return 0


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------


def cmd_report(_args: argparse.Namespace) -> int:
    report_path = CARTOGRAPHY_DIR / "cartography_report.json"
    if not report_path.exists():
        print("No cartography report found. Run `map` first.", file=sys.stderr)
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    total = report["total_mapped"]

    print("Dataset Cartography — Report")
    print(f"  Items mapped: {total}")
    print(f"  Min models threshold: {report['min_models_threshold']}")
    print()

    regions = report.get("regions", {})
    for region in ("easy-to-learn", "ambiguous", "hard-to-learn"):
        data = regions.get(region, {})
        count = data.get("count", 0)
        pct = data.get("fraction", 0) * 100
        flag = "  ← Layer 5 flag" if region == "hard-to-learn" else ""
        print(f"  {region:<18s}  {count:4d}  ({pct:.1f}%){flag}")

    map_path = CARTOGRAPHY_DIR / "cartography_map.jsonl"
    if map_path.exists():
        # Per-category breakdown
        by_cat: dict[str, dict[str, int]] = {}
        for line in map_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cat = item.get("category", "unknown")
            region = item.get("region", "unknown")
            if cat not in by_cat:
                by_cat[cat] = {"easy-to-learn": 0, "ambiguous": 0, "hard-to-learn": 0}
            by_cat[cat][region] = by_cat[cat].get(region, 0) + 1

        print(f"\n  {'Category':45s}  {'Easy':>5}  {'Ambig':>5}  {'Hard':>5}")
        print("  " + "-" * 65)
        for cat, counts in sorted(by_cat.items()):
            print(
                f"  {cat:45s}  "
                f"{counts['easy-to-learn']:5d}  "
                f"{counts['ambiguous']:5d}  "
                f"{counts['hard-to-learn']:5d}"
            )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset cartography mapping for MCQ benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    map_p = sub.add_parser("map", help="Compute cartography map from IRT responses")
    map_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show data availability without computing",
    )

    sub.add_parser("report", help="Print cartography summary")

    args = parser.parse_args()
    return {"map": cmd_map, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
