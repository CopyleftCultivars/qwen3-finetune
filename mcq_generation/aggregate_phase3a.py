#!/usr/bin/env python3
"""Aggregate judge-scored MCQs across all 9 categories into the final Phase 3a
training source: qwen3-finetune/data/sources/qmd_mcq_generated.jsonl.

Only "keep" verdict items are included by default — "review" items are borderline
by definition and excluded unless --include-review is passed. Re-runs the benchmark
dedup check one more time on the aggregate as a final safety net (belt-and-suspenders
with the per-category dedup already applied during generation).

Usage:
    python aggregate_phase3a.py --output ../data/sources/qmd_mcq_generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dedup_against_benchmark import load_benchmark_stems, best_match

CATEGORIES = [
    "permaculture", "organic_farming", "natural_fertilizers", "regenerative_agriculture",
    "agroecology", "soil_science_and_microbiology", "sustainable_water_management",
    "integrated_farming_systems", "indigenous_and_traditional_knowledge",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--include-review", action="store_true",
                     help="Also include 'review' verdict items, not just 'keep'")
    ap.add_argument("--benchmark", default="[LOCAL_PATH]", type=Path)
    args = ap.parse_args()

    accepted_verdicts = {"keep", "review"} if args.include_review else {"keep"}

    aggregated: list[dict] = []
    per_cat_counts: dict[str, int] = {}
    for cat in CATEGORIES:
        path = Path(f"quality/scores/{cat}_judged.jsonl")
        if not path.exists():
            print(f"warning: {path} not found, skipping {cat}", file=sys.stderr)
            continue
        count = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["quality_scores"]["verdict"] in accepted_verdicts:
                    aggregated.append(rec)
                    count += 1
        per_cat_counts[cat] = count

    print("Per-category counts:", file=sys.stderr)
    for cat, count in per_cat_counts.items():
        print(f"  {cat}: {count}", file=sys.stderr)
    print(f"Total aggregated: {len(aggregated)}", file=sys.stderr)

    # Final dedup safety net against the benchmark
    stems = load_benchmark_stems(args.benchmark)
    final: list[dict] = []
    rejected = 0
    for rec in aggregated:
        _, ratio = best_match(rec["question"], stems)
        if ratio >= 0.75:
            rejected += 1
            continue
        final.append(rec)
    if rejected:
        print(f"Final dedup pass rejected {rejected} more (unexpected — should be 0 given "
              "per-category dedup already ran)", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for rec in final:
            # Strip judge-internal fields, keep the canonical MCQ + provenance fields
            out = {
                "question_id": rec["question_id"],
                "question": rec["question"],
                "A": rec["A"], "B": rec["B"], "C": rec["C"], "D": rec["D"],
                "answer": rec["answer"],
                "topic": rec["topic"],
                "category": rec["category"],
                "source_passage": rec.get("source_passage", ""),
                "judge_composite": rec["quality_scores"]["composite"],
                "judge_verdict": rec["quality_scores"]["verdict"],
            }
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(final)} questions to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
