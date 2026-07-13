#!/usr/bin/env python3
"""
Feedback aggregator for MCQ regeneration (Issue #64).

Reads judge rejection signals from quality/scores/{category}_cache.json and
quality/final/regeneration_needed.json, then writes a structured
feedback_context.json that the prompt injector (Issue #65) and regeneration
mode (Issue #66) can consume.

For each topic that needs regeneration:
  - Computes per-dimension average scores across all rejected items
  - Identifies the 1–2 weakest quality dimensions
  - Extracts representative failure examples from judge justifications
  - Records the items_needed count from the regen report

Output: quality/feedback/{category}_feedback_context.json

Usage
-----
    # Aggregate feedback for one category
    conda run -n nf-benchmark python quality/feedback_aggregator.py \\
        --category natural_fertilizers

    # Aggregate for all categories that have a regen report
    conda run -n nf-benchmark python quality/feedback_aggregator.py --all

    # Dry run (show plan without writing)
    conda run -n nf-benchmark python quality/feedback_aggregator.py \\
        --category natural_fertilizers --dry-run

Output schema (per topic key in the JSON)
-----------------------------------------
{
  "<topic>": {
    "category": "natural_fertilizers",
    "items_needed": 7,
    "items_surviving": 3,
    "total_rejected": 12,
    "weakest_dimensions": [
      {
        "dimension": "distractor_quality",
        "avg_score": 2.1,
        "failure_examples": ["Distractors too obviously wrong", "..."]
      }
    ],
    "common_rejection_reasons": ["...", "..."]
  }
}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
QUALITY_DIR = WORKSPACE / "quality"
SCORES_DIR = QUALITY_DIR / "scores"
FINAL_DIR = QUALITY_DIR / "final"
FEEDBACK_DIR = QUALITY_DIR / "feedback"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSIONS = [
    "factual_correctness",
    "question_clarity",
    "distractor_quality",
    "domain_relevance",
    "difficulty_appropriateness",
]
# How many of the weakest dimensions to surface in feedback
TOP_WEAK_DIMS = 2
# Max failure examples per dimension
MAX_EXAMPLES = 3
# Max characters per failure example (trim long justifications)
MAX_EXAMPLE_CHARS = 120
# Minimum rejected items in a topic to attempt weakness analysis
MIN_REJECTED = 3


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_judge_cache(category: str) -> dict[str, dict]:
    """Return {question_id: record} from the judge scoring cache."""
    path = SCORES_DIR / f"{category}_cache.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_regen_report() -> dict[str, dict]:
    """
    Return {(category, topic): entry} from regeneration_needed.json.
    Returns {} if the report doesn't exist yet.
    """
    path = FINAL_DIR / "regeneration_needed.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for entry in report.get("topics", []):
        key = (entry["category"], entry["topic"])
        result[key] = entry
    return result


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _dim_avg(rejected_items: list[dict], dimension: str) -> float | None:
    """Compute mean score for one dimension across rejected items."""
    scores = []
    for rec in rejected_items:
        qs = rec.get("quality_scores", {})
        val = qs.get(dimension)
        if val is not None:
            try:
                scores.append(float(val))
            except (TypeError, ValueError):
                continue  # skip non-numeric score values
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def _failure_examples(
    rejected_items: list[dict],
    dimension: str,
    n: int = MAX_EXAMPLES,
) -> list[str]:
    """
    Pull the worst-scoring items for a dimension and return their
    judge justification texts (truncated).
    """
    scored = []
    for rec in rejected_items:
        qs = rec.get("quality_scores", {})
        score = qs.get(dimension)
        justif = (rec.get("quality_justifications") or {}).get(dimension, "")
        if score is not None and justif:
            scored.append((float(score), justif))

    # Sort by score ascending (worst first), then take unique snippets
    scored.sort(key=lambda x: x[0])
    examples = []
    seen: set[str] = set()
    for _, text in scored:
        snippet = text.strip()[:MAX_EXAMPLE_CHARS]
        if snippet not in seen:
            seen.add(snippet)
            examples.append(snippet)
        if len(examples) >= n:
            break
    return examples


def _aggregate_topic(
    category: str,
    topic: str,
    rejected_items: list[dict],
    regen_entry: dict,
) -> dict:
    """Build the feedback record for one topic."""
    dim_avgs = {}
    for dim in DIMENSIONS:
        avg = _dim_avg(rejected_items, dim)
        if avg is not None:
            dim_avgs[dim] = avg

    # Rank dimensions by avg score ascending (weakest first)
    ranked = sorted(dim_avgs.items(), key=lambda x: x[1])
    weakest = ranked[:TOP_WEAK_DIMS]

    weakest_dims = []
    for dim, avg in weakest:
        weakest_dims.append({
            "dimension": dim,
            "avg_score": avg,
            "failure_examples": _failure_examples(rejected_items, dim),
        })

    # Common rejection reasons: pull justifications for weakest dims from worst items
    common_reasons: list[str] = []
    seen_reasons: set[str] = set()
    for dim, _ in weakest:
        for rec in rejected_items:
            qs = rec.get("quality_scores", {})
            score = qs.get(dim)
            if score is not None and float(score) <= 2.5:
                justif = (rec.get("quality_justifications") or {}).get(dim, "")
                snippet = justif.strip()[:MAX_EXAMPLE_CHARS]
                if snippet and snippet not in seen_reasons:
                    seen_reasons.add(snippet)
                    common_reasons.append(snippet)
                if len(common_reasons) >= 4:
                    break
        if len(common_reasons) >= 4:
            break

    return {
        "category": category,
        "items_needed": regen_entry["items_needed"],
        "items_surviving": regen_entry["items_surviving"],
        "total_rejected": len(rejected_items),
        "weakest_dimensions": weakest_dims,
        "common_rejection_reasons": common_reasons,
    }


# ---------------------------------------------------------------------------
# Core: aggregate one category
# ---------------------------------------------------------------------------


def aggregate_category(category: str, dry_run: bool) -> dict:
    """
    Aggregate feedback for all topics in a category that need regeneration.
    Returns the feedback_context dict (may be empty if nothing needs regen).
    """
    cache = _load_judge_cache(category)
    if not cache:
        print(f"  WARNING: no judge cache found for '{category}'", file=sys.stderr)
        return {}

    regen_map = _load_regen_report()
    topics_needing_regen = {
        topic: entry
        for (cat, topic), entry in regen_map.items()
        if cat == category
    }

    if not topics_needing_regen:
        print(f"  No regeneration needed for '{category}' (or regen report missing).")
        return {}

    # Group rejected items by topic
    rejected_by_topic: dict[str, list[dict]] = defaultdict(list)
    for rec in cache.values():
        verdict = rec.get("quality_scores", {}).get("verdict")
        if verdict == "reject":
            topic = rec.get("topic", "unknown")
            rejected_by_topic[topic].append(rec)

    feedback_context: dict[str, dict] = {}

    for topic, regen_entry in sorted(topics_needing_regen.items()):
        rejected = rejected_by_topic.get(topic, [])
        n_rej = len(rejected)
        n_need = regen_entry["items_needed"]

        if dry_run:
            print(
                f"  [dry-run] {topic:<45} "
                f"rejected={n_rej}  needs={n_need}"
            )
            continue

        if n_rej < MIN_REJECTED:
            print(
                f"  WARNING: '{topic}' has only {n_rej} rejected items — "
                f"skipping weakness analysis (min={MIN_REJECTED})"
            )
            feedback_context[topic] = {
                "category": category,
                "items_needed": n_need,
                "items_surviving": regen_entry["items_surviving"],
                "total_rejected": n_rej,
                "weakest_dimensions": [],
                "common_rejection_reasons": [],
            }
            continue

        feedback_context[topic] = _aggregate_topic(
            category, topic, rejected, regen_entry
        )

    return feedback_context


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate judge feedback for MCQ regeneration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--category",
        metavar="NAME",
        help="Category to aggregate (e.g. natural_fertilizers)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Aggregate all categories present in regeneration_needed.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without writing files",
    )
    args = parser.parse_args()

    regen_map = _load_regen_report()
    if not regen_map and not args.dry_run:
        print(
            "ERROR: quality/final/regeneration_needed.json not found. "
            "Run the full quality pipeline first.",
            file=sys.stderr,
        )
        return 1

    if args.all:
        categories = sorted({cat for (cat, _) in regen_map})
    else:
        categories = [args.category]

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    for category in categories:
        print(f"\n{category}")
        print("-" * 60)
        feedback = aggregate_category(category, args.dry_run)
        if args.dry_run or not feedback:
            continue

        out_path = FEEDBACK_DIR / f"{category}_feedback_context.json"
        out_path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")

        print(f"  Topics with feedback: {len(feedback)}")
        for topic, entry in sorted(feedback.items()):
            weak_dims = ", ".join(
                f"{d['dimension']} ({d['avg_score']:.1f})"
                for d in entry["weakest_dimensions"]
            )
            print(
                f"  {topic:<45} "
                f"needs={entry['items_needed']}  "
                f"weakest: {weak_dims or 'n/a'}"
            )
        print(f"  → {out_path.relative_to(WORKSPACE)}")

    if args.dry_run:
        print("\nDry run complete — no files written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
