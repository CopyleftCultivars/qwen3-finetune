#!/usr/bin/env python3
"""
Final benchmark packaging and reporting script (Issue #59).

Merges all quality-pipeline metadata into a single per-item final JSONL,
computes survival rates per topic, and outputs a regeneration report for
topics where too few items survived the quality filters.

Metadata sources merged per item
---------------------------------
- difficulty         (from benchmarks/{cat}/{cat}_grading_cache.json)
- quality_scores     (from quality/scores/{cat}_judged.jsonl)
- verdict            (keep / review / reject from LLM judge)
- irt_disc, irt_diff (from quality/irt/irt_params.json)
- irt_removed        (from quality/irt/filter_report.json)
- confidence, variability, region  (from quality/cartography/cartography_map.jsonl)
- layer5_flags       (from quality/cleanlab/flagged_items.jsonl)

Final status per item
---------------------
keep    Passes all automated quality checks; no Layer 5 flags.
review  Flagged for human review (Layer 5 flag OR judge verdict = review).
reject  Judge verdict = reject, OR IRT removed (low discrimination / ceiling).
floor   IRT floor item (all models wrong → possible label error).

Regeneration report
-------------------
For each topic with fewer than MIN_ITEMS_PER_TOPIC surviving items (keep +
review), a regeneration request is written to
    quality/final/regeneration_needed.json

This file can be inspected to determine which topics need additional MCQs
generated, and then fed back into generate_mcqs_rag.py.

Usage
-----
    # Package a single category
    conda run -n nf-benchmark python quality/package_benchmark.py \\
        --category natural_fertilizers

    # Package all categories
    conda run -n nf-benchmark python quality/package_benchmark.py --all

    # Show what would be merged without writing files
    conda run -n nf-benchmark python quality/package_benchmark.py \\
        --category natural_fertilizers --dry-run

Output
------
    quality/final/{category}_final.jsonl       final items with all metadata
    quality/final/{category}_summary.json      per-category stats
    quality/final/regeneration_needed.json     topics below MIN_ITEMS_PER_TOPIC
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
BENCHMARKS_DIR = WORKSPACE / "benchmarks"
QUALITY_DIR = WORKSPACE / "quality"
FINAL_DIR = WORKSPACE / "quality" / "final"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_ITEMS_PER_TOPIC = 10   # topics below this after filtering get flagged for regen


# ---------------------------------------------------------------------------
# Loaders — each returns {} / [] when source file not yet available
# ---------------------------------------------------------------------------


def _load_jsonl_by_qid(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            result[item["question_id"]] = item
    return result


def _load_grading_cache(category: str) -> dict[str, dict]:
    path = BENCHMARKS_DIR / category / f"{category}_grading_cache.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_judge_scores(category: str) -> dict[str, dict]:
    return _load_jsonl_by_qid(QUALITY_DIR / "scores" / f"{category}_judged.jsonl")


def _load_irt_params(category: str) -> dict[str, dict]:
    """Return {question_id: {disc, diff}} from irt_params.json."""
    params_path = QUALITY_DIR / "irt" / "irt_params.json"
    if not params_path.exists():
        return {}
    params = json.loads(params_path.read_text(encoding="utf-8"))
    ix_to_item = params.get("item_ids", {})
    disc_list = params.get("disc", [])
    diff_list = params.get("diff", [])
    result = {}
    for ix_str, qid in ix_to_item.items():
        ix = int(ix_str)
        result[qid] = {
            "irt_disc": disc_list[ix] if ix < len(disc_list) else None,
            "irt_diff": diff_list[ix] if ix < len(diff_list) else None,
        }
    return result


def _load_irt_removed(category: str) -> dict[str, str]:
    """Return {question_id: irt_reason} for items removed by IRT filter."""
    report_path = QUALITY_DIR / "irt" / "filter_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    removed = {}
    for item in report.get("items", []):
        if item.get("decision") == "remove":
            removed[item["question_id"]] = item.get("reason", "irt_removed")
        elif item.get("decision") == "flag_floor":
            removed[item["question_id"]] = "floor"
    return removed


def _load_cartography(category: str) -> dict[str, dict]:
    return _load_jsonl_by_qid(QUALITY_DIR / "cartography" / "cartography_map.jsonl")


def _load_layer5_flags(category: str) -> dict[str, str]:
    """Return {question_id: flag_source} for all Layer 5 flagged items."""
    path = QUALITY_DIR / "cleanlab" / "flagged_items.jsonl"
    if not path.exists():
        return {}
    flags = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            flags[item["question_id"]] = item.get("flag_source", "layer5")
    return flags


# ---------------------------------------------------------------------------
# Final status logic
# ---------------------------------------------------------------------------


def compute_final_status(
    qid: str,
    judge_rec: dict | None,
    irt_removed: dict[str, str],
    layer5_flags: dict[str, str],
) -> tuple[str, list[str]]:
    """
    Compute final_status and reasons list for one item.

    Returns (status, [reason, ...])
    """
    reasons = []

    # IRT floor → special status (possible label error)
    irt_reason = irt_removed.get(qid)
    if irt_reason == "floor":
        return "floor", ["irt_floor"]

    # IRT remove (ceiling or low discrimination)
    if irt_reason:
        return "reject", [irt_reason]

    # LLM judge verdict
    verdict = (judge_rec or {}).get("quality_scores", {}).get("verdict") or \
              (judge_rec or {}).get("verdict")
    if verdict == "reject":
        return "reject", ["judge_reject"]

    # Layer 5 flag → review
    layer5_src = layer5_flags.get(qid)
    if layer5_src:
        reasons.append(layer5_src)

    # Judge gray-zone → review
    if verdict == "review":
        reasons.append("judge_review")

    if reasons:
        return "review", reasons

    return "keep", []


# ---------------------------------------------------------------------------
# Package one category
# ---------------------------------------------------------------------------


def package_category(category: str, dry_run: bool) -> dict:
    benchmark_path = BENCHMARKS_DIR / category / f"{category}_benchmark_v1.jsonl"
    if not benchmark_path.exists():
        print(f"  ERROR: benchmark not found: {benchmark_path}", file=sys.stderr)
        return {}

    # Load all metadata sources
    grading = _load_grading_cache(category)
    judge = _load_judge_scores(category)
    irt_params = _load_irt_params(category)
    irt_removed = _load_irt_removed(category)
    cartography = _load_cartography(category)
    layer5 = _load_layer5_flags(category)

    items = []
    for line in benchmark_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))

    status_counts: dict[str, int] = defaultdict(int)
    topic_surviving: dict[str, list[str]] = defaultdict(list)  # topic → [qids]
    topic_original: dict[str, int] = defaultdict(int)
    final_items = []

    for item in items:
        qid = item["question_id"]
        topic = item.get("topic", "unknown")
        topic_original[topic] += 1

        merged = dict(item)

        # Difficulty
        grade = grading.get(qid, {})
        if grade:
            merged["difficulty"] = grade.get("difficulty")
            merged["strong_answer"] = grade.get("strong_answer")
            merged["weak_answer"] = grade.get("weak_answer")

        # LLM judge
        judge_rec = judge.get(qid)
        if judge_rec:
            qs = judge_rec.get("quality_scores", {})
            merged["quality_scores"] = qs.get("scores") if isinstance(qs, dict) else qs
            merged["quality_composite"] = (
                qs.get("composite") if isinstance(qs, dict) else None
            )
            merged["judge_verdict"] = judge_rec.get("verdict") or (
                qs.get("verdict") if isinstance(qs, dict) else None
            )

        # IRT
        irt = irt_params.get(qid, {})
        if irt:
            merged["irt_disc"] = irt.get("irt_disc")
            merged["irt_diff"] = irt.get("irt_diff")

        # Cartography
        cart = cartography.get(qid, {})
        if cart:
            merged["cart_confidence"] = cart.get("confidence")
            merged["cart_variability"] = cart.get("variability")
            merged["cart_region"] = cart.get("region")

        # Final status
        final_status, reasons = compute_final_status(qid, judge_rec, irt_removed, layer5)
        merged["final_status"] = final_status
        merged["quality_reasons"] = reasons

        status_counts[final_status] += 1
        if final_status in ("keep", "review"):
            topic_surviving[topic].append(qid)

        final_items.append(merged)

    # Regeneration report for this category
    regen_topics = []
    for topic, orig_count in sorted(topic_original.items()):
        surviving = len(topic_surviving.get(topic, []))
        if surviving < MIN_ITEMS_PER_TOPIC:
            regen_topics.append({
                "category": category,
                "topic": topic,
                "items_original": orig_count,
                "items_surviving": surviving,
                "items_needed": MIN_ITEMS_PER_TOPIC - surviving,
            })

    summary = {
        "category": category,
        "total": len(items),
        "by_status": dict(status_counts),
        "survival_rate": round(
            (status_counts["keep"] + status_counts["review"]) / max(1, len(items)), 4
        ),
        "topics_needing_regeneration": len(regen_topics),
        "metadata_sources": {
            "grading": bool(grading),
            "judge": bool(judge),
            "irt_params": bool(irt_params),
            "irt_filter": bool(irt_removed),
            "cartography": bool(cartography),
            "layer5_flags": bool(layer5),
        },
    }

    if dry_run:
        print(f"  {category}: {len(items)} items — would write final JSONL")
        for src, avail in summary["metadata_sources"].items():
            mark = "✓" if avail else "✗ (not yet run)"
            print(f"    {src:<15s} {mark}")
        return {"summary": summary, "regen_topics": regen_topics, "final_items": []}

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # Write final JSONL
    out_path = FINAL_DIR / f"{category}_final.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in final_items:
            fh.write(json.dumps(item) + "\n")

    # Write summary
    summary_path = FINAL_DIR / f"{category}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    keep = status_counts["keep"]
    review = status_counts["review"]
    reject = status_counts["reject"]
    floor = status_counts["floor"]
    total = len(items)

    print(f"  {category}")
    print(f"    keep:   {keep:4d}  ({keep / total * 100:.1f}%)")
    print(f"    review: {review:4d}  ({review / total * 100:.1f}%)")
    print(f"    reject: {reject:4d}  ({reject / total * 100:.1f}%)")
    print(f"    floor:  {floor:4d}  ({floor / total * 100:.1f}%)")
    print(f"    survival rate: {summary['survival_rate'] * 100:.1f}%")
    if regen_topics:
        print(f"    topics needing regeneration: {len(regen_topics)}")
    print(f"    → {out_path.relative_to(WORKSPACE)}")

    return {"summary": summary, "regen_topics": regen_topics, "final_items": final_items}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package final benchmark with all quality metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--category",
        metavar="NAME",
        help="Package a single category (e.g. natural_fertilizers)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Package all benchmark categories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing files",
    )
    args = parser.parse_args()

    if args.all:
        categories = sorted(
            d.name for d in BENCHMARKS_DIR.iterdir()
            if d.is_dir() and (d / f"{d.name}_benchmark_v1.jsonl").exists()
        )
    else:
        categories = [args.category]

    all_regen: list[dict] = []
    all_summaries: list[dict] = []

    for category in categories:
        result = package_category(category, args.dry_run)
        if result:
            all_regen.extend(result.get("regen_topics", []))
            if result.get("summary"):
                all_summaries.append(result["summary"])

    if not args.dry_run and all_regen:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        regen_path = FINAL_DIR / "regeneration_needed.json"
        regen_path.write_text(
            json.dumps(
                {
                    "min_items_per_topic": MIN_ITEMS_PER_TOPIC,
                    "total_topics_flagged": len(all_regen),
                    "topics": all_regen,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nRegeneration report: {regen_path.relative_to(WORKSPACE)}")
        print(f"  {len(all_regen)} topic(s) need additional MCQs")
        for entry in all_regen:
            print(
                f"  {entry['category']:30s} / {entry['topic']:40s} "
                f"→ needs {entry['items_needed']} more items "
                f"({entry['items_surviving']}/{entry['items_original']} survived)"
            )
    elif not args.dry_run:
        print("\nAll topics have sufficient surviving items — no regeneration needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
