#!/usr/bin/env python3
"""
Merge regenerated MCQs from one round into the main benchmark JSONL (Step 3).

Reads all per-topic JSON files from
    regenerated_mcqs/{category}_round{round}/
and appends them (flattened to canonical schema) to
    benchmarks/{category}/{category}_benchmark_v1.jsonl

Skips malformed questions (missing options, bad correct_answer) with a warning
rather than aborting, so a single bad item doesn't block the whole merge.

Usage
-----
    conda run -n nf-benchmark python quality/merge_regen_round.py \\
        --category natural_fertilizers \\
        --round 3 \\
        --prefix NF

    # Dry-run: show what would be merged without writing
    conda run -n nf-benchmark python quality/merge_regen_round.py \\
        --category natural_fertilizers \\
        --round 3 \\
        --prefix NF \\
        --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
REGEN_DIR = WORKSPACE / "regenerated_mcqs"
BENCH_DIR = WORKSPACE / "benchmarks"

_LETTERS = ["A", "B", "C", "D"]


def _next_question_num(items: list[dict], prefix: str) -> int:
    """Return the next sequential question number for the given prefix."""
    nums = []
    for item in items:
        qid = item.get("question_id", "")
        if qid.startswith(f"{prefix}-"):
            try:
                nums.append(int(qid.split("-", 1)[1]))
            except ValueError:
                continue
    return max(nums) + 1 if nums else 1


def _flatten_question(q: dict, question_id: str) -> dict | None:
    """
    Convert a normalized question dict to the flat benchmark schema.
    Returns None (with a warning printed) if the question is unusable.
    """
    opts = q.get("options", {})
    if not isinstance(opts, dict) or not all(k in opts for k in _LETTERS):
        print(f"  [skip] {question_id}: options missing A/B/C/D — {list(opts.keys())}", file=sys.stderr)
        return None

    ans = str(q.get("correct_answer", "")).strip()
    if ans not in _LETTERS:
        print(f"  [skip] {question_id}: correct_answer {ans!r} not in A/B/C/D", file=sys.stderr)
        return None

    question_text = str(q.get("question", "")).strip()
    if not question_text:
        print(f"  [skip] {question_id}: empty question stem", file=sys.stderr)
        return None

    return {
        "question":        question_text,
        "A":               opts["A"],
        "B":               opts["B"],
        "C":               opts["C"],
        "D":               opts["D"],
        "answer":          ans,
        "question_id":     question_id,
        "topic":           str(q.get("topic", "")),
        "source_document": str(q.get("source_passage") or q.get("source_document", "")),
    }


def merge_round(
    category: str,
    round_num: int,
    prefix: str,
    dry_run: bool = False,
) -> int:
    """Merge one round's output into the benchmark JSONL. Returns number of items merged."""
    regen_dir = REGEN_DIR / f"{category}_round{round_num}"
    bench_path = BENCH_DIR / category / f"{category}_benchmark_v1.jsonl"

    if not regen_dir.exists():
        print(f"ERROR: regen dir not found: {regen_dir}", file=sys.stderr)
        return -1

    if not bench_path.exists():
        print(f"ERROR: benchmark JSONL not found: {bench_path}", file=sys.stderr)
        return -1

    # Load existing items to determine next ID
    existing = [json.loads(line) for line in bench_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    next_num = _next_question_num(existing, prefix)

    topic_files = sorted(f for f in regen_dir.glob("*.json") if f.stem != "round_stats")
    if not topic_files:
        print(f"WARNING: no topic JSON files found in {regen_dir}", file=sys.stderr)
        return 0

    new_items: list[dict] = []
    skipped = 0

    for topic_file in topic_files:
        try:
            data = json.loads(topic_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {topic_file.name}: could not read — {e}", file=sys.stderr)
            skipped += 1
            continue

        for q in data.get("questions", []):
            qid = f"{prefix}-{next_num:04d}"
            flat = _flatten_question(q, qid)
            if flat is None:
                skipped += 1
                continue
            new_items.append(flat)
            next_num += 1

    if dry_run:
        print(f"[dry-run] Would merge {len(new_items)} items from round {round_num} ({skipped} skipped)")
        for item in new_items:
            print(f"  {item['question_id']}  [{item['topic']}]  {item['question'][:70]}...")
        return len(new_items)

    if new_items:
        with bench_path.open("a", encoding="utf-8") as f:
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Merged {len(new_items)} items from round {round_num} into {bench_path.name}"
          + (f" ({skipped} skipped)" if skipped else ""))
    return len(new_items)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge one regeneration round into the benchmark JSONL (pipeline step 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--category", required=True, metavar="NAME",
                        help="Category slug (e.g. natural_fertilizers)")
    parser.add_argument("--round", required=True, type=int, metavar="N",
                        help="Round number to merge (e.g. 3 to merge _round3/ output)")
    parser.add_argument("--prefix", required=True, metavar="PFX",
                        help="Question ID prefix (e.g. NF)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be merged without writing anything")
    args = parser.parse_args()

    result = merge_round(args.category, args.round, args.prefix, dry_run=args.dry_run)
    return 0 if result >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
