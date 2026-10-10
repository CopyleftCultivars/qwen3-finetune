#!/usr/bin/env python3
"""Filter freshly generated MCQs for overlap with benchmark_combined.jsonl.

Run this after generate_mcqs_rag.py's "generate" mode and before quality/llm_judge.py —
no point spending judge API calls scoring an item that gets discarded anyway for being
too similar to an eval question. See MCQ_DIAGNOSTIC_PLAN.md Phase 3 for why this exists:
the original v2/v3 fine-tunes were trained directly on benchmark_combined.jsonl, and this
script is the safety net that should have existed from day one.

Fuzzy match (difflib.SequenceMatcher ratio on lowercased question text), not just exact
match — mirrors the threshold style already used by generate_mcqs_rag.py's own
_is_duplicate() for within-category dedup (0.75 for a strict "too similar" bar).

Usage:
    python dedup_against_benchmark.py \\
        --input-dir <output_dir from generate_mcqs_rag.py, per-topic *.json files> \\
        --benchmark "$BENCHMARK_ROOT" \\
        [--threshold 0.75] \\
        [--report dedup_report.json]

Modifies each {topic}.json in place: any question exceeding the similarity threshold is
moved from "questions" into a new "questions_rejected_benchmark_overlap" list (with the
matched benchmark question_id and ratio recorded), not silently deleted — so a human can
audit what got cut.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.75


def load_benchmark_stems(benchmark_path: Path) -> list[tuple[str, str]]:
    """Return [(question_id, lowercased question text), ...]."""
    stems: list[tuple[str, str]] = []
    with open(benchmark_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("question", "").strip()
            if q:
                stems.append((rec.get("question_id", "?"), q.lower()))
    return stems


def best_match(question_text: str, stems: list[tuple[str, str]]) -> tuple[str, float]:
    """Return (best_matching_question_id, best_ratio) against the benchmark stems."""
    q = question_text.lower()
    best_id, best_ratio = "", 0.0
    for qid, stem in stems:
        ratio = difflib.SequenceMatcher(None, q, stem).ratio()
        if ratio > best_ratio:
            best_id, best_ratio = qid, ratio
    return best_id, best_ratio


def process_file(path: Path, stems: list[tuple[str, str]], threshold: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    kept: list[dict] = []
    rejected: list[dict] = []

    for q in data.get("questions", []):
        question_text = q.get("question", "")
        if not question_text:
            kept.append(q)
            continue
        match_id, ratio = best_match(question_text, stems)
        if ratio >= threshold:
            rejected.append({**q, "_benchmark_match_id": match_id, "_benchmark_match_ratio": round(ratio, 4)})
        else:
            kept.append(q)

    data["questions"] = kept
    if rejected:
        data["questions_rejected_benchmark_overlap"] = rejected
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "file": path.name,
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_details": [
            {"question": r["question"][:100], "matched_id": r["_benchmark_match_id"], "ratio": r["_benchmark_match_ratio"]}
            for r in rejected
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path,
                     help="Directory of per-topic *.json files from generate_mcqs_rag.py")
    ap.add_argument("--benchmark", required=True, type=Path,
                     help="Path to benchmark_combined.jsonl")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                     help=f"Similarity ratio at/above which a question is rejected (default: {DEFAULT_THRESHOLD})")
    ap.add_argument("--report", type=Path, default=None,
                     help="Optional path to write a JSON summary report")
    args = ap.parse_args()

    if not args.benchmark.exists():
        print(f"error: benchmark file not found: {args.benchmark}", file=sys.stderr)
        sys.exit(1)
    if not args.input_dir.is_dir():
        print(f"error: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    stems = load_benchmark_stems(args.benchmark)
    print(f"Loaded {len(stems)} benchmark question stems from {args.benchmark}", file=sys.stderr)

    topic_files = sorted(args.input_dir.glob("*.json"))
    print(f"Processing {len(topic_files)} topic files in {args.input_dir}", file=sys.stderr)

    results = []
    total_kept = total_rejected = 0
    for path in topic_files:
        result = process_file(path, stems, args.threshold)
        results.append(result)
        total_kept += result["kept"]
        total_rejected += result["rejected"]
        if result["rejected"]:
            print(f"  {path.name}: kept {result['kept']}, REJECTED {result['rejected']} (benchmark overlap)", file=sys.stderr)

    print(f"\nTotal: {total_kept} kept, {total_rejected} rejected for benchmark overlap "
          f"(threshold={args.threshold})", file=sys.stderr)

    if args.report:
        args.report.write_text(json.dumps({
            "threshold": args.threshold,
            "total_kept": total_kept,
            "total_rejected": total_rejected,
            "per_file": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
