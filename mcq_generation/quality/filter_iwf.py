#!/usr/bin/env python3
"""
IWF (Item Writing Flaw) structural filter for MCQ benchmarks.

Applies rule-based checks derived from Haladyna, Downing & Rodriguez (2002)
to flag or reject structurally flawed items before any LLM or human review.

Decision rule (from quality_assessment_plan.md):
  0–1 flaws  → ACCEPT
  2+ flaws   → REJECT (written to rejected output)

Stem-level checks:
  source_reference    Stem begins with "According to…", "Based on…", etc.
  fill_in_blank       Stem contains blank placeholder ("___")
  negative_wording    Stem uses NOT/EXCEPT as the discriminating cue
  none_all_above      Any option is "None of the above" / "All of the above"

Option-level checks:
  absolute_terms      An option contains always/never as a standalone word
  vague_terms         An option contains usually/sometimes/often/generally
  longest_correct     Correct answer is notably longer than all distractors
  compound_options    An option uses "both…and" / "A and B" phrasing

Usage:
  python quality/filter_iwf.py                    # all benchmarks/
  python quality/filter_iwf.py --file FILE.jsonl
  python quality/filter_iwf.py --out-dir quality/iwf_output/
  python quality/filter_iwf.py --json-report quality/iwf_report.json

Outputs (written to --out-dir, default quality/iwf_output/):
  <category>_accepted.jsonl   items with 0–1 flaws
  <category>_rejected.jsonl   items with 2+ flaws (include "iwf_flaws" key)

Exits 0 always (filtering is non-destructive; inspect reports before discarding).
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

SOURCE_REF_PATTERNS = re.compile(
    r"^\s*(according\s+to|based\s+on(\s+the)?\s+(text|passage|document|excerpt|reading)|"
    r"the\s+(passage|document|text|excerpt)\s+(states|says|mentions|indicates|shows|describes)|"
    r"as\s+(stated|described|mentioned)\s+in(\s+the)?)",
    re.IGNORECASE,
)

FILL_IN_BLANK_RE = re.compile(r"_{2,}")  # two or more underscores

NEGATIVE_WORDING_RE = re.compile(
    r"\b(NOT|EXCEPT)\b|"
    r"\b(which|that|what)\s+(is|are|does|do|was|were)\s+NOT\b|"
    r"\b(does|do|is|are)\s+NOT\b",
    re.IGNORECASE,
)
# Only flag if the negative word appears to be load-bearing (in the latter half
# of the stem or near "?") — simple heuristic: if stem is all-caps NOT that's ok,
# but caps NOT mid-sentence usually signals the flaw.

ABSOLUTE_TERMS_RE = re.compile(r"\b(always|never)\b", re.IGNORECASE)
VAGUE_TERMS_RE = re.compile(
    r"\b(usually|sometimes|often|generally|typically|may|might|could|can)\b",
    re.IGNORECASE,
)
NONE_ALL_ABOVE_RE = re.compile(
    r"\b(none|all)\s+of\s+(the\s+)?(above|below)\b", re.IGNORECASE
)
COMPOUND_OPTIONS_RE = re.compile(
    r"\bboth\b.+\band\b|\b(A\s+and\s+[BCD]|B\s+and\s+[ACD]|[ABC]\s+and\s+[BCD])\b|"
    r"\b(i\s+and\s+ii|a\s+and\s+b)\b",
    re.IGNORECASE,
)

LONGEST_CORRECT_RATIO = 1.5  # correct must be > 1.5× avg distractor length to flag


# ---------------------------------------------------------------------------
# Per-item check functions (each returns flaw name or None)
# ---------------------------------------------------------------------------

def check_source_reference(item: dict) -> str | None:
    q = item.get("question", "")
    return "source_reference" if SOURCE_REF_PATTERNS.match(q) else None


def check_fill_in_blank(item: dict) -> str | None:
    q = item.get("question", "")
    return "fill_in_blank" if FILL_IN_BLANK_RE.search(q) else None


def check_negative_wording(item: dict) -> str | None:
    q = item.get("question", "")
    return "negative_wording" if NEGATIVE_WORDING_RE.search(q) else None


def check_none_all_above(item: dict) -> str | None:
    for key in ("A", "B", "C", "D"):
        val = item.get(key, "")
        if isinstance(val, str) and NONE_ALL_ABOVE_RE.search(val):
            return "none_all_above"
    return None


def check_absolute_terms(item: dict) -> str | None:
    for key in ("A", "B", "C", "D"):
        val = item.get(key, "")
        if isinstance(val, str) and ABSOLUTE_TERMS_RE.search(val):
            return "absolute_terms"
    return None


def check_vague_terms(item: dict) -> str | None:
    for key in ("A", "B", "C", "D"):
        val = item.get(key, "")
        if isinstance(val, str) and VAGUE_TERMS_RE.search(val):
            return "vague_terms"
    return None


def check_longest_correct(item: dict) -> str | None:
    answer = item.get("answer")
    if answer not in {"A", "B", "C", "D"}:
        return None
    correct_text = item.get(answer, "")
    if not isinstance(correct_text, str):
        return None
    distractors = [
        item.get(k, "")
        for k in ("A", "B", "C", "D")
        if k != answer and isinstance(item.get(k), str)
    ]
    if not distractors:
        return None
    avg_distractor = sum(len(d) for d in distractors) / len(distractors)
    if avg_distractor == 0:
        return None
    if len(correct_text) > LONGEST_CORRECT_RATIO * avg_distractor:
        return "longest_correct"
    return None


def check_compound_options(item: dict) -> str | None:
    for key in ("A", "B", "C", "D"):
        val = item.get(key, "")
        if isinstance(val, str) and COMPOUND_OPTIONS_RE.search(val):
            return "compound_options"
    return None


CHECKS = [
    check_source_reference,
    check_fill_in_blank,
    check_negative_wording,
    check_none_all_above,
    check_absolute_terms,
    check_vague_terms,
    check_longest_correct,
    check_compound_options,
]


# ---------------------------------------------------------------------------
# Item-level filtering
# ---------------------------------------------------------------------------

MAX_FLAWS_ACCEPTED = 1  # items with <= this many flaws are accepted


def apply_iwf_checks(item: dict) -> list[str]:
    """Return list of flaw names found in item."""
    flaws = []
    for check in CHECKS:
        flaw = check(item)
        if flaw:
            flaws.append(flaw)
    return flaws


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

@dataclass
class FileStats:
    path: Path
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    flaw_counts: dict = field(default_factory=dict)


def process_file(path: Path, out_dir: Path, max_flaws: int = MAX_FLAWS_ACCEPTED) -> FileStats:
    stats = FileStats(path=path)
    stem = path.stem  # e.g. "natural_fertilizers_benchmark_v1"
    accepted_path = out_dir / f"{stem}_accepted.jsonl"
    rejected_path = out_dir / f"{stem}_rejected.jsonl"

    accepted_lines: list[str] = []
    rejected_lines: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue  # schema validator catches this; skip here

        flaws = apply_iwf_checks(item)
        stats.total += 1

        for flaw in flaws:
            stats.flaw_counts[flaw] = stats.flaw_counts.get(flaw, 0) + 1

        if len(flaws) <= max_flaws:
            stats.accepted += 1
            accepted_lines.append(json.dumps(item, ensure_ascii=False))
        else:
            stats.rejected += 1
            item_out = dict(item)
            item_out["iwf_flaws"] = flaws
            rejected_lines.append(json.dumps(item_out, ensure_ascii=False))

    accepted_path.write_text("\n".join(accepted_lines) + "\n", encoding="utf-8")
    rejected_path.write_text("\n".join(rejected_lines) + "\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_file_stats(stats: FileStats) -> None:
    pct = 100 * stats.accepted / stats.total if stats.total else 0
    try:
        rel = stats.path.relative_to(Path.cwd())
    except ValueError:
        rel = stats.path
    print(
        f"  {rel}\n"
        f"    Total: {stats.total}  |  Accepted: {stats.accepted} ({pct:.1f}%)  |  Rejected: {stats.rejected}"
    )
    if stats.flaw_counts:
        for flaw, count in sorted(stats.flaw_counts.items(), key=lambda x: -x[1]):
            print(f"    [{flaw}] {count}")


def write_json_report(path: Path, all_stats: list[FileStats]) -> None:
    data = {
        "summary": {
            "total": sum(s.total for s in all_stats),
            "accepted": sum(s.accepted for s in all_stats),
            "rejected": sum(s.rejected for s in all_stats),
        },
        "files": [
            {
                "path": str(s.path),
                "total": s.total,
                "accepted": s.accepted,
                "rejected": s.rejected,
                "flaw_counts": s.flaw_counts,
            }
            for s in all_stats
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="IWF structural filter for MCQ benchmark JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path(__file__).parent.parent / "benchmarks",
        help="Root benchmarks directory (default: benchmarks/)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        dest="files",
        metavar="JSONL",
        help="Filter a specific JSONL file (repeatable; overrides --benchmarks-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "iwf_output",
        help="Directory for accepted/rejected output files (default: quality/iwf_output/)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        metavar="PATH",
        help="Write structured JSON report to this file",
    )
    parser.add_argument(
        "--max-flaws",
        type=int,
        default=MAX_FLAWS_ACCEPTED,
        metavar="N",
        help=f"Max flaws for acceptance (default: {MAX_FLAWS_ACCEPTED})",
    )
    args = parser.parse_args()

    max_flaws = args.max_flaws

    if args.files:
        jsonl_files = sorted(args.files)
    else:
        if not args.benchmarks_dir.exists():
            print(f"Benchmarks directory not found: {args.benchmarks_dir}", file=sys.stderr)
            return 1
        jsonl_files = sorted(args.benchmarks_dir.rglob("*_benchmark_v1.jsonl"))

    if not jsonl_files:
        print("No JSONL benchmark files found.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[FileStats] = []
    print(f"IWF filter  (max_flaws={max_flaws}, out={args.out_dir})\n")

    for path in jsonl_files:
        stats = process_file(path, args.out_dir, max_flaws=max_flaws)
        print_file_stats(stats)
        all_stats.append(stats)

    total = sum(s.total for s in all_stats)
    accepted = sum(s.accepted for s in all_stats)
    rejected = sum(s.rejected for s in all_stats)
    pct = 100 * accepted / total if total else 0

    # Aggregate flaw counts across all files
    aggregate_flaws: dict[str, int] = {}
    for s in all_stats:
        for flaw, count in s.flaw_counts.items():
            aggregate_flaws[flaw] = aggregate_flaws.get(flaw, 0) + count

    print(f"\n{'=' * 60}")
    print(f"Total items : {total}")
    print(f"Accepted    : {accepted} ({pct:.1f}%)")
    print(f"Rejected    : {rejected} ({100 - pct:.1f}%)")
    if aggregate_flaws:
        print("\nFlaw breakdown (items may have multiple flaws):")
        for flaw, count in sorted(aggregate_flaws.items(), key=lambda x: -x[1]):
            print(f"  {flaw:<25}  {count}")

    if args.json_report:
        write_json_report(args.json_report, all_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
