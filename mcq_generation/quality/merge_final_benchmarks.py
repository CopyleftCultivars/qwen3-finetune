#!/usr/bin/env python3
"""
Merge all per-category final JSONL files into one comprehensive benchmark.

Reads quality/final/{category}_final.jsonl for every category, keeps items
with final_status in ("keep", "review"), and writes the result to
benchmarks/comprehensive/comprehensive_agri_benchmark_v2.jsonl.

Usage
-----
    conda run -n nf-benchmark python quality/merge_final_benchmarks.py

    # Dry-run: show counts without writing
    conda run -n nf-benchmark python quality/merge_final_benchmarks.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
FINAL_DIR = WORKSPACE / "quality" / "final"
OUT_PATH = WORKSPACE / "benchmarks" / "comprehensive" / "comprehensive_agri_benchmark_v2.jsonl"

ACCEPTED_STATUSES = {"keep", "review"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge all final category benchmarks into one comprehensive file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without writing output file")
    args = parser.parse_args()

    final_files = sorted(FINAL_DIR.glob("*_final.jsonl"))
    if not final_files:
        print(f"ERROR: no *_final.jsonl files found in {FINAL_DIR}", file=sys.stderr)
        return 1

    merged: list[dict] = []
    print(f"Merging {len(final_files)} categories (keep + review only):\n")

    for f in final_files:
        cat = f.stem.replace("_final", "")
        items = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        accepted = [i for i in items if i.get("final_status") in ACCEPTED_STATUSES]
        by_status = {s: sum(1 for item in accepted if item.get("final_status") == s) for s in ACCEPTED_STATUSES}
        print(f"  {cat:<45} keep={by_status['keep']:>3}  review={by_status['review']:>3}  "
              f"total={len(accepted):>3}  (of {len(items)} scored)")
        merged.extend(accepted)

    print(f"\n  {'TOTAL':<45} {len(merged)} items")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in merged) + "\n",
        encoding="utf-8",
    )
    print(f"\nWritten → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
