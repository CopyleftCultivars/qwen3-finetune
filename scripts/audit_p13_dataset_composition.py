#!/usr/bin/env python3
"""Write a reproducible composition report for the P13 candidate dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_dataset_v3 import _load_conversations, _load_mcq_with_rationale, _load_qmd_mcq


TARGET_RE = re.compile(r"</think>\s*([A-D])\s*$", re.DOTALL)
MCQ_RE = re.compile(r"^.+\n\nA\. .+\nB\. .+\nC\. .+\nD\. .+$", re.DOTALL)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rationale", type=Path, required=True)
    parser.add_argument("--qmd", type=Path, required=True)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conversation_count = sum(1 for _ in _load_conversations(args.conversations))
    rationale_count = sum(
        1
        for _ in _load_mcq_with_rationale(
            args.rationale,
            categories={"natural_fertilizers"},
            overlay=False,
            sft_format="bare-letter",
        )
    )
    rationale_overlay_count = sum(
        1
        for _ in _load_mcq_with_rationale(
            args.rationale,
            categories={"natural_fertilizers"},
            overlay=True,
            sft_format="bare-letter",
        )
    )
    qmd_raw_count = len(jsonl(args.qmd))
    qmd_expanded_count = sum(
        1 for _ in _load_qmd_mcq(args.qmd, sft_format="bare-letter")
    )
    train_rows = jsonl(args.train)
    val_rows = jsonl(args.val)
    final_rows = train_rows + val_rows
    final_mcq_letters = Counter(
        match.group(1)
        for row in final_rows
        if MCQ_RE.match(str(row.get("instruction", "")))
        and (match := TARGET_RE.search(str(row.get("output", ""))))
    )
    before_filter = conversation_count + rationale_overlay_count + qmd_expanded_count
    report = {
        "format": "p13_dataset_composition_v1",
        "config": str(args.config),
        "sealed_test_paths_read": [],
        "sources": {
            "gemma4_conversations": conversation_count,
            "mcq_rationale_natural_fertilizers_audited": rationale_count,
            "mcq_rationale_natural_fertilizers_overlay": rationale_overlay_count,
            "qmd_mcq_raw": qmd_raw_count,
            "qmd_mcq_cyclic_expanded": qmd_expanded_count,
        },
        "composition": {
            "total_before_filtering": before_filter,
            "total_after_filtering": len(final_rows),
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "final_mcq_rows": sum(final_mcq_letters.values()),
            "final_mcq_letter_counts": dict(sorted(final_mcq_letters.items())),
        },
        "checks": {
            "rationale_source_is_audit_approved": all(
                row.get("_contamination_status") == "passed"
                for row in jsonl(args.rationale)
            ),
            "rationale_category_is_natural_fertilizers": all(
                row.get("_source_category") == "natural_fertilizers"
                for row in jsonl(args.rationale)
            ),
            "final_mcq_letter_balance_exact": final_mcq_letters == Counter(
                {letter: sum(final_mcq_letters.values()) // 4 for letter in "ABCD"}
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
