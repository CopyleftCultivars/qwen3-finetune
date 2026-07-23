#!/usr/bin/env python3
"""Select and provenance-package genuinely re-authored P13 rationale targets.

The existing clean rationale source is useful for question IDs, categories, and
the contamination boundary, but its think blocks are exact QMD replays.  This
utility selects one category for local step-back re-authoring and then packages
the generated pairs with source provenance.  It refuses to package missing,
duplicate, or textually unchanged generated rationales.

Examples:
    python scripts/build_p13_novel_rationale.py select \
        --source data/sources/qmd_mcq_rationale_clean.jsonl \
        --category natural_fertilizers \
        --output data/sources/p13_natural_fertilizers_input.jsonl

    python scripts/build_p13_novel_rationale.py package \
        --source data/sources/p13_natural_fertilizers_input.jsonl \
        --generated data/sources/p13_natural_fertilizers_generated.jsonl \
        --output data/sources/p13_novel_rationale_nf_raw.jsonl \
        --report results/p13_novel_rationale_authoring_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*([A-D])\s*$", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def rationale(output: str) -> tuple[str, str]:
    match = THINK_RE.search(output)
    if not match:
        raise ValueError("generated output is not <think>...</think> plus a bare letter")
    return match.group(1).strip(), match.group(2)


def select(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.source)
    selected = [row for row in rows if row.get("_source_category") == args.category]
    if not selected:
        raise ValueError(f"no rows found for category {args.category!r}")
    dump_jsonl(args.output, selected)
    print(json.dumps({"source_rows": len(rows), "selected_rows": len(selected), "category": args.category}))
    return 0


def package(args: argparse.Namespace) -> int:
    source = load_jsonl(args.source)
    generated = load_jsonl(args.generated)
    by_index = {int(row["_src_idx"]): row for row in generated if "_src_idx" in row}
    if len(by_index) != len(generated):
        raise ValueError("generated rows must each have a unique _src_idx")
    expected = set(range(len(source)))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(f"generated indices do not match source: missing={missing} extra={extra}")

    output_rows: list[dict[str, Any]] = []
    exact_output_matches = 0
    exact_rationale_matches = 0
    for index, original in enumerate(source):
        generated_row = by_index[index]
        if generated_row.get("instruction") != original.get("instruction"):
            raise ValueError(f"instruction changed for source index {index}")
        original_rationale, original_letter = rationale(str(original["output"]))
        generated_rationale, generated_letter = rationale(str(generated_row["output"]))
        if original_letter != generated_letter:
            raise ValueError(f"answer letter changed for source index {index}")
        if generated_row["output"] == original["output"]:
            exact_output_matches += 1
        if normalize(generated_rationale) == normalize(original_rationale):
            exact_rationale_matches += 1
        output_rows.append(
            {
                "instruction": generated_row["instruction"],
                "output": generated_row["output"],
                "_source": "p13_novel_rationale_authoring",
                "_source_question_id": original["_source_question_id"],
                "_source_category": original["_source_category"],
                "_rationale_origin": "local_qwen_stepback_from_qmd_question",
                "_rationale_replay_comparison": "compared_before_training",
            }
        )

    if exact_output_matches or exact_rationale_matches:
        raise ValueError(
            "generated rationales are not novel: "
            f"exact_output_matches={exact_output_matches} "
            f"exact_rationale_matches={exact_rationale_matches}"
        )

    dump_jsonl(args.output, output_rows)
    report = {
        "format": "p13_novel_rationale_authoring_audit_v1",
        "source": str(args.source),
        "generated": str(args.generated),
        "output": str(args.output),
        "category": source[0].get("_source_category"),
        "source_count": len(source),
        "generated_count": len(generated),
        "exact_output_match_count": exact_output_matches,
        "exact_rationale_match_count": exact_rationale_matches,
        "source_question_ids": [row["_source_question_id"] for row in source],
        "passed_textual_novelty": True,
        "sealed_test_paths_read": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "category", "source_count", "generated_count", "exact_output_match_count",
        "exact_rationale_match_count", "passed_textual_novelty",
    )}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--source", type=Path, required=True)
    select_parser.add_argument("--category", required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.set_defaults(func=select)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--source", type=Path, required=True)
    package_parser.add_argument("--generated", type=Path, required=True)
    package_parser.add_argument("--output", type=Path, required=True)
    package_parser.add_argument("--report", type=Path, required=True)
    package_parser.set_defaults(func=package)

    args = parser.parse_args()
    for name in ("source", "generated"):
        path = getattr(args, name, None)
        if path is not None and not path.exists():
            parser.error(f"file not found: {path}")
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
