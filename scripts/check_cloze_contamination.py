#!/usr/bin/env python3
"""Audit cloze source items against a benchmark question bank.

Only records whose source instruction is neither an exact nor a high-overlap
match to a benchmark question are emitted to ``--output``. The audit does not
claim semantic equivalence detection; it is a deterministic floor that catches
direct question reuse before a more expensive near-duplicate review.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_TOKEN.findall(text))


def token_set(text: str) -> set[str]:
    return set(_TOKEN.findall(normalize(text)))


def overlap_score(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if len(left_tokens) < 8 or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(cloze_items: list[dict], benchmark_items: list[dict]) -> tuple[list[dict], dict]:
    benchmark_questions = [str(item.get("question", "")) for item in benchmark_items]
    exact = {normalize(question): index for index, question in enumerate(benchmark_questions)}
    kept: list[dict] = []
    matches: list[dict] = []
    for item in cloze_items:
        source_instruction = str(item.get("_source_instruction", ""))
        normalized = normalize(source_instruction)
        exact_index = exact.get(normalized)
        best_index = None
        best_score = 0.0
        if exact_index is None:
            for index, question in enumerate(benchmark_questions):
                score = overlap_score(source_instruction, question)
                if score > best_score:
                    best_index, best_score = index, score
        if exact_index is not None or best_score >= 0.85:
            matches.append(
                {
                    "cloze_id": item.get("_cloze_id"),
                    "match_type": "exact" if exact_index is not None else "high_token_overlap",
                    "benchmark_index": exact_index if exact_index is not None else best_index,
                    "score": 1.0 if exact_index is not None else best_score,
                }
            )
            continue
        clean = dict(item)
        clean["_contamination_status"] = "passed"
        kept.append(clean)

    report = {
        "format": "cloze_contamination_audit_v1",
        "cloze_count": len(cloze_items),
        "kept_count": len(kept),
        "rejected_count": len(matches),
        "threshold": 0.85,
        "matches": matches,
        "passed": not matches,
    }
    return kept, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloze", required=True, metavar="PATH")
    parser.add_argument("--benchmark", required=True, metavar="PATH")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument("--report", required=True, metavar="PATH")
    args = parser.parse_args()

    kept, report = audit(load_jsonl(Path(args.cloze)), load_jsonl(Path(args.benchmark)))
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Cloze audit: {report['kept_count']}/{report['cloze_count']} kept; "
        f"{report['rejected_count']} rejected → {output_path}"
    )
    if report["rejected_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
