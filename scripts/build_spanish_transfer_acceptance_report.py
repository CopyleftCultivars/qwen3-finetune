#!/usr/bin/env python3
"""Build the baseline bilingual acceptance report for the Spanish diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, default=Path("results/en_v5clean_anchor_baseline_v5clean.json"))
    parser.add_argument("--spanish", type=Path, default=Path("results/es_transfer_baseline_v5clean.json"))
    parser.add_argument("--contamination", type=Path, default=Path("results/spanish_transfer_contamination_audit.json"))
    parser.add_argument("--manifest", type=Path, default=Path("../spanish-benchmark-workspace/data/provenance/translations/en_to_es_english_v5clean_manifest.json"))
    parser.add_argument("--split-manifest", type=Path, default=Path("../spanish-benchmark-workspace/data/provenance/translations/en_to_es_english_v5clean_split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/spanish_transfer_acceptance_baseline.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    english = load(args.english)
    spanish = load(args.spanish)
    contamination = load(args.contamination)
    manifest = load(args.manifest)
    split_manifest = load(args.split_manifest)
    english_summary = english.get("summary", {})
    spanish_summary = spanish.get("summary", {})
    if english_summary.get("scored_mcq_rows") != 1603:
        raise SystemExit("English anchor baseline must score all 1603 records")
    if spanish_summary.get("scored_mcq_rows") != 1603:
        raise SystemExit("Spanish transfer baseline must score all 1603 records")
    if english_summary.get("skipped_non_mcq_rows") or spanish_summary.get("skipped_non_mcq_rows"):
        raise SystemExit("baseline scorer skipped one or more records")
    if not contamination.get("passed"):
        raise SystemExit("Spanish transfer contamination audit did not pass")
    if manifest.get("training_allowed") is not False:
        raise SystemExit("Spanish transfer manifest must set training_allowed=false")
    if split_manifest.get("training_allowed") is not False:
        raise SystemExit("Spanish transfer split manifest must set training_allowed=false")

    english_ll = english_summary["mean_option_set_gold_logprob"]
    spanish_ll = spanish_summary["mean_option_set_gold_logprob"]
    report = {
        "format": "spanish_transfer_acceptance_report_v1",
        "status": "baseline_established_candidate_comparison_pending",
        "model": english_summary.get("model"),
        "device": english_summary.get("device"),
        "scorer": "autoresearch/score_mcq_ll.py",
        "contamination_audit": {
            "path": str(args.contamination),
            "passed": True,
            "target_files": len(contamination.get("targets", [])),
            "exact_matches": sum(item["exact_match_count"] for item in contamination["targets"]),
            "high_lexical_overlap_matches": sum(item["high_lexical_overlap_count"] for item in contamination["targets"]),
        },
        "english_v5clean_anchor": {
            "path": english_summary.get("dataset"),
            "records": english_summary["scored_mcq_rows"],
            "mean_option_set_gold_logprob": english_ll,
            "argmax_accuracy": english_summary["argmax_accuracy"],
        },
        "spanish_transfer_diagnostic": {
            "path": spanish_summary.get("dataset"),
            "records": spanish_summary["scored_mcq_rows"],
            "mean_option_set_gold_logprob": spanish_ll,
            "argmax_accuracy": spanish_summary["argmax_accuracy"],
            "benchmark_role": manifest.get("benchmark_role"),
            "source_sha256": manifest.get("source_sha256"),
            "answer_counts": manifest.get("answer_counts"),
            "human_review_status": manifest.get("human_review_status"),
            "training_allowed": manifest.get("training_allowed"),
            "dev_records": split_manifest.get("dev_count"),
            "sealed_test_records": split_manifest.get("sealed_test_count"),
        },
        "diagnostic_delta": {
            "spanish_minus_english_mean_option_set_gold_logprob": spanish_ll - english_ll,
            "interpretation": "language-transfer diagnostic only; not a cross-language quality ranking",
        },
        "acceptance_gates": {
            "spanish_baseline_established": True,
            "english_no_regression": "pending_candidate_comparison",
            "spanish_improvement": "pending_candidate_comparison",
            "training_contamination": "pass",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
