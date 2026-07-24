#!/usr/bin/env python3
"""Compare the trained Spanish-QMD candidate with CUDA-matched v5clean baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def metric(path: Path) -> dict[str, Any]:
    return load(path).get("summary", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-english", required=True, type=Path)
    parser.add_argument("--baseline-spanish", required=True, type=Path)
    parser.add_argument("--candidate-english", required=True, type=Path)
    parser.add_argument("--candidate-spanish", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--english-contamination", required=True, type=Path)
    parser.add_argument("--spanish-contamination", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_en = metric(args.baseline_english)
    baseline_es = metric(args.baseline_spanish)
    candidate_en = metric(args.candidate_english)
    candidate_es = metric(args.candidate_spanish)
    dataset = load(args.dataset_manifest)
    english_audit = load(args.english_contamination)
    spanish_audit = load(args.spanish_contamination)

    def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
        return candidate[key] - baseline[key]

    spanish_ll_delta = delta(candidate_es, baseline_es, "mean_option_set_gold_logprob")
    english_ll_delta = delta(candidate_en, baseline_en, "mean_option_set_gold_logprob")
    spanish_accuracy_delta = delta(candidate_es, baseline_es, "argmax_accuracy")
    english_accuracy_delta = delta(candidate_en, baseline_en, "argmax_accuracy")
    complete = all(
        summary.get("scored_mcq_rows") == 1603 and not summary.get("skipped_non_mcq_rows")
        for summary in (baseline_en, baseline_es, candidate_en, candidate_es)
    )
    report = {
        "format": "spanish_qmd_candidate_comparison_v1",
        "status": "candidate_rejected_regression",
        "candidate_model": candidate_en.get("model"),
        "baseline_model": baseline_en.get("model"),
        "training": {
            "dataset_manifest": str(args.dataset_manifest),
            "dataset_manifest_status": dataset.get("spanish_records_training_allowed"),
            "train_count": dataset.get("output_train", {}).get("count"),
            "validation_count": dataset.get("output_validation", {}).get("count"),
        },
        "cuda_matched": True,
        "english_v5clean_anchor": {
            "baseline": {
                "mean_option_set_gold_logprob": baseline_en.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": baseline_en.get("argmax_accuracy"),
            },
            "candidate": {
                "mean_option_set_gold_logprob": candidate_en.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": candidate_en.get("argmax_accuracy"),
            },
            "delta": {
                "mean_option_set_gold_logprob": english_ll_delta,
                "argmax_accuracy": english_accuracy_delta,
            },
        },
        "spanish_transfer_diagnostic": {
            "baseline": {
                "mean_option_set_gold_logprob": baseline_es.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": baseline_es.get("argmax_accuracy"),
            },
            "candidate": {
                "mean_option_set_gold_logprob": candidate_es.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": candidate_es.get("argmax_accuracy"),
            },
            "delta": {
                "mean_option_set_gold_logprob": spanish_ll_delta,
                "argmax_accuracy": spanish_accuracy_delta,
            },
        },
        "acceptance_gates": {
            "all_scores_complete": complete,
            "spanish_improvement": spanish_ll_delta > 0.0,
            "english_no_regression": english_ll_delta >= 0.0,
            "english_training_contamination_pass": english_audit.get("passed") is True,
            "spanish_training_contamination_pass": spanish_audit.get("passed") is True,
        },
        "interpretation": [
            "The candidate is a completed trained baseline point, not an accepted autoresearch candidate.",
            "Both language gates fail: Spanish transfer and English anchor are lower than CUDA-matched v5clean.",
            "The candidate should not be promoted or used as the autoresearch champion.",
        ],
        "score_artifacts": {
            "baseline_english": str(args.baseline_english),
            "baseline_spanish": str(args.baseline_spanish),
            "candidate_english": str(args.candidate_english),
            "candidate_spanish": str(args.candidate_spanish),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "spanish_ll_delta": spanish_ll_delta, "english_ll_delta": english_ll_delta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
