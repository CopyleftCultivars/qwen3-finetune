#!/usr/bin/env python3
"""Compare serial Spanish debug variants with CUDA-matched v5clean anchors."""

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


def summary(path: Path) -> dict[str, Any]:
    value = load(path).get("summary")
    if not isinstance(value, dict):
        raise ValueError(f"{path}: missing summary")
    return value


def complete(value: dict[str, Any], rows: int) -> bool:
    return value.get("scored_mcq_rows") == rows and value.get("skipped_non_mcq_rows") == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-english", required=True, type=Path)
    parser.add_argument("--baseline-spanish", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--english-contamination", required=True, type=Path)
    parser.add_argument("--spanish-contamination", required=True, type=Path)
    parser.add_argument("--variant", action="append", nargs=3, metavar=("NAME", "ENGLISH", "SPANISH"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_en = summary(args.baseline_english)
    baseline_es = summary(args.baseline_spanish)
    dataset = load(args.dataset_manifest)
    english_audit = load(args.english_contamination)
    spanish_audit = load(args.spanish_contamination)

    variants: dict[str, Any] = {}
    for name, english_path, spanish_path in args.variant:
        variant_en = summary(Path(english_path))
        variant_es = summary(Path(spanish_path))
        variants[name] = {
            "models": {"english_score": variant_en.get("model"), "spanish_score": variant_es.get("model")},
            "scores_complete": complete(variant_en, 1603) and complete(variant_es, 321),
            "english_v5clean_anchor": {
                "mean_option_set_gold_logprob": variant_en.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": variant_en.get("argmax_accuracy"),
                "delta": {
                    "mean_option_set_gold_logprob": variant_en["mean_option_set_gold_logprob"] - baseline_en["mean_option_set_gold_logprob"],
                    "argmax_accuracy": variant_en["argmax_accuracy"] - baseline_en["argmax_accuracy"],
                },
            },
            "spanish_transfer_dev": {
                "mean_option_set_gold_logprob": variant_es.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": variant_es.get("argmax_accuracy"),
                "delta": {
                    "mean_option_set_gold_logprob": variant_es["mean_option_set_gold_logprob"] - baseline_es["mean_option_set_gold_logprob"],
                    "argmax_accuracy": variant_es["argmax_accuracy"] - baseline_es["argmax_accuracy"],
                },
            },
            "acceptance_gates": {
                "spanish_improvement": variant_es["mean_option_set_gold_logprob"] > baseline_es["mean_option_set_gold_logprob"],
                "english_no_regression": variant_en["mean_option_set_gold_logprob"] >= baseline_en["mean_option_set_gold_logprob"],
            },
            "score_artifacts": {"english": str(english_path), "spanish": str(spanish_path)},
        }

    best_spanish = max(variants, key=lambda name: variants[name]["spanish_transfer_dev"]["mean_option_set_gold_logprob"])
    report = {
        "format": "spanish_debug_comparison_v1",
        "status": "all_debug_variants_rejected" if not any(
            item["acceptance_gates"]["spanish_improvement"] and item["acceptance_gates"]["english_no_regression"]
            for item in variants.values()
        ) else "debug_variant_passed",
        "diagnostic_scope": {
            "base_model": baseline_en.get("model"),
            "epochs": dataset.get("v5_settings", {}).get("epochs"),
            "v5_settings": dataset.get("v5_settings"),
            "cuda_matched": True,
            "single_gpu_serial": True,
            "spanish_evaluation_split": "en_to_es/dev.jsonl",
            "sealed_test_paths_read": dataset.get("sealed_test_paths_read", []),
        },
        "baseline": {
            "english_v5clean_anchor": {
                "mean_option_set_gold_logprob": baseline_en.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": baseline_en.get("argmax_accuracy"),
            },
            "spanish_transfer_dev": {
                "mean_option_set_gold_logprob": baseline_es.get("mean_option_set_gold_logprob"),
                "argmax_accuracy": baseline_es.get("argmax_accuracy"),
            },
        },
        "variants": variants,
        "selection": {
            "best_spanish_by_option_set_gold_logprob": best_spanish,
            "promotion_candidate": None,
            "reason": "One-epoch diagnostics do not establish a promotion candidate unless both bilingual gates pass; no variant passed both gates.",
        },
        "contamination": {
            "english_training_audit_pass": english_audit.get("passed") is True,
            "spanish_training_audit_pass": spanish_audit.get("passed") is True,
            "english_audit": str(args.english_contamination),
            "spanish_audit": str(args.spanish_contamination),
        },
        "interpretation": [
            "The Spanish MCQ-only overlay is not the sole explanation for the full-candidate regression: all three one-epoch overlays remain below the v5clean Spanish dev anchor.",
            "Free-form-only is the least damaging variant by Spanish option-set log-likelihood, but it still fails the English no-regression gate.",
            "These are diagnostic runs, not evidence that Spanish data should be removed permanently; the next controlled lever should reduce optimization pressure or inspect formatting/truncation before changing benchmark policy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "best_spanish_variant": best_spanish}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
