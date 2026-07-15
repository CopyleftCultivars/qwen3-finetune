#!/usr/bin/env python3
"""Combine continuous MCQ and free-form LL signals into one primary score."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = "autoresearch/objective_weights.json"
DEFAULT_MCQ_OUTPUT = "results/primary_mcq_gold_ll.json"
DEFAULT_FREEFORM_OUTPUT = "results/primary_freeform_refll.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def validate_weights(weights: dict[str, Any], path: Path) -> None:
    required = (
        "formula",
        "mcq_gold_ll_weight",
        "freeform_ref_ll_weight",
        "objective_direction",
    )
    for key in required:
        if key not in weights:
            raise ValueError(f"{path}: missing required weight config key {key!r}")
    for key in ("mcq_gold_ll_weight", "freeform_ref_ll_weight"):
        if not isinstance(weights[key], (int, float)):
            raise ValueError(f"{path}: {key} must be numeric")
    if weights["objective_direction"] != "maximize":
        raise ValueError(f"{path}: objective_direction must be 'maximize'")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_command(command: list[str], dry_run: bool = False) -> None:
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return
    subprocess.run(command, cwd=repo_root(), check=True)


def require_model(component: str, explicit_model: str | None, shared_model: str | None) -> str:
    model = explicit_model or shared_model
    if not model:
        raise SystemExit(
            f"{component} model is required unless a precomputed result is passed "
            f"(--{component}-result)."
        )
    return model


def ensure_mcq_result(args: argparse.Namespace) -> Path:
    if args.mcq_result:
        return Path(args.mcq_result)

    model = require_model("mcq", args.mcq_model, args.model)
    output = Path(args.mcq_output)
    command = [
        sys.executable,
        "autoresearch/score_mcq_ll.py",
        "--model",
        model,
        "--dataset",
        args.mcq_dataset,
        "--device",
        args.mcq_device,
        "--output",
        output.as_posix(),
        "--diagnostics-dir",
        args.diagnostics_dir,
    ]
    if args.mcq_limit is not None:
        command.extend(["--limit", str(args.mcq_limit)])
    if not args.mcq_full_precision:
        command.append("--no-full-precision")
    run_command(command, args.dry_run)
    return output


def ensure_freeform_result(args: argparse.Namespace) -> Path:
    if args.freeform_result:
        return Path(args.freeform_result)

    model = require_model("freeform", args.freeform_model, args.model)
    output = Path(args.freeform_output)
    command = [
        sys.executable,
        "eval_freeform/score_refll.py",
        "--model",
        model,
        "--input",
        args.freeform_input,
        "--output",
        output.as_posix(),
        "--device",
        args.freeform_device,
    ]
    if args.freeform_limit is not None:
        command.extend(["--limit", str(args.freeform_limit)])
    if args.mock_freeform_model:
        command.append("--mock-model")
    run_command(command, args.dry_run)
    return output


def weighted_rubric_percent(path: Path) -> float:
    rows = load_jsonl(path)
    routine_scores = [
        float(row["weighted_score"])
        for row in rows
        if row.get("grader_role", "routine") == "routine" and "weighted_score" in row
    ]
    if not routine_scores:
        raise ValueError(f"{path}: no routine weighted_score rows found")
    return 100.0 * sum(routine_scores) / len(routine_scores)


def build_result(
    weights: dict[str, Any],
    mcq_result_path: Path,
    freeform_result_path: Path,
    rubric_result_path: Path | None,
) -> dict[str, Any]:
    mcq = load_json(mcq_result_path)
    freeform = load_json(freeform_result_path)
    mcq_summary = mcq.get("summary", {})
    freeform_summary = freeform.get("summary", {})

    mcq_ll = float(mcq_summary["mean_gold_logprob"])
    freeform_ll = float(freeform_summary["mean_reference_logprob_per_token"])
    mcq_weight = float(weights["mcq_gold_ll_weight"])
    freeform_weight = float(weights["freeform_ref_ll_weight"])
    primary = mcq_weight * mcq_ll + freeform_weight * freeform_ll

    secondary: dict[str, Any] = {
        "label": "reported_only_not_used_for_accept_revert",
        "mcq_argmax_accuracy": mcq_summary.get("argmax_accuracy"),
        "freeform_rubric_percent": None,
    }
    if rubric_result_path is not None:
        secondary["freeform_rubric_percent"] = weighted_rubric_percent(rubric_result_path)
        secondary["freeform_rubric_result"] = rubric_result_path.as_posix()

    return {
        "format": "autoresearch_primary_objective_result_v1",
        "objective_direction": "maximize",
        "primary": {
            "label": "primary_continuous_objective",
            "formula": weights["formula"],
            "score": primary,
            "components": {
                "mcq_mean_gold_logprob": {
                    "value": mcq_ll,
                    "weight": mcq_weight,
                    "contribution": mcq_weight * mcq_ll,
                    "source": mcq_result_path.as_posix(),
                },
                "freeform_mean_reference_logprob_per_token": {
                    "value": freeform_ll,
                    "weight": freeform_weight,
                    "contribution": freeform_weight * freeform_ll,
                    "source": freeform_result_path.as_posix(),
                },
            },
        },
        "secondary_reported_only": secondary,
        "weights_config": weights,
        "inputs": {
            "mcq_result": mcq_result_path.as_posix(),
            "freeform_refll_result": freeform_result_path.as_posix(),
            "rubric_result": rubric_result_path.as_posix() if rubric_result_path else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or combine MCQ/free-form LL scorers into the frozen primary objective.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default="results/primary_objective.json")
    parser.add_argument("--model", default=None, help="Shared model path/name for both scorers")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scorer commands; combine only if the referenced result files already exist.",
    )

    parser.add_argument("--mcq-result", default=None, help="Precomputed score_mcq_ll.py JSON")
    parser.add_argument("--mcq-model", default=None)
    parser.add_argument("--mcq-dataset", default="data/val_v3.jsonl")
    parser.add_argument("--mcq-output", default=DEFAULT_MCQ_OUTPUT)
    parser.add_argument("--mcq-device", default="cuda")
    parser.add_argument("--mcq-limit", type=int, default=None)
    parser.add_argument(
        "--mcq-full-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="../benchmark-eval-workspace/diagnostics",
    )

    parser.add_argument("--freeform-result", default=None, help="Precomputed score_refll.py JSON")
    parser.add_argument("--freeform-model", default=None)
    parser.add_argument("--freeform-input", default="eval_freeform/dev.jsonl")
    parser.add_argument("--freeform-output", default=DEFAULT_FREEFORM_OUTPUT)
    parser.add_argument("--freeform-device", default="cuda")
    parser.add_argument("--freeform-limit", type=int, default=None)
    parser.add_argument("--mock-freeform-model", action="store_true")

    parser.add_argument(
        "--rubric-result",
        default=None,
        help="Optional score_rubric.py JSONL; reported as a secondary metric only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weights_path = Path(args.weights)
    weights = load_json(weights_path)
    validate_weights(weights, weights_path)

    # Frozen equal weights are a neutral starting prior, not a tuned choice. Both
    # components remain raw log-likelihoods so #35 can validate whether this
    # literal objective is sane before the accept/revert gate ever climbs it.
    mcq_result_path = ensure_mcq_result(args)
    freeform_result_path = ensure_freeform_result(args)
    if args.dry_run and (not mcq_result_path.exists() or not freeform_result_path.exists()):
        return 0
    rubric_result_path = Path(args.rubric_result) if args.rubric_result else None
    result = build_result(weights, mcq_result_path, freeform_result_path, rubric_result_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {output_path.as_posix()}")
    print(f"primary_score={result['primary']['score']:.8f}")
    for name, component in result["primary"]["components"].items():
        print(
            f"{name}={component['value']:.8f} "
            f"weight={component['weight']:.6f} "
            f"contribution={component['contribution']:.8f}"
        )
    print("secondary_metrics=reported_only_not_used_for_accept_revert")
    print(f"mcq_argmax_accuracy={result['secondary_reported_only']['mcq_argmax_accuracy']}")
    print(f"freeform_rubric_percent={result['secondary_reported_only']['freeform_rubric_percent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
