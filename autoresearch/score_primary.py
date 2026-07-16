#!/usr/bin/env python3
"""Combine continuous MCQ and free-form LL signals into one primary score."""

from __future__ import annotations

import argparse
import json
import math
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
        "cross_model_refll_comparison",
        "first_c3_climbable_objective",
        "formula",
        "freeform_ref_ll_role",
        "mcq_metric",
        "mcq_gold_ll_role",
        "mcq_gold_ll_weight",
        "freeform_ref_ll_weight",
        "objective_direction",
        "primary_quality_guard",
    )
    for key in required:
        if key not in weights:
            raise ValueError(f"{path}: missing required weight config key {key!r}")
    for key in ("mcq_gold_ll_weight", "freeform_ref_ll_weight"):
        if not isinstance(weights[key], (int, float)):
            raise ValueError(f"{path}: {key} must be numeric")
    if weights["objective_direction"] != "maximize":
        raise ValueError(f"{path}: objective_direction must be 'maximize'")
    if weights["cross_model_refll_comparison"] != "invalid_unless_same_prompt_lineage":
        raise ValueError(
            f"{path}: cross_model_refll_comparison must be "
            "'invalid_unless_same_prompt_lineage'"
        )


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


def ensure_freeform_result(args: argparse.Namespace, *, required: bool) -> Path | None:
    if args.freeform_result:
        return Path(args.freeform_result)
    if not required and not args.run_freeform_diagnostic:
        return None

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


def mcq_metric_value(mcq: dict[str, Any], metric: str) -> float:
    summary = mcq.get("summary", {})
    if metric in summary:
        return float(summary[metric])
    if metric == "mean_option_set_gold_logprob":
        per_item = mcq.get("per_item", [])
        if not per_item:
            raise ValueError("MCQ result missing per_item rows for option-set fallback")
        values = []
        for row in per_item:
            if "option_set_gold_logprob" in row:
                values.append(float(row["option_set_gold_logprob"]))
            else:
                values.append(math.log(float(row["gold_probability"])))
        return sum(values) / len(values)
    raise KeyError(f"MCQ summary missing configured metric {metric!r}")


def build_result(
    weights: dict[str, Any],
    mcq_result_path: Path,
    freeform_result_path: Path | None,
    rubric_result_path: Path | None,
) -> dict[str, Any]:
    mcq = load_json(mcq_result_path)
    mcq_summary = mcq.get("summary", {})

    mcq_metric = str(weights["mcq_metric"])
    mcq_ll = mcq_metric_value(mcq, mcq_metric)
    mcq_weight = float(weights["mcq_gold_ll_weight"])
    freeform_weight = float(weights["freeform_ref_ll_weight"])
    primary = mcq_weight * mcq_ll
    primary_components: dict[str, Any] = {
        f"mcq_{mcq_metric}": {
            "value": mcq_ll,
            "weight": mcq_weight,
            "contribution": mcq_weight * mcq_ll,
            "source": mcq_result_path.as_posix(),
            "metric": mcq_metric,
            "role": weights["mcq_gold_ll_role"],
        }
    }

    freeform_summary: dict[str, Any] = {}
    freeform_ll: float | None = None
    if freeform_result_path is not None:
        freeform = load_json(freeform_result_path)
        freeform_summary = freeform.get("summary", {})
        freeform_ll = float(freeform_summary["mean_reference_logprob_per_token"])
        if freeform_weight != 0.0:
            primary += freeform_weight * freeform_ll
            primary_components["freeform_mean_reference_logprob_per_token"] = {
                "value": freeform_ll,
                "weight": freeform_weight,
                "contribution": freeform_weight * freeform_ll,
                "source": freeform_result_path.as_posix(),
                "role": "primary_component_requires_same_lineage_calibration",
            }
    elif freeform_weight != 0.0:
        raise ValueError("freeform_ref_ll_weight is non-zero but no freeform ref-LL result was provided")

    freeform_policy = weights.get("freeform_ref_ll_policy", {})
    freeform_diagnostic: dict[str, Any] = {
        "role": weights["freeform_ref_ll_role"],
        "primary_weight": freeform_weight,
        "status": "not_run",
        "source": None,
        "value": None,
        "prompt_mode": None,
        "chat_template_used": None,
        "cross_model_refll_comparison": weights["cross_model_refll_comparison"],
        "same_lineage_requirement": freeform_policy.get("same_lineage_requirement"),
        "interpretation": "not_a_cross_approach_quality_signal",
    }
    if freeform_result_path is not None:
        freeform_diagnostic.update(
            {
                "status": "reported_only_not_used_for_accept_revert"
                if freeform_weight == 0.0
                else "primary_component",
                "source": freeform_result_path.as_posix(),
                "value": freeform_ll,
                "prompt_mode": freeform_summary.get("prompt_mode"),
                "chat_template_used": freeform_summary.get("chat_template_used"),
                "refll_chat_template_enable_thinking": freeform_summary.get(
                    "refll_chat_template_enable_thinking"
                ),
                "legacy_non_gating": freeform_summary.get("legacy_non_gating"),
            }
        )

    secondary: dict[str, Any] = {
        "label": "reported_only_not_used_for_accept_revert",
        "mcq_argmax_accuracy": mcq_summary.get("argmax_accuracy"),
        "freeform_rubric_percent": None,
        "freeform_refll_diagnostic": freeform_ll,
    }
    guardrails: dict[str, Any] = {
        "freeform_rubric": {
            "role": weights["primary_quality_guard"],
            "status": "not_provided",
            "value_percent": None,
            "source": None,
        }
    }
    if rubric_result_path is not None:
        rubric_percent = weighted_rubric_percent(rubric_result_path)
        secondary["freeform_rubric_percent"] = rubric_percent
        secondary["freeform_rubric_result"] = rubric_result_path.as_posix()
        guardrails["freeform_rubric"].update(
            {
                "status": "provided_hard_periodic_guard",
                "value_percent": rubric_percent,
                "source": rubric_result_path.as_posix(),
            }
        )

    return {
        "format": "autoresearch_primary_objective_result_v2",
        "objective_direction": "maximize",
        "objective_policy": {
            "first_c3_climbable_objective": weights["first_c3_climbable_objective"],
            "formula": weights["formula"],
            "cross_model_refll_comparison": weights["cross_model_refll_comparison"],
            "freeform_ref_ll_policy": freeform_policy,
            "primary_quality_guard": weights["primary_quality_guard"],
        },
        "primary": {
            "label": "primary_continuous_objective",
            "formula": weights["formula"],
            "score": primary,
            "components": primary_components,
        },
        "guardrails": guardrails,
        "diagnostics_reported_only": {
            "freeform_refll": freeform_diagnostic,
        },
        "secondary_reported_only": secondary,
        "weights_config": weights,
        "inputs": {
            "mcq_result": mcq_result_path.as_posix(),
            "freeform_refll_result": freeform_result_path.as_posix()
            if freeform_result_path is not None
            else None,
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
        "--run-freeform-diagnostic",
        action="store_true",
        help="Run score_refll.py as an optional reported-only diagnostic when ref-LL is not primary.",
    )

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

    # The first C3 climbable primary is MCQ option-set LL only. Ref-LL is
    # optional/report-only unless the frozen policy explicitly gives it
    # non-zero weight after same-lineage calibration.
    mcq_result_path = ensure_mcq_result(args)
    freeform_required = float(weights["freeform_ref_ll_weight"]) != 0.0
    freeform_result_path = ensure_freeform_result(args, required=freeform_required)
    if args.dry_run and (
        not mcq_result_path.exists()
        or (
            freeform_required
            and (freeform_result_path is None or not freeform_result_path.exists())
        )
    ):
        return 0
    if args.dry_run and freeform_result_path is not None and not freeform_result_path.exists():
        freeform_result_path = None
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
    refll = result["diagnostics_reported_only"]["freeform_refll"]
    print(f"freeform_refll_status={refll['status']}")
    print(f"freeform_refll_role={refll['role']}")
    print(f"cross_model_refll_comparison={refll['cross_model_refll_comparison']}")
    print("secondary_metrics=reported_only_not_used_for_accept_revert")
    print(f"mcq_argmax_accuracy={result['secondary_reported_only']['mcq_argmax_accuracy']}")
    print(f"freeform_rubric_percent={result['secondary_reported_only']['freeform_rubric_percent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
