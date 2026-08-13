#!/usr/bin/env python3
"""Accept/reject/pause gate for autoresearch candidate runs.

The gate is deliberately deterministic and file-driven. It does not train,
evaluate, edit, or revert by itself; it emits the decision and reasons that the
loop/revert layer will enforce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "autoresearch/gate_config.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def primary_score(result: dict[str, Any], label: str) -> float:
    try:
        return float(result["primary"]["score"])
    except KeyError as exc:
        raise ValueError(f"{label}: missing primary.score") from exc


def validate_config(config: dict[str, Any], budget_key: str = "compute_budget") -> None:
    if config.get("format") != "autoresearch_gate_config_v1":
        raise ValueError("gate config format must be autoresearch_gate_config_v1")
    for path in (
        ("primary_gate", "min_delta_standard_errors"),
        ("primary_gate", "standard_error"),
        (budget_key, "max_minutes_per_iteration"),
        ("forgetting_guard", "max_relative_drop_percent"),
        ("bias_guard", "max_abs_letter_frequency_deviation"),
        ("bias_guard", "max_deviation_increase_vs_baseline"),
        ("rubric_guard", "max_drop_percent_points"),
        ("grader_anchor_guard", "max_abs_drift"),
    ):
        node: Any = config
        for key in path:
            node = node[key]
        if not isinstance(node, (int, float)):
            raise ValueError(f"gate config {'.'.join(path)} must be numeric")


def relative_drop_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline score must be > 0 for relative drop guard")
    return 100.0 * (baseline - candidate) / baseline


def guardrail_reasons(
    config: dict[str, Any],
    guardrails: dict[str, Any],
    budget_key: str = "compute_budget",
) -> tuple[list[str], list[str]]:
    reject: list[str] = []
    pause: list[str] = []

    compute = guardrails.get("compute")
    if compute is None:
        reject.append("missing_compute_guard")
    else:
        elapsed = float(compute["elapsed_minutes"])
        limit = float(config[budget_key]["max_minutes_per_iteration"])
        if elapsed > limit:
            reject.append(f"compute_budget_exceeded elapsed={elapsed:.4f} limit={limit:.4f}")

    sealed = guardrails.get("sealed_test_access")
    if sealed is None:
        reject.append("missing_sealed_test_guard")
    elif bool(sealed.get("touched", False)):
        reject.append("sealed_test_touched")

    forgetting = guardrails.get("forgetting")
    if forgetting is None:
        reject.append("missing_forgetting_guard")
    else:
        if "relative_drop_percent" in forgetting:
            drop = float(forgetting["relative_drop_percent"])
        else:
            drop = relative_drop_percent(
                float(forgetting["baseline_score"]),
                float(forgetting["candidate_score"]),
            )
        limit = float(config["forgetting_guard"]["max_relative_drop_percent"])
        if drop > limit:
            reject.append(f"forgetting_drop_exceeded drop={drop:.4f} limit={limit:.4f}")

    bias = guardrails.get("bias")
    if bias is None:
        reject.append("missing_bias_guard")
    else:
        reference = config["bias_guard"]["reference_distribution"]

        def deviation(distribution: dict[str, Any]) -> float:
            return max(
                abs(float(distribution.get(letter, 0.0)) - float(reference[letter]))
                for letter in reference
            )

        observed_dev = deviation(bias.get("letter_distribution", {}))
        absolute_pass = float(config["bias_guard"]["max_abs_letter_frequency_deviation"])
        baseline_dist = bias.get("baseline_letter_distribution")
        if observed_dev <= absolute_pass:
            # Balanced enough in absolute terms: always acceptable.
            pass
        elif baseline_dist is not None:
            # Relative rule (2026-07-18 policy): the guard exists to prevent
            # REINTRODUCING letter bias, so a candidate may not sit further
            # from uniform than the champion baseline plus a noise tolerance.
            baseline_dev = deviation(baseline_dist)
            allowed = baseline_dev + float(
                config["bias_guard"]["max_deviation_increase_vs_baseline"]
            )
            if observed_dev > allowed:
                reject.append(
                    f"bias_reintroduced observed={observed_dev:.4f} "
                    f"baseline={baseline_dev:.4f} allowed={allowed:.4f}"
                )
        else:
            # No baseline marginals supplied: legacy absolute behavior.
            reject.append(
                f"bias_drift_exceeded max_abs={observed_dev:.4f} limit={absolute_pass:.4f}"
            )

    rubric = guardrails.get("rubric")
    if rubric is None:
        reject.append("missing_rubric_guard")
    else:
        if "drop_percent_points" in rubric:
            drop_points = float(rubric["drop_percent_points"])
        else:
            drop_points = float(rubric["baseline_percent"]) - float(rubric["candidate_percent"])
        limit = float(config["rubric_guard"]["max_drop_percent_points"])
        if drop_points > limit:
            reject.append(f"rubric_drop_exceeded drop={drop_points:.4f} limit={limit:.4f}")

    anchor = guardrails.get("grader_anchor")
    if anchor is None:
        pause.append("missing_grader_anchor_guard")
    else:
        drift = float(anchor["max_abs_drift"])
        limit = float(config["grader_anchor_guard"]["max_abs_drift"])
        if drift > limit:
            pause.append(f"grader_anchor_drift_exceeded drift={drift:.4f} limit={limit:.4f}")

    return reject, pause


def decide(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    guardrails: dict[str, Any],
    config: dict[str, Any],
    budget_key: str = "compute_budget",
) -> dict[str, Any]:
    validate_config(config, budget_key=budget_key)
    baseline_score = primary_score(baseline, "baseline")
    candidate_score = primary_score(candidate, "candidate")
    delta = candidate_score - baseline_score
    se = float(config["primary_gate"]["standard_error"])
    min_se = float(config["primary_gate"]["min_delta_standard_errors"])
    threshold = min_se * se

    reject_reasons, pause_reasons = guardrail_reasons(config, guardrails, budget_key=budget_key)
    metric_reasons: list[str] = []
    if delta <= threshold:
        metric_reasons.append(f"primary_delta_not_significant delta={delta:.10f} threshold={threshold:.10f}")

    if pause_reasons:
        decision = "pause"
        recommended_action = "pause_for_human"
    elif reject_reasons or metric_reasons:
        decision = "reject"
        recommended_action = "revert_candidate"
    else:
        decision = "accept"
        recommended_action = "keep_candidate"

    return {
        "format": "autoresearch_gate_decision_v1",
        "decision": decision,
        "recommended_action": recommended_action,
        "baseline_primary": baseline_score,
        "candidate_primary": candidate_score,
        "primary_delta": delta,
        "primary_standard_error": se,
        "primary_delta_threshold": threshold,
        "primary_delta_standard_errors": delta / se if se else None,
        "reasons": {
            "metric": metric_reasons,
            "reject_guardrails": reject_reasons,
            "pause_guardrails": pause_reasons,
        },
    }


def synthetic_primary(score: float) -> dict[str, Any]:
    return {"primary": {"score": score}}


def passing_guardrails() -> dict[str, Any]:
    return {
        "bias": {"letter_distribution": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}},
        "compute": {"elapsed_minutes": 10.0},
        "forgetting": {"baseline_score": 0.80, "candidate_score": 0.79},
        "grader_anchor": {"max_abs_drift": 0.02},
        "rubric": {"baseline_percent": 85.0, "candidate_percent": 83.0},
        "sealed_test_access": {"touched": False},
    }


def run_self_test(config: dict[str, Any]) -> None:
    threshold = (
        float(config["primary_gate"]["min_delta_standard_errors"])
        * float(config["primary_gate"]["standard_error"])
    )
    cases: list[tuple[str, dict[str, Any], str]] = [
        (
            "known_good_accept",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                passing_guardrails(),
                config,
            ),
            "accept",
        ),
        (
            "noop_reject",
            decide(synthetic_primary(0.0), synthetic_primary(0.0), passing_guardrails(), config),
            "reject",
        ),
        (
            "compute_budget_reject",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {**passing_guardrails(), "compute": {"elapsed_minutes": 30.0}},
                config,
            ),
            "reject",
        ),
        (
            "forgetting_guard_reject",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {**passing_guardrails(), "forgetting": {"baseline_score": 0.80, "candidate_score": 0.70}},
                config,
            ),
            "reject",
        ),
        (
            "bias_guard_reject",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {**passing_guardrails(), "bias": {"letter_distribution": {"A": 0.50, "B": 0.20, "C": 0.15, "D": 0.15}}},
                config,
            ),
            "reject",
        ),
        (
            "bias_relative_improvement_accept",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {
                    **passing_guardrails(),
                    "bias": {
                        "letter_distribution": {"A": 0.45, "B": 0.15, "C": 0.25, "D": 0.15},
                        "baseline_letter_distribution": {"A": 0.52, "B": 0.11, "C": 0.24, "D": 0.13},
                    },
                },
                config,
            ),
            "accept",
        ),
        (
            "bias_reintroduction_reject",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {
                    **passing_guardrails(),
                    "bias": {
                        "letter_distribution": {"A": 0.60, "B": 0.10, "C": 0.20, "D": 0.10},
                        "baseline_letter_distribution": {"A": 0.40, "B": 0.20, "C": 0.20, "D": 0.20},
                    },
                },
                config,
            ),
            "reject",
        ),
        (
            "rubric_guard_reject",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {**passing_guardrails(), "rubric": {"baseline_percent": 85.0, "candidate_percent": 70.0}},
                config,
            ),
            "reject",
        ),
        (
            "anchor_drift_pause",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {**passing_guardrails(), "grader_anchor": {"max_abs_drift": 0.50}},
                config,
            ),
            "pause",
        ),
        (
            "sealed_test_reject",
            decide(
                synthetic_primary(0.0),
                synthetic_primary(threshold + 0.1),
                {**passing_guardrails(), "sealed_test_access": {"touched": True}},
                config,
            ),
            "reject",
        ),
    ]
    failures = []
    for name, result, expected in cases:
        actual = result["decision"]
        print(f"{name}: expected={expected} actual={actual}")
        if actual != expected:
            failures.append((name, expected, actual, result))
    if failures:
        raise SystemExit(json.dumps(failures, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply frozen autoresearch accept/reject/pause gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline", default=None, help="Baseline score_primary.py JSON")
    parser.add_argument("--candidate", default=None, help="Candidate score_primary.py JSON")
    parser.add_argument("--guardrails", default=None, help="Guardrail report JSON")
    parser.add_argument("--output", default=None, help="Decision JSON output path")
    parser.add_argument("--self-test", action="store_true", help="Run built-in synthetic gate fixtures")
    parser.add_argument(
        "--budget-key",
        default="compute_budget",
        help=(
            "Which gate_config.json compute-budget block to enforce (default: "
            "compute_budget). GRPO iterations pass compute_budget_grpo -- rollout "
            "generation is far more expensive per-iteration than SFT forward/backward "
            "passes, so it gets its own threshold rather than sharing the SFT lever "
            "family's calibrated 28-minute budget."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_json(Path(args.config))
    validate_config(config, budget_key=args.budget_key)

    if args.self_test:
        run_self_test(config)
        return 0

    missing = [name for name in ("baseline", "candidate", "guardrails") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required argument(s) without --self-test: {', '.join(missing)}")

    result = decide(
        load_json(Path(args.baseline)),
        load_json(Path(args.candidate)),
        load_json(Path(args.guardrails)),
        config,
        budget_key=args.budget_key,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"decision={result['decision']}")
    print(f"recommended_action={result['recommended_action']}")
    print(f"primary_delta={result['primary_delta']:.10f}")
    print(f"primary_delta_threshold={result['primary_delta_threshold']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
