#!/usr/bin/env python3
"""Orchestrate a real periodic-guard measurement (chainlink #66): rubric drop
and grader-anchor drift, baseline vs one candidate.

Pipeline: generate free-form candidate answers for both models on a
deterministic dev-set subset (generate_freeform_candidates.py), grade both
sets with the real OpenRouter judge (eval_freeform/score_rubric.py, routine
model only — audit tier is for judge-drift monitoring, not this comparison),
then compute rubric drop_percent_points and read off grader_anchor drift.

Any OpenRouter auth/quota failure is classified and reported explicitly
(never silently downgraded to a passing placeholder) so a run stopped by a
403 or exhausted credits is unambiguous in the output and in --output.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

CONDA_ENV = "qwen3-diagnostics"

_ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("credits_exhausted", re.compile(r"insufficient_quota|payment required|\b402\b|out of credit", re.I)),
    ("auth_failed_403", re.compile(r"\b403\b|forbidden", re.I)),
    ("auth_failed_401", re.compile(r"\b401\b|unauthorized|invalid_api_key", re.I)),
    ("rate_limited_429", re.compile(r"\b429\b|rate.?limit", re.I)),
]


def classify_failure(stderr: str) -> str:
    for label, pattern in _ERROR_PATTERNS:
        if pattern.search(stderr):
            return label
    return "unknown_error"


def run(argv: list[str], label: str) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode != 0:
        failure = classify_failure(result.stderr)
        print(f"FAILED [{label}] classification={failure}", flush=True)
        print(result.stderr[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(
            json.dumps(
                {
                    "stage": label,
                    "failure_classification": failure,
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-2000:],
                }
            )
        )
    return result


def conda_run(args: list[str]) -> list[str]:
    return ["conda", "run", "-n", CONDA_ENV, "python"] + args


def generate(model: str, output: str, items_per_type: int, *, reuse: bool = False) -> None:
    if reuse and Path(output).exists():
        print(f"reusing cached generation: {output}", flush=True)
        return
    run(
        conda_run(
            [
                "autoresearch/generate_freeform_candidates.py",
                "--model",
                model,
                "--output",
                output,
                "--items-per-type",
                str(items_per_type),
            ]
        ),
        label=f"generate:{output}",
    )


def grade(
    input_path: str,
    output: str,
    anchor_report: str,
    *,
    reuse: bool = False,
    judge_backend: str = "openrouter",
) -> dict[str, Any]:
    if not (reuse and Path(output).exists() and Path(anchor_report).exists()):
        run(
            conda_run(
                [
                    "eval_freeform/score_rubric.py",
                    "--input",
                    input_path,
                    "--output",
                    output,
                    "--anchor-report",
                    anchor_report,
                    "--audit-first",
                    "0",
                    "--judge-backend",
                    judge_backend,
                ]
            ),
            label=f"grade:{output}",
        )
    elif reuse:
        print(f"reusing cached grading: {output}", flush=True)
    rows = [json.loads(line) for line in Path(output).open(encoding="utf-8") if line.strip()]
    mean_score = sum(row["weighted_score"] for row in rows) / len(rows)
    return {
        "output": output,
        "n_items": len(rows),
        "mean_weighted_score": mean_score,
        "percent": mean_score * 100.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure real rubric + grader-anchor guards, baseline vs candidate."
    )
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--label", required=True, help="Suffix for output filenames")
    parser.add_argument(
        "--baseline-label",
        default=None,
        help=(
            "Suffix for baseline-side filenames; defaults to --label. Pass the "
            "same value across streams in one tournament (same baseline model) "
            "with --reuse-baseline so the baseline is only generated/graded once."
        ),
    )
    parser.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="Reuse existing baseline generation/grading output if present",
    )
    parser.add_argument("--items-per-type", type=int, default=6)
    parser.add_argument(
        "--judge-backend",
        choices=("openrouter", "codex"),
        default="openrouter",
        help=(
            "Passed through to score_rubric.py. Switching backend is a "
            "grader-version change: do NOT --reuse-baseline output graded "
            "under a different backend"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    baseline_label = args.baseline_label or args.label
    Path("results").mkdir(exist_ok=True)
    baseline_gen = f"results/freeform_candidates_baseline_{baseline_label}.jsonl"
    candidate_gen = f"results/freeform_candidates_candidate_{args.label}.jsonl"

    generate(args.baseline_model, baseline_gen, args.items_per_type, reuse=args.reuse_baseline)
    generate(args.candidate_model, candidate_gen, args.items_per_type)

    baseline_anchor_report = f"results/periodic_anchor_baseline_{baseline_label}.json"
    candidate_anchor_report = f"results/periodic_anchor_candidate_{args.label}.json"
    baseline_rubric = grade(
        baseline_gen,
        f"results/periodic_rubric_baseline_{baseline_label}.jsonl",
        baseline_anchor_report,
        reuse=args.reuse_baseline,
        judge_backend=args.judge_backend,
    )
    candidate_rubric = grade(
        candidate_gen,
        f"results/periodic_rubric_candidate_{args.label}.jsonl",
        candidate_anchor_report,
        judge_backend=args.judge_backend,
    )

    anchor_reports = [
        json.loads(Path(baseline_anchor_report).read_text()),
        json.loads(Path(candidate_anchor_report).read_text()),
    ]
    max_abs_drift = max(
        abs(record["delta"])
        for report in anchor_reports
        for record in report["anchors"]
    )
    total_drift_alerts = sum(report["drift_alert_count"] for report in anchor_reports)

    drop_percent_points = baseline_rubric["percent"] - candidate_rubric["percent"]

    summary = {
        "label": args.label,
        "baseline_label": baseline_label,
        "baseline_model": args.baseline_model,
        "candidate_model": args.candidate_model,
        "items_per_type": args.items_per_type,
        "judge_backend": args.judge_backend,
        "rubric": {
            "baseline_percent": baseline_rubric["percent"],
            "candidate_percent": candidate_rubric["percent"],
            "drop_percent_points": drop_percent_points,
            "baseline_detail": baseline_rubric,
            "candidate_detail": candidate_rubric,
        },
        "grader_anchor": {
            "max_abs_drift": max_abs_drift,
            "total_drift_alerts": total_drift_alerts,
            "reports": [baseline_anchor_report, candidate_anchor_report],
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"rubric_drop_percent_points={drop_percent_points:.4f}")
    print(f"grader_anchor_max_abs_drift={max_abs_drift:.4f}")
    print(f"grader_anchor_total_drift_alerts={total_drift_alerts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
