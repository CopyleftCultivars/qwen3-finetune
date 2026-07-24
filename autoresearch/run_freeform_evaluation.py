#!/usr/bin/env python3
"""Run the English free-form dev evaluation as one auditable pipeline.

The pipeline generates deterministic answers for a fixed dev subset, grades
both baseline and candidate with the same rubric backend, and emits a paired
summary. OpenRouter is deliberately opt-in; the default ``mock`` backend is
only a schema/flow smoke test and is never a reportable quality score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_freeform.guards import (  # noqa: E402
    assert_not_sealed_path,
    load_jsonl,
    validate_dev_ids,
)


def select_subset(rows: list[dict[str, Any]], items_per_type: int) -> list[dict[str, Any]]:
    if items_per_type < 1:
        raise ValueError("items_per_type must be positive")
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        row_type = str(row.get("type", "unknown"))
        if counts[row_type] < items_per_type:
            selected.append(row)
            counts[row_type] += 1
    if not selected:
        raise ValueError("free-form dev input produced an empty subset")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_command(command: list[str], *, label: str) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"completed={label}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def summarize_rubric(path: Path, expected_ids: set[str]) -> dict[str, Any]:
    rows = read_jsonl(path)
    routine = [row for row in rows if row.get("grader_role", "routine") == "routine"]
    ids = [str(row.get("id", "")) for row in routine]
    if set(ids) != expected_ids or len(ids) != len(expected_ids):
        raise ValueError(f"{path}: rubric IDs do not match the selected dev subset")
    scores = [float(row["weighted_score"]) for row in routine]
    by_type: dict[str, list[float]] = {}
    for row in routine:
        by_type.setdefault(str(row.get("type", "unknown")), []).append(float(row["weighted_score"]))
    return {
        "path": str(path),
        "item_count": len(routine),
        "mean_weighted_score": sum(scores) / len(scores),
        "percent": 100.0 * sum(scores) / len(scores),
        "by_type_percent": {
            item_type: 100.0 * sum(values) / len(values)
            for item_type, values in sorted(by_type.items())
        },
        "grader_models": sorted({str(row.get("grader_model", "unknown")) for row in routine}),
        "mock_grader": all(bool(row.get("mock_grader")) for row in routine),
    }


def summarize_anchor(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("anchors", [])
    max_abs_drift = max((abs(float(row["delta"])) for row in records), default=0.0)
    return {
        "path": str(path),
        "anchor_count": int(report.get("anchor_count", len(records))),
        "max_abs_drift": max_abs_drift,
        "drift_alert_count": int(report.get("drift_alert_count", 0)),
    }


def summarize_refll(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    return {
        "path": str(path),
        "scored_items": int(summary.get("scored_items", 0)),
        "mean_reference_logprob_per_token": summary.get("mean_reference_logprob_per_token"),
        "prompt_mode": summary.get("prompt_mode"),
        "chat_template_used": summary.get("chat_template_used"),
        "legacy_non_gating": summary.get("legacy_non_gating"),
    }


def evaluate_model(
    *,
    label: str,
    model: str,
    subset_path: Path,
    selected_ids: set[str],
    output_dir: Path,
    python_executable: str,
    config: Path,
    device: str,
    max_new_tokens: int,
    max_workers: int,
    judge_backend: str,
    openrouter_key: str | None,
    run_refll: bool,
    reuse: bool,
) -> dict[str, Any]:
    generation = output_dir / f"{label}_candidates.jsonl"
    rubric = output_dir / f"{label}_rubric.jsonl"
    anchors = output_dir / f"{label}_anchor_report.json"
    refll = output_dir / f"{label}_refll.json"

    if not (reuse and generation.exists()):
        run_command(
            [
                python_executable,
                "autoresearch/generate_freeform_candidates.py",
                "--model",
                model,
                "--dev-input",
                str(subset_path),
                "--output",
                str(generation),
                "--items-per-type",
                "999999",
                "--max-new-tokens",
                str(max_new_tokens),
                "--device",
                device,
            ],
            label=f"generate:{label}",
        )
    generated_ids = {str(row.get("id", "")) for row in read_jsonl(generation)}
    if generated_ids != selected_ids:
        raise ValueError(f"{generation}: generated IDs do not match selected dev IDs")

    if not (reuse and rubric.exists() and anchors.exists()):
        command = [
            python_executable,
            "eval_freeform/score_rubric.py",
            "--input",
            str(generation),
            "--output",
            str(rubric),
            "--config",
            str(config),
            "--anchor-report",
            str(anchors),
            "--audit-first",
            "0",
            "--max-workers",
            str(max_workers),
        ]
        if judge_backend == "mock":
            command.append("--mock-grader")
        else:
            command.extend(["--judge-backend", judge_backend])
            if openrouter_key:
                command.extend(["--openrouter-key", openrouter_key])
        run_command(command, label=f"rubric:{label}")

    result: dict[str, Any] = {
        "model": model,
        "generation": str(generation),
        "rubric": summarize_rubric(rubric, selected_ids),
        "anchors": summarize_anchor(anchors),
    }
    if run_refll:
        if not (reuse and refll.exists()):
            run_command(
                [
                    python_executable,
                    "eval_freeform/score_refll.py",
                    "--model",
                    model,
                    "--input",
                    str(subset_path),
                    "--output",
                    str(refll),
                    "--device",
                    device,
                ],
                label=f"refll:{label}",
            )
        result["refll"] = summarize_refll(refll)
    return result


def build_report(args: argparse.Namespace, subset: list[dict[str, Any]], baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_percent = float(baseline["rubric"]["percent"])
    candidate_percent = float(candidate["rubric"]["percent"])
    return {
        "format": "freeform_paired_evaluation_report_v1",
        "status": "mock_smoke_only" if args.judge_backend == "mock" else "report_only_pending_gate_calibration",
        "dev_only": True,
        "sealed_test_paths_read": [],
        "input": {
            "path": str(args.dev_input),
            "sha256": sha256_file(Path(args.dev_input)),
            "selected_count": len(subset),
            "selected_ids": [str(row["id"]) for row in subset],
            "type_counts": dict(sorted(Counter(str(row.get("type", "unknown")) for row in subset).items())),
        },
        "judge": {
            "backend": args.judge_backend,
            "external_content_transmission": args.judge_backend != "mock",
            "config": str(args.config),
            "config_sha256": sha256_file(Path(args.config)),
            "same_backend_required": True,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "rubric_candidate_minus_baseline_percent_points": candidate_percent - baseline_percent,
            "rubric_drop_percent_points": baseline_percent - candidate_percent,
            "anchor_max_abs_drift": max(
                float(baseline["anchors"]["max_abs_drift"]),
                float(candidate["anchors"]["max_abs_drift"]),
            ),
        },
        "policy": {
            "rubric_role": "secondary_guardrail_until_human_calibrated_and_frozen",
            "reference_ll_role": "reported_only_diagnostic_not_primary",
            "promotion_requires": [
                "same fixed dev IDs for baseline and candidate",
                "same judge backend/model/config",
                "anchor drift within threshold",
                "English MCQ primary no-regression gate",
                "no sealed test access",
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dev-input", default="eval_freeform/dev.jsonl")
    parser.add_argument("--split-manifest", default="eval_freeform/split_manifest.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="eval_freeform/scorer_config.json")
    parser.add_argument("--python", default=sys.executable, dest="python_executable")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--items-per-type", type=int, default=18)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--judge-backend",
        choices=("mock", "openrouter", "codex"),
        default="mock",
        help="mock is a non-reportable smoke test; OpenRouter/Codex are explicit external judge choices",
    )
    parser.add_argument("--openrouter-key", default=None)
    parser.add_argument("--run-refll", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.judge_backend == "openrouter" and args.openrouter_key:
        # The key is passed only to the child process and never written to a report.
        pass
    dev_input = assert_not_sealed_path(args.dev_input)
    config = assert_not_sealed_path(args.config)
    rows = load_jsonl(dev_input)
    validate_dev_ids((row.get("id", "") for row in rows), args.split_manifest)
    subset = select_subset(rows, args.items_per_type)
    selected_ids = validate_dev_ids((row.get("id", "") for row in subset), args.split_manifest)

    output_dir = Path(args.output_dir or f"results/freeform_eval_{args.label}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "dev_subset.jsonl"
    write_jsonl(subset_path, subset)

    baseline = evaluate_model(
        label="baseline",
        model=args.baseline_model,
        subset_path=subset_path,
        selected_ids=selected_ids,
        output_dir=output_dir,
        python_executable=args.python_executable,
        config=config,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_workers=args.max_workers,
        judge_backend=args.judge_backend,
        openrouter_key=args.openrouter_key,
        run_refll=args.run_refll,
        reuse=args.reuse,
    )
    candidate = evaluate_model(
        label="candidate",
        model=args.candidate_model,
        subset_path=subset_path,
        selected_ids=selected_ids,
        output_dir=output_dir,
        python_executable=args.python_executable,
        config=config,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_workers=args.max_workers,
        judge_backend=args.judge_backend,
        openrouter_key=args.openrouter_key,
        run_refll=args.run_refll,
        reuse=args.reuse,
    )

    report = build_report(args, subset, baseline, candidate)
    output = Path(args.output or f"results/freeform_eval_{args.label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"judge_backend={args.judge_backend}")
    print(f"selected_items={len(subset)}")
    print(f"baseline_rubric_percent={baseline['rubric']['percent']:.4f}")
    print(f"candidate_rubric_percent={candidate['rubric']['percent']:.4f}")
    print(
        "rubric_candidate_minus_baseline_percent_points="
        f"{report['comparison']['rubric_candidate_minus_baseline_percent_points']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
