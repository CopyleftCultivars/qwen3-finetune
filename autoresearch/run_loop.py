#!/usr/bin/env python3
"""One-iteration autoresearch loop orchestrator.

This script does not propose edits itself. It runs one explicit candidate
iteration, applies the frozen gate, and appends an audit-ready journal row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROGRAM = "autoresearch/program.md"
DEFAULT_JOURNAL = "autoresearch/journal.jsonl"
DEFAULT_GATE_OUTPUT = "results/autoresearch_gate_decision.json"
SEALED_TEST_PATHS = (
    "eval_freeform/test.jsonl",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def run_command(argv: list[str], *, timeout_seconds: int | None, dry_run: bool) -> dict[str, Any]:
    command = " ".join(argv)
    if dry_run:
        return {
            "command": command,
            "dry_run": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }
    completed = subprocess.run(
        argv,
        cwd=repo_root(),
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    return {
        "command": command,
        "dry_run": False,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def assert_success(result: dict[str, Any], label: str) -> None:
    if int(result["returncode"]) != 0:
        raise SystemExit(f"{label} failed with returncode={result['returncode']}\n{result['stderr']}")


def default_passing_guardrails(path: Path) -> None:
    payload = {
        "bias": {"letter_distribution": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}},
        "compute": {"elapsed_minutes": 10.0},
        "forgetting": {"baseline_score": 0.80, "candidate_score": 0.79},
        "grader_anchor": {"max_abs_drift": 0.02},
        "rubric": {"baseline_percent": 85.0, "candidate_percent": 83.0},
        "sealed_test_access": {"touched": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_sealed_paths(strings: list[str]) -> dict[str, Any]:
    hits = []
    for text in strings:
        for forbidden in SEALED_TEST_PATHS:
            if forbidden in text:
                hits.append({"path": forbidden, "text": text})
    return {
        "forbidden_paths": list(SEALED_TEST_PATHS),
        "hits": hits,
        "passed": not hits,
    }


def append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def finetune_diff() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "diff", "--", "autoresearch/finetune.py"],
        cwd=repo_root(),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"git diff failed: {completed.stderr}")
    return {
        "path": "autoresearch/finetune.py",
        "text": completed.stdout,
        "empty": completed.stdout == "",
    }


def build_finetune_command(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        "autoresearch/finetune.py",
        "--config",
        args.finetune_config,
        "--run-name",
        args.run_name,
    ]
    if args.finetune_dry_run:
        argv.append("--dry-run")
    return argv


def build_gate_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "autoresearch/gate.py",
        "--baseline",
        args.baseline_primary,
        "--candidate",
        args.candidate_primary,
        "--guardrails",
        args.guardrails,
        "--output",
        args.gate_output,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one explicit autoresearch iteration and append a journal row.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--lever-tier", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--predicted-direction", required=True)
    parser.add_argument("--candidate-edit", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--finetune-config", default="autoresearch/configs/smoke-test.yaml")
    parser.add_argument("--finetune-dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--baseline-primary", required=True)
    parser.add_argument("--candidate-primary", required=True)
    parser.add_argument("--guardrails", default="results/autoresearch_guardrails_smoke.json")
    parser.add_argument("--write-passing-guardrails", action="store_true")
    parser.add_argument("--gate-output", default=DEFAULT_GATE_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    program_text = (root / args.program).read_text(encoding="utf-8")
    levers_text = (root / "autoresearch/levers.md").read_text(encoding="utf-8")

    if args.write_passing_guardrails:
        default_passing_guardrails(root / args.guardrails)

    manifest = run_command(
        [sys.executable, "autoresearch/freeze_manifest.py", "check"],
        timeout_seconds=args.timeout_seconds,
        dry_run=False,
    )
    assert_success(manifest, "freeze_manifest check")

    finetune_command = build_finetune_command(args)
    finetune = run_command(
        finetune_command,
        timeout_seconds=args.timeout_seconds,
        dry_run=False,
    )
    assert_success(finetune, "finetune")

    gate_command = build_gate_command(args)
    gate = run_command(gate_command, timeout_seconds=args.timeout_seconds, dry_run=False)
    assert_success(gate, "gate")
    gate_result = load_json(root / args.gate_output)

    audit_inputs = [
        program_text,
        levers_text,
        manifest["command"],
        finetune["command"],
        finetune["stdout"],
        finetune["stderr"],
        gate["command"],
        gate["stdout"],
        gate["stderr"],
        args.baseline_primary,
        args.candidate_primary,
        args.guardrails,
        args.gate_output,
    ]
    sealed_audit = audit_sealed_paths(audit_inputs)
    if not sealed_audit["passed"]:
        raise SystemExit(f"sealed test path audit failed: {sealed_audit['hits']}")

    row = {
        "timestamp_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "run_name": args.run_name,
        "hypothesis": args.hypothesis,
        "lever_tier": args.lever_tier,
        "prior": args.prior,
        "predicted_direction": args.predicted_direction,
        "candidate_edit": args.candidate_edit,
        "diff": finetune_diff(),
        "rationale": args.rationale,
        "program": args.program,
        "finetune_config": args.finetune_config,
        "finetune_dry_run": args.finetune_dry_run,
        "commands": {
            "manifest": manifest["command"],
            "finetune": finetune["command"],
            "gate": gate["command"],
        },
        "artifacts": {
            "baseline_primary": args.baseline_primary,
            "candidate_primary": args.candidate_primary,
            "guardrails": args.guardrails,
            "gate_output": args.gate_output,
        },
        "gate": gate_result,
        "decision": gate_result["decision"],
        "recommended_action": gate_result["recommended_action"],
        "sealed_test_audit": sealed_audit,
    }
    append_journal(root / args.journal, row)

    print(f"journal_appended={args.journal}")
    print(f"decision={row['decision']}")
    print(f"recommended_action={row['recommended_action']}")
    print(f"sealed_test_audit_passed={sealed_audit['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
