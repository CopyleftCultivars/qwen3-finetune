#!/usr/bin/env python3
"""One-iteration autoresearch loop orchestrator.

This script does not propose edits itself. It runs one explicit candidate
iteration, applies the frozen gate, and appends an audit-ready journal row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

import journal_batch


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


#: Single-mode arguments that argparse used to mark required=True. They stay
#: required in single mode (enforced after parsing) but must be absent in
#: batch mode, where stream spec files carry them per stream.
SINGLE_MODE_REQUIRED = (
    "run_name",
    "hypothesis",
    "lever_tier",
    "prior",
    "predicted_direction",
    "candidate_edit",
    "rationale",
    "baseline_primary",
    "candidate_primary",
)

STREAM_SPEC_REQUIRED = (
    "run_name",
    "hypothesis",
    "lever_tier",
    "prior",
    "predicted_direction",
    "candidate_edit",
    "rationale",
    "baseline_primary",
    "candidate_primary",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one explicit autoresearch iteration and append a journal row.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--lever-tier", default=None)
    parser.add_argument("--prior", default=None)
    parser.add_argument("--predicted-direction", default=None)
    parser.add_argument("--candidate-edit", default=None)
    parser.add_argument("--rationale", default=None)
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--finetune-config", default="autoresearch/configs/smoke-test.yaml")
    parser.add_argument("--finetune-dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--baseline-primary", default=None)
    parser.add_argument("--candidate-primary", default=None)
    parser.add_argument("--guardrails", default="results/autoresearch_guardrails_smoke.json")
    parser.add_argument("--write-passing-guardrails", action="store_true")
    parser.add_argument("--gate-output", default=DEFAULT_GATE_OUTPUT)

    # Batch tournament mode (Phase 5 M2, chainlink #60). Loop-level
    # multi-stream: sequential execution, at most one accepted stream.
    parser.add_argument("--batch-id", default=None)
    parser.add_argument(
        "--stream-configs",
        default=None,
        help="Comma-separated per-stream YAML spec paths (batch mode)",
    )
    parser.add_argument("--max-streams", type=int, default=4)
    parser.add_argument(
        "--execution-mode",
        choices=("sequential", "parallel"),
        default="sequential",
    )
    parser.add_argument(
        "--accept-policy",
        choices=("best_passing", "first_passing"),
        default="best_passing",
    )
    parser.add_argument(
        "--journal-mode",
        choices=("single", "batch"),
        default="single",
    )
    parser.add_argument("--baseline-run-name", default="v3-clean")
    return parser


def load_stream_spec(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a flat YAML map")
    missing = [key for key in STREAM_SPEC_REQUIRED if not data.get(key)]
    if missing:
        raise SystemExit(f"{path}: missing stream key(s): {', '.join(missing)}")
    run_name = str(data["run_name"])
    spec = {key: str(data[key]) for key in STREAM_SPEC_REQUIRED}
    spec.update(
        {
            "stream_id": str(data.get("stream_id") or path.stem),
            "spec_path": path.as_posix(),
            "finetune_config": str(data.get("finetune_config") or args.finetune_config),
            "finetune_dry_run": bool(data.get("finetune_dry_run", True)),
            "guardrails": str(
                data.get("guardrails") or f"results/autoresearch_guardrails_{run_name}.json"
            ),
            "gate_output": str(
                data.get("gate_output") or f"results/autoresearch_gate_{run_name}.json"
            ),
            "write_passing_guardrails": bool(data.get("write_passing_guardrails", False)),
            "eval_commands": list(data.get("eval_commands") or []),
            "guardrails_inject_compute": bool(
                data.get("guardrails_inject_compute", False)
            ),
        }
    )
    if not all(isinstance(cmd, str) for cmd in spec["eval_commands"]):
        raise SystemExit(f"{path}: eval_commands must be a list of command strings")
    return spec


def run_stream(
    spec: dict[str, Any],
    args: argparse.Namespace,
    root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Run manifest check, finetune, and gate for one stream.

    Failures do not abort the batch: the stream is journaled as failed_run
    (gate_decision null) and the tournament continues.
    """
    if spec["write_passing_guardrails"]:
        default_passing_guardrails(root / spec["guardrails"])

    stream_start = time.monotonic()
    commands: dict[str, str] = {}
    audit_strings: list[str] = []
    failure_stage = None
    gate_result: dict[str, Any] | None = None

    steps: list[tuple[str, list[str]]] = [
        (
            "manifest",
            [sys.executable, "autoresearch/freeze_manifest.py", "check"],
        ),
        (
            "finetune",
            [
                sys.executable,
                "autoresearch/finetune.py",
                "--config",
                spec["finetune_config"],
                "--run-name",
                spec["run_name"],
            ]
            + (["--dry-run"] if spec["finetune_dry_run"] else []),
        ),
    ]
    for index, command_text in enumerate(spec["eval_commands"]):
        steps.append((f"eval_{index}", shlex.split(command_text)))

    stage_minutes: dict[str, float] = {}
    for stage, argv in steps:
        stage_start = time.monotonic()
        result = run_command(argv, timeout_seconds=args.timeout_seconds, dry_run=False)
        stage_minutes[stage] = round((time.monotonic() - stage_start) / 60.0, 2)
        commands[stage] = result["command"]
        audit_strings.extend([result["command"], result["stdout"], result["stderr"]])
        if int(result["returncode"]) != 0:
            failure_stage = stage
            print(f"stream {spec['stream_id']}: {stage} failed -> failed_run")
            break

    elapsed_minutes = round((time.monotonic() - stream_start) / 60.0, 2)
    train_minutes = stage_minutes.get("finetune")
    if failure_stage is None:
        if spec["guardrails_inject_compute"]:
            # The compute guard's intent (program.md) is "do not let longer
            # TRAINING masquerade as a discovery", so the finetune stage's
            # wall clock is what the frozen guard judges (chainlink #76).
            # Total stream cost (scoring + periodic-guard measurement) is
            # recorded alongside for cost visibility, not gated.
            guardrails_path = root / spec["guardrails"]
            payload = load_json(guardrails_path)
            payload.setdefault("compute", {})["elapsed_minutes"] = train_minutes
            payload["compute"]["total_stream_minutes"] = elapsed_minutes
            payload["compute"]["injected_by"] = "run_loop_batch_mode"
            guardrails_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        gate_argv = [
            sys.executable,
            "autoresearch/gate.py",
            "--baseline",
            spec["baseline_primary"],
            "--candidate",
            spec["candidate_primary"],
            "--guardrails",
            spec["guardrails"],
            "--output",
            spec["gate_output"],
        ]
        result = run_command(
            gate_argv, timeout_seconds=args.timeout_seconds, dry_run=False
        )
        commands["gate"] = result["command"]
        audit_strings.extend([result["command"], result["stdout"], result["stderr"]])
        if int(result["returncode"]) != 0:
            failure_stage = "gate"
            print(f"stream {spec['stream_id']}: gate failed -> failed_run")
        else:
            gate_result = load_json(root / spec["gate_output"])
        elapsed_minutes = round((time.monotonic() - stream_start) / 60.0, 2)

    metrics = {
        "baseline_primary": gate_result.get("baseline_primary") if gate_result else None,
        "candidate_primary": gate_result.get("candidate_primary") if gate_result else None,
        "primary_delta": gate_result.get("primary_delta") if gate_result else None,
    }
    audit_strings.extend(
        [
            spec["hypothesis"],
            spec["candidate_edit"],
            spec["rationale"],
            spec["baseline_primary"],
            spec["candidate_primary"],
            spec["guardrails"],
            spec["gate_output"],
            spec["finetune_config"],
        ]
    )
    record = {
        "stream_id": spec["stream_id"],
        "run_name": spec["run_name"],
        "hypothesis": spec["hypothesis"],
        "lever_tier": spec["lever_tier"],
        "prior": spec["prior"],
        "predicted_direction": spec["predicted_direction"],
        "candidate_edit": spec["candidate_edit"],
        "rationale": spec["rationale"],
        "spec_path": spec["spec_path"],
        "config_path": spec["finetune_config"],
        "finetune_dry_run": spec["finetune_dry_run"],
        "commands": commands,
        "metrics": metrics,
        "guardrails": spec["guardrails"],
        "gate_decision": gate_result,
        "failure_stage": failure_stage,
        "elapsed_minutes": elapsed_minutes,
        "train_minutes": train_minutes,
        "stage_minutes": stage_minutes,
        "rank_in_batch": None,
        "final_outcome": None,
    }
    return record, audit_strings


def run_batch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    root = repo_root()
    program_text = (root / args.program).read_text(encoding="utf-8")
    levers_text = (root / "autoresearch/levers.md").read_text(encoding="utf-8")

    if args.execution_mode == "parallel":
        raise SystemExit(
            "--execution-mode parallel is not implemented until chainlink #62; "
            "use sequential"
        )
    spec_paths = [
        Path(part.strip()) for part in args.stream_configs.split(",") if part.strip()
    ]
    if not spec_paths:
        parser.error("--stream-configs is empty")
    if len(spec_paths) > args.max_streams:
        parser.error(
            f"{len(spec_paths)} stream configs exceed --max-streams={args.max_streams}"
        )
    specs = [load_stream_spec(root / path, args) for path in spec_paths]
    for field in ("stream_id", "run_name", "gate_output"):
        values = [spec[field] for spec in specs]
        if len(set(values)) != len(values):
            raise SystemExit(f"stream {field} values collide across streams: {values}")

    records: list[dict[str, Any]] = []
    audit_inputs = [program_text, levers_text]
    for spec in specs:
        record, audit_strings = run_stream(spec, args, root)
        records.append(record)
        audit_inputs.extend(audit_strings)

    sealed_audit = audit_sealed_paths(audit_inputs)
    resolved = journal_batch.resolve_tournament(
        records, sealed_audit["passed"], accept_policy=args.accept_policy
    )
    for record in records:
        stream_id = record["stream_id"]
        record["rank_in_batch"] = resolved["rank_in_batch"][stream_id]
        record["final_outcome"] = resolved["final_outcome"][stream_id]

    row = {
        "format": journal_batch.BATCH_FORMAT,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "batch_id": args.batch_id,
        "baseline_run_name": args.baseline_run_name,
        "accept_policy": args.accept_policy,
        "execution_mode": args.execution_mode,
        "program": args.program,
        "streams": records,
        "accepted_stream_id": resolved["accepted_stream_id"],
        "batch_outcome": resolved["batch_outcome"],
        "sealed_test_audit": sealed_audit,
    }
    errors = journal_batch.validate_batch_row(row)
    if errors:
        raise SystemExit(f"internal error: batch row failed schema validation: {errors}")
    append_journal(root / args.journal, row)

    print(f"journal_appended={args.journal}")
    print(f"batch_id={args.batch_id}")
    print(f"batch_outcome={row['batch_outcome']}")
    print(f"accepted_stream_id={row['accepted_stream_id']}")
    for record in records:
        print(
            f"stream {record['stream_id']}: outcome={record['final_outcome']} "
            f"rank={record['rank_in_batch']}"
        )
    print(f"sealed_test_audit_passed={sealed_audit['passed']}")
    return 0 if sealed_audit["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    batch_requested = (
        args.journal_mode == "batch"
        or args.batch_id is not None
        or args.stream_configs is not None
    )
    if batch_requested:
        if args.journal_mode != "batch":
            parser.error("batch flags require --journal-mode batch")
        if args.batch_id is None or args.stream_configs is None:
            parser.error("batch mode requires --batch-id and --stream-configs")
        set_single = [
            name
            for name in SINGLE_MODE_REQUIRED
            if getattr(args, name) is not None
        ]
        if set_single:
            parser.error(
                "single-stream flags are not valid in batch mode "
                f"(set per stream in the spec files): {', '.join(set_single)}"
            )
        return run_batch(args, parser)

    missing = [name for name in SINGLE_MODE_REQUIRED if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error(f"the following arguments are required: {flags}")

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
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
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
