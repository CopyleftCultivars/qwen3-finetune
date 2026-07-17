#!/usr/bin/env python3
"""Multi-stream batch journal schema, validation, and tournament resolution.

Phase 5 M1 (chainlink #59). Loop-level multi-stream tournaments: one research
cycle proposes N candidate streams (default 2-4), evaluates each against the
same frozen baseline under the same fixed budget, gates each independently,
and accepts at most one winner. This module defines the batch journal row
format, validates rows (batch and legacy single-stream), and resolves
tournament outcomes deterministically. It does not train, gate, or revert;
frozen objective/gate semantics are consumed, never modified.

Journal row formats
===================

``autoresearch/journal.jsonl`` is append-only and now holds two row types,
discriminated by the ``format`` key:

Legacy single-stream rows (no ``format`` key)
    The shape written by run_loop.py today. They remain valid forever and are
    recognized by their ``run_name`` + ``gate`` + ``decision`` +
    ``sealed_test_audit`` keys.

Batch rows: ``"format": "autoresearch_journal_batch_v1"``
    Top-level fields:

    - ``format``            -- discriminator, always the string above.
    - ``timestamp_utc``     -- ISO-8601 UTC timestamp.
    - ``batch_id``          -- unique batch identifier.
    - ``baseline_run_name`` -- champion every stream is compared against.
    - ``streams``           -- list of stream records (see below), >= 1.
    - ``accepted_stream_id``-- winning stream_id, or null.
    - ``batch_outcome``     -- one of ``accepted``, ``no_winner``, ``paused``,
                               ``rejected_sealed_test``.
    - ``sealed_test_audit`` -- batch-level audit, same shape as the
                               single-stream field (``forbidden_paths`` /
                               ``hits`` / ``passed``).

    Per-stream record fields (all required unless noted):

    - ``stream_id``      -- unique within the batch.
    - ``run_name``       -- per-stream run/output name.
    - ``hypothesis``     -- one hypothesis per stream.
    - ``lever_tier``     -- lever tier cited from levers.md.
    - ``config_path``    -- finetune config path or patch summary.
    - ``commands``       -- {label: command string} actually executed.
    - ``metrics``        -- at least ``baseline_primary`` and
                            ``candidate_primary`` (floats; null for failed runs).
    - ``guardrails``     -- guardrail report path or embedded object.
    - ``gate_decision``  -- full gate.py result object
                            (``autoresearch_gate_decision_v1``); null only when
                            ``final_outcome`` is ``failed_run``.
    - ``rank_in_batch``  -- 1-based rank by candidate primary (descending,
                            ties broken by ascending stream_id); null for
                            streams without a numeric candidate primary.
    - ``final_outcome``  -- one of ``accepted``, ``rejected_by_gate``,
                            ``paused``, ``defeated_in_tournament``,
                            ``failed_run``.

Tournament semantics (deterministic, in priority order)
=======================================================

1. Batch sealed-test audit failed -> ``rejected_sealed_test``: nothing is
   accepted; gate-accepted streams are demoted to ``rejected_by_gate``.
2. Any stream gate decision is ``pause`` -> ``paused``: pause blocks
   auto-accept batch-wide; every non-failed stream's ``final_outcome`` is
   recorded as ``paused`` pending human review.
3. No stream gate decision is ``accept`` -> ``no_winner``: gate-rejected
   streams are ``rejected_by_gate``.
4. Otherwise -> ``accepted``: the gate-accepted stream with the highest
   ``metrics.candidate_primary`` wins (ties broken by ascending stream_id);
   losers that passed the gate are ``defeated_in_tournament``, gate-rejected
   streams are ``rejected_by_gate``.

Streams with ``gate_decision`` null are ``failed_run`` in every case and never
win. At most one stream is accepted per batch, always.

Self-test: ``python autoresearch/journal_batch.py --self-test`` validates the
fixtures under autoresearch/fixtures/multistream/, re-resolves each tournament
from its gate decisions alone, asserts the stored outcomes match, and checks
that legacy rows in autoresearch/journal.jsonl still validate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BATCH_FORMAT = "autoresearch_journal_batch_v1"
GATE_DECISION_FORMAT = "autoresearch_gate_decision_v1"
FIXTURES_DIR = Path("autoresearch/fixtures/multistream")
DEFAULT_JOURNAL = Path("autoresearch/journal.jsonl")

BATCH_OUTCOMES = ("accepted", "no_winner", "paused", "rejected_sealed_test")
STREAM_OUTCOMES = (
    "accepted",
    "rejected_by_gate",
    "paused",
    "defeated_in_tournament",
    "failed_run",
)

_REQUIRED_BATCH_KEYS = (
    "format",
    "timestamp_utc",
    "batch_id",
    "baseline_run_name",
    "streams",
    "accepted_stream_id",
    "batch_outcome",
    "sealed_test_audit",
)
_REQUIRED_STREAM_KEYS = (
    "stream_id",
    "run_name",
    "hypothesis",
    "lever_tier",
    "config_path",
    "commands",
    "metrics",
    "guardrails",
    "gate_decision",
    "rank_in_batch",
    "final_outcome",
)
_REQUIRED_LEGACY_KEYS = ("run_name", "hypothesis", "gate", "decision", "sealed_test_audit")


def row_kind(row: dict[str, Any]) -> str:
    """Classify a journal row: 'batch', 'legacy_single', or 'unknown'."""
    if row.get("format") == BATCH_FORMAT:
        return "batch"
    if "format" not in row and all(key in row for key in _REQUIRED_LEGACY_KEYS):
        return "legacy_single"
    return "unknown"


def validate_legacy_row(row: dict[str, Any]) -> list[str]:
    errors = [f"missing key: {key}" for key in _REQUIRED_LEGACY_KEYS if key not in row]
    if "format" in row:
        errors.append("legacy rows must not carry a format key")
    return errors


def validate_batch_row(row: dict[str, Any]) -> list[str]:
    """Return a list of schema violations; empty means the row is valid."""
    errors = [f"missing key: {key}" for key in _REQUIRED_BATCH_KEYS if key not in row]
    if errors:
        return errors
    if row["format"] != BATCH_FORMAT:
        errors.append(f"format must be {BATCH_FORMAT}")
    if row["batch_outcome"] not in BATCH_OUTCOMES:
        errors.append(f"batch_outcome must be one of {BATCH_OUTCOMES}")
    audit = row["sealed_test_audit"]
    if not isinstance(audit, dict) or not {"forbidden_paths", "hits", "passed"} <= set(audit):
        errors.append("sealed_test_audit must carry forbidden_paths/hits/passed")

    streams = row["streams"]
    if not isinstance(streams, list) or not streams:
        return errors + ["streams must be a non-empty list"]
    seen_ids: set[str] = set()
    accepted = [s for s in streams if s.get("final_outcome") == "accepted"]
    if len(accepted) > 1:
        errors.append("at most one stream may be accepted per batch")
    for index, stream in enumerate(streams):
        label = f"streams[{index}]"
        for key in _REQUIRED_STREAM_KEYS:
            if key not in stream:
                errors.append(f"{label}: missing key: {key}")
        stream_id = stream.get("stream_id")
        if stream_id in seen_ids:
            errors.append(f"{label}: duplicate stream_id {stream_id!r}")
        seen_ids.add(stream_id)
        outcome = stream.get("final_outcome")
        if outcome not in STREAM_OUTCOMES:
            errors.append(f"{label}: final_outcome must be one of {STREAM_OUTCOMES}")
        gate = stream.get("gate_decision")
        if gate is None:
            if outcome != "failed_run":
                errors.append(f"{label}: gate_decision may be null only for failed_run")
        elif not isinstance(gate, dict) or gate.get("format") != GATE_DECISION_FORMAT:
            errors.append(f"{label}: gate_decision must be a {GATE_DECISION_FORMAT} object")

    accepted_id = row["accepted_stream_id"]
    if accepted_id is not None:
        if accepted_id not in seen_ids:
            errors.append(f"accepted_stream_id {accepted_id!r} is not a stream in this batch")
        if row["batch_outcome"] != "accepted":
            errors.append("accepted_stream_id set but batch_outcome is not 'accepted'")
    elif row["batch_outcome"] == "accepted":
        errors.append("batch_outcome 'accepted' requires accepted_stream_id")
    return errors


def _candidate_primary(stream: dict[str, Any]) -> float | None:
    value = (stream.get("metrics") or {}).get("candidate_primary")
    return float(value) if isinstance(value, (int, float)) else None


def rank_streams(streams: list[dict[str, Any]]) -> dict[str, int | None]:
    """1-based rank by candidate primary descending; ties by stream_id ascending."""
    scored = [
        (stream["stream_id"], _candidate_primary(stream))
        for stream in streams
    ]
    ordered = sorted(
        (entry for entry in scored if entry[1] is not None),
        key=lambda entry: (-entry[1], entry[0]),
    )
    ranks: dict[str, int | None] = {stream_id: None for stream_id, _ in scored}
    for position, (stream_id, _) in enumerate(ordered, start=1):
        ranks[stream_id] = position
    return ranks


def resolve_tournament(
    streams: list[dict[str, Any]],
    sealed_audit_passed: bool,
) -> dict[str, Any]:
    """Compute batch_outcome, accepted_stream_id, rank and final_outcome per
    stream from gate decisions alone, per the module-docstring semantics."""
    ranks = rank_streams(streams)
    decisions = {
        stream["stream_id"]: (stream.get("gate_decision") or {}).get("decision")
        for stream in streams
    }
    outcomes: dict[str, str] = {}
    for stream_id, decision in decisions.items():
        if decision is None:
            outcomes[stream_id] = "failed_run"
        elif decision == "accept":
            outcomes[stream_id] = "accepted"  # provisional; demoted below
        elif decision == "pause":
            outcomes[stream_id] = "paused"
        else:
            outcomes[stream_id] = "rejected_by_gate"

    accepted_ids = [sid for sid, outcome in outcomes.items() if outcome == "accepted"]
    if not sealed_audit_passed:
        batch_outcome = "rejected_sealed_test"
        accepted_stream_id = None
        for stream_id in accepted_ids:
            outcomes[stream_id] = "rejected_by_gate"
    elif any(outcome == "paused" for outcome in outcomes.values()):
        batch_outcome = "paused"
        accepted_stream_id = None
        for stream_id, outcome in outcomes.items():
            if outcome != "failed_run":
                outcomes[stream_id] = "paused"
    elif not accepted_ids:
        batch_outcome = "no_winner"
        accepted_stream_id = None
    else:
        batch_outcome = "accepted"
        accepted_stream_id = min(
            accepted_ids,
            key=lambda sid: (
                ranks[sid] if ranks[sid] is not None else float("inf"),
                sid,
            ),
        )
        for stream_id in accepted_ids:
            if stream_id != accepted_stream_id:
                outcomes[stream_id] = "defeated_in_tournament"

    return {
        "batch_outcome": batch_outcome,
        "accepted_stream_id": accepted_stream_id,
        "rank_in_batch": ranks,
        "final_outcome": outcomes,
    }


def read_journal(path: Path) -> list[tuple[str, dict[str, Any], list[str]]]:
    """Read a journal file; return (kind, row, errors) per line."""
    results = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        kind = row_kind(row)
        if kind == "batch":
            errors = validate_batch_row(row)
        elif kind == "legacy_single":
            errors = validate_legacy_row(row)
        else:
            errors = ["unrecognized journal row"]
        results.append((kind, row, errors))
    return results


def run_self_test(fixtures_dir: Path, journal_path: Path) -> int:
    failures: list[str] = []
    fixture_paths = sorted(fixtures_dir.glob("*.json"))
    if not fixture_paths:
        raise SystemExit(f"no fixtures found under {fixtures_dir}")

    for fixture_path in fixture_paths:
        row = json.loads(fixture_path.read_text(encoding="utf-8"))
        errors = validate_batch_row(row)
        if errors:
            failures.append(f"{fixture_path.name}: schema errors: {errors}")
            continue
        resolved = resolve_tournament(
            row["streams"], bool(row["sealed_test_audit"]["passed"])
        )
        checks = [
            ("batch_outcome", row["batch_outcome"], resolved["batch_outcome"]),
            ("accepted_stream_id", row["accepted_stream_id"], resolved["accepted_stream_id"]),
        ]
        for stream in row["streams"]:
            sid = stream["stream_id"]
            checks.append(
                (f"{sid}.final_outcome", stream["final_outcome"], resolved["final_outcome"][sid])
            )
            checks.append(
                (f"{sid}.rank_in_batch", stream["rank_in_batch"], resolved["rank_in_batch"][sid])
            )
        mismatches = [
            f"{field}: stored={stored!r} resolved={live!r}"
            for field, stored, live in checks
            if stored != live
        ]
        status = "ok" if not mismatches else "MISMATCH"
        print(f"{fixture_path.name}: {status}")
        if mismatches:
            failures.extend(f"{fixture_path.name}: {m}" for m in mismatches)

    if journal_path.exists():
        for index, (kind, _row, errors) in enumerate(read_journal(journal_path)):
            print(f"journal row {index}: kind={kind} errors={errors or 'none'}")
            if errors:
                failures.append(f"journal row {index} ({kind}): {errors}")

    if failures:
        print(json.dumps(failures, indent=2))
        return 1
    print(f"self-test passed: {len(fixture_paths)} fixtures, journal backwards-compatible")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate multi-stream batch journal rows and fixtures."
    )
    parser.add_argument("--fixtures-dir", default=str(FIXTURES_DIR))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", metavar="PATH", default=None, help="Validate one row/JSONL file")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test(Path(args.fixtures_dir), Path(args.journal))
    if args.validate:
        bad = 0
        for kind, _row, errors in read_journal(Path(args.validate)):
            print(f"kind={kind} errors={errors or 'none'}")
            bad += bool(errors)
        return 1 if bad else 0
    parser.error("pass --self-test or --validate PATH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
