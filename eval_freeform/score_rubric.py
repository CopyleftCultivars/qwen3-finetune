#!/usr/bin/env python3
"""LLM rubric scorer for free-form eval draft/canonical items."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from judge_training_data import (  # type: ignore[import-not-found]
    DEFAULT_MODEL,
    _call_judge,
    _get_client,
    _load_api_key,
)


log = logging.getLogger(__name__)

DEFAULT_AUDIT_MODEL = "anthropic/claude-sonnet-4-5"
DEFAULT_CONFIG = "eval_freeform/scorer_config.json"
# judge_training_data._call_judge defaults to 256, sized for its own compact
# 1-5 Likert-score JSON caller; a per-point verdict+justification response
# (4-6 rubric points typical) needs much more headroom or the JSON gets
# truncated before it closes.
RUBRIC_JUDGE_MAX_TOKENS = 1536

# Optional codex-CLI judge backend (2026-07-19, user-directed): grades via
# `codex exec` on the user's ChatGPT-account quota instead of metered
# OpenRouter credits. Grader-version policy applies unchanged: switching
# backend is a grader change, so historical Haiku-graded scores are not
# comparable and baselines must be re-graded under the same backend.
CODEX_JUDGE_MODEL = "gpt-5.6-luna"
CODEX_JUDGE_EFFORT = "medium"
_CODEX_RETRY_DELAYS = [2, 5, 15]


def _call_judge_codex(system: str, prompt: str, model: str) -> str:
    import subprocess
    import tempfile
    import time

    full_prompt = f"{system}\n\n{prompt}"
    for attempt, delay in enumerate([0] + _CODEX_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        with tempfile.TemporaryDirectory(prefix="codex_judge_") as workdir:
            out_path = Path(workdir) / "last_message.txt"
            result = subprocess.run(
                [
                    "codex", "exec",
                    "-C", workdir,
                    "--skip-git-repo-check",
                    "-s", "read-only",
                    "-m", model,
                    "-c", f'model_reasoning_effort="{CODEX_JUDGE_EFFORT}"',
                    "-o", str(out_path),
                    full_prompt,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and out_path.exists():
                raw = out_path.read_text(encoding="utf-8").strip()
                if raw:
                    return raw
        if attempt == len(_CODEX_RETRY_DELAYS):
            raise RuntimeError(
                f"codex judge failed after retries: rc={result.returncode} "
                f"stderr_tail={result.stderr[-500:]}"
            )
        log.warning("codex judge error (attempt %d) — retrying", attempt + 1)
    raise RuntimeError("codex judge: all retries exhausted")

SYSTEM_PROMPT = """\
You are a strict but fair agricultural free-form evaluation judge.
Grade only whether the candidate answer satisfies each rubric point.
Return ONLY valid JSON, with no markdown fences or prose."""

USER_PROMPT = """\
Evaluate the candidate answer against each rubric point independently.

{anchor_block}

Question:
{question}

Candidate answer:
{candidate_answer}

Rubric points:
{rubric_json}

Return this JSON shape exactly:
{{
  "verdicts": [
    {{"point": "same rubric point text", "verdict": true, "justification": "brief reason"}}
  ]
}}"""


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    return config


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_anchors(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    anchors = load_jsonl(path)
    for anchor in anchors:
        for key in ("id", "question", "candidate_answer", "rubric", "expected_verdicts"):
            if key not in anchor:
                raise ValueError(f"anchor {anchor.get('id')}: missing {key}")
        if len(anchor["rubric"]) != len(anchor["expected_verdicts"]):
            raise ValueError(f"anchor {anchor['id']}: expected_verdicts length mismatch")
    return anchors


def format_anchor_block(anchors: list[dict[str, Any]]) -> str:
    if not anchors:
        return ""
    examples: list[str] = [
        "Calibration anchors. Match this grading style before grading the target answer."
    ]
    for anchor in anchors:
        examples.append(
            "\nAnchor question:\n{question}\n\n"
            "Anchor candidate answer:\n{candidate_answer}\n\n"
            "Anchor rubric:\n{rubric}\n\n"
            "Expected JSON verdicts:\n{verdicts}".format(
                question=anchor["question"],
                candidate_answer=anchor["candidate_answer"],
                rubric=json.dumps(anchor["rubric"], ensure_ascii=False, indent=2),
                verdicts=json.dumps(
                    {"verdicts": anchor["expected_verdicts"]},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        )
    return "\n".join(examples) + "\n"


def get_candidate(row: dict[str, Any], candidate_field: str, allow_reference: bool) -> str:
    candidate = row.get(candidate_field)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if allow_reference:
        reference = row.get("reference_answer")
        if isinstance(reference, str) and reference.strip():
            return reference.strip()
    raise ValueError(
        f"item {row.get('id', row.get('_line_number'))}: missing candidate field "
        f"{candidate_field!r}; pass --candidate-from-reference for draft smoke tests"
    )


def validate_item(row: dict[str, Any]) -> None:
    for key in ("id", "question", "rubric", "reference_answer"):
        if key not in row:
            raise ValueError(f"line {row.get('_line_number')}: missing required key {key!r}")
    if not isinstance(row["rubric"], list) or not row["rubric"]:
        raise ValueError(f"item {row['id']}: rubric must be a non-empty list")
    for index, point in enumerate(row["rubric"]):
        if not isinstance(point, dict) or not point.get("point"):
            raise ValueError(f"item {row['id']}: rubric[{index}] missing point text")
        if not isinstance(point.get("weight"), (int, float)):
            raise ValueError(f"item {row['id']}: rubric[{index}] missing numeric weight")


def parse_verdicts(raw: str, rubric: list[dict[str, Any]]) -> list[dict[str, Any]]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in grader response: {raw[:200]}")
    parsed = json.loads(match.group())
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, list):
        raise ValueError("grader response missing verdicts list")
    if len(verdicts) != len(rubric):
        raise ValueError(f"expected {len(rubric)} verdicts, got {len(verdicts)}")

    normalized: list[dict[str, Any]] = []
    for index, (verdict, point) in enumerate(zip(verdicts, rubric, strict=True)):
        if not isinstance(verdict, dict):
            raise ValueError(f"verdict {index} is not an object")
        if not isinstance(verdict.get("verdict"), bool):
            raise ValueError(f"verdict {index} has non-boolean verdict")
        justification = verdict.get("justification", "")
        if not isinstance(justification, str) or not justification.strip():
            raise ValueError(f"verdict {index} missing justification")
        normalized.append(
            {
                "point": str(verdict.get("point") or point["point"]),
                "verdict": bool(verdict["verdict"]),
                "justification": justification.strip(),
                "weight": float(point["weight"]),
            }
        )
    return normalized


def weighted_score(verdicts: list[dict[str, Any]]) -> float:
    total_weight = sum(float(row["weight"]) for row in verdicts)
    if total_weight <= 0:
        return 0.0
    earned = sum(float(row["weight"]) for row in verdicts if row["verdict"])
    return earned / total_weight


def mock_verdicts(row: dict[str, Any], candidate_answer: str) -> list[dict[str, Any]]:
    if isinstance(row.get("expected_verdicts"), list):
        return [
            {
                "point": point["point"],
                "verdict": bool(expected["verdict"]),
                "justification": str(expected.get("justification", "mock anchor expected verdict")),
                "weight": float(point["weight"]),
            }
            for expected, point in zip(row["expected_verdicts"], row["rubric"], strict=True)
        ]

    candidate_terms = set(re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", candidate_answer.lower()))
    verdicts: list[dict[str, Any]] = []
    for point in row["rubric"]:
        point_terms = set(re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", point["point"].lower()))
        overlap = len(candidate_terms & point_terms)
        verdicts.append(
            {
                "point": point["point"],
                "verdict": overlap > 0,
                "justification": f"mock lexical overlap count={overlap}; not a real grade",
                "weight": float(point["weight"]),
            }
        )
    return verdicts


def call_grader(
    row: dict[str, Any],
    candidate_answer: str,
    client: Any,
    model: str,
    mock_grader: bool,
    anchor_block: str = "",
) -> list[dict[str, Any]]:
    if mock_grader:
        return mock_verdicts(row, candidate_answer)
    prompt = USER_PROMPT.format(
        anchor_block=anchor_block,
        question=row["question"],
        candidate_answer=candidate_answer,
        rubric_json=json.dumps(row["rubric"], ensure_ascii=False, indent=2),
    )
    if str(model).startswith("codex/"):
        raw = _call_judge_codex(SYSTEM_PROMPT, prompt, model.split("/", 1)[1])
    else:
        raw = _call_judge(
            SYSTEM_PROMPT, prompt, client, model, max_tokens=RUBRIC_JUDGE_MAX_TOKENS
        )
    return parse_verdicts(raw, row["rubric"])


def build_result(
    row: dict[str, Any],
    candidate_answer: str,
    verdicts: list[dict[str, Any]],
    model: str,
    model_role: str,
    draft_only: bool,
    mock_grader: bool,
    config_path: str | None = None,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "type": row.get("type"),
        "source_doc": row.get("source_doc"),
        "question": row["question"],
        "candidate_answer": candidate_answer,
        "rubric_verdicts": verdicts,
        "weighted_score": round(weighted_score(verdicts), 6),
        "grader_model": model,
        "grader_model_version": model,
        "grader_role": model_role,
        "scorer_config": config_path,
        "mock_grader": mock_grader,
        "draft_only_non_reportable": draft_only,
    }


def expected_anchor_score(anchor: dict[str, Any]) -> float:
    verdicts = []
    for expected, point in zip(anchor["expected_verdicts"], anchor["rubric"], strict=True):
        verdicts.append(
            {
                "point": point["point"],
                "weight": float(point["weight"]),
                "verdict": bool(expected["verdict"]),
                "justification": expected.get("justification", "expected anchor verdict"),
            }
        )
    return weighted_score(verdicts)


def score_anchors(
    anchors: list[dict[str, Any]],
    client: Any,
    routine_model: str,
    audit_model: str,
    mock_grader: bool,
    anchor_block: str,
    drift_threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    drift_alerts: list[dict[str, Any]] = []
    for anchor in anchors:
        expected = expected_anchor_score(anchor)
        for role, model in (("routine", routine_model), ("audit", audit_model)):
            verdicts = call_grader(
                anchor,
                anchor["candidate_answer"],
                client,
                model,
                mock_grader,
                anchor_block,
            )
            actual = weighted_score(verdicts)
            delta = round(actual - expected, 6)
            record = {
                "id": anchor["id"],
                "type": anchor.get("type"),
                "grader_role": role,
                "grader_model": model,
                "grader_model_version": model,
                "expected_score": round(expected, 6),
                "actual_score": round(actual, 6),
                "delta": delta,
                "drift_threshold": drift_threshold,
                "drift_alert": abs(delta) > drift_threshold,
                "mock_grader": mock_grader,
            }
            rows.append(record)
            if record["drift_alert"]:
                drift_alerts.append(record)
    return {
        "anchor_count": len(anchors),
        "drift_threshold": drift_threshold,
        "drift_alert_count": len(drift_alerts),
        "drift_alerts": drift_alerts,
        "anchors": rows,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score free-form candidate answers against item rubrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Free-form item JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with rubric verdicts")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Pinned scorer config JSON")
    parser.add_argument("--anchors", default=None, help="Anchor JSONL. Defaults to config anchor_path.")
    parser.add_argument("--anchor-report", default=None, help="Anchor drift report JSON")
    parser.add_argument("--candidate-field", default="candidate_answer")
    parser.add_argument(
        "--candidate-from-reference",
        action="store_true",
        help="Use reference_answer when candidate field is absent; for smoke tests only.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Routine OpenRouter grader model")
    parser.add_argument("--audit-model", default=DEFAULT_AUDIT_MODEL)
    parser.add_argument(
        "--judge-backend",
        choices=("openrouter", "codex"),
        default="openrouter",
        help=(
            "codex routes BOTH grader roles through `codex exec` "
            f"({CODEX_JUDGE_MODEL} @ {CODEX_JUDGE_EFFORT}) on the ChatGPT-account "
            "quota; a backend switch is a grader-version change (see config policy)"
        ),
    )
    parser.add_argument("--audit-first", type=int, default=0, help="Audit the first N rows with audit model")
    parser.add_argument("--openrouter-key", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock-grader", action="store_true", help="No API calls; deterministic schema smoke")
    parser.add_argument("--draft-only", action="store_true", help="Mark output as draft-only/non-reportable")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    if args.model == DEFAULT_MODEL and config.get("routine_model"):
        args.model = str(config["routine_model"])
    if args.audit_model == DEFAULT_AUDIT_MODEL and config.get("audit_model"):
        args.audit_model = str(config["audit_model"])
    anchor_path = Path(args.anchors or config.get("anchor_path", "")) if (args.anchors or config.get("anchor_path")) else None
    anchor_report_path = Path(
        args.anchor_report
        or config.get("anchor_report_path", "")
        or str(Path(args.output).with_suffix(".anchors.json"))
    )
    drift_threshold = float(config.get("anchor_drift_threshold", 0.10))

    rows = load_jsonl(Path(args.input), args.limit)
    if not rows:
        raise SystemExit(f"no rows loaded from {args.input}")
    for row in rows:
        validate_item(row)

    client = None
    if args.judge_backend == "codex":
        args.model = f"codex/{CODEX_JUDGE_MODEL}"
        args.audit_model = f"codex/{CODEX_JUDGE_MODEL}"
    elif not args.mock_grader:
        api_key = _load_api_key(args.openrouter_key)
        client, _ = _get_client(api_key, args.model)

    anchors = load_anchors(anchor_path)
    anchor_block = format_anchor_block(anchors)
    anchor_report = score_anchors(
        anchors,
        client,
        args.model,
        args.audit_model,
        args.mock_grader,
        anchor_block,
        drift_threshold,
    )
    anchor_report_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_report_path.write_text(json.dumps(anchor_report, indent=2, sort_keys=True) + "\n")
    if anchor_report["drift_alert_count"]:
        raise RuntimeError(
            f"anchor drift alerts: {anchor_report['drift_alert_count']} "
            f"(see {anchor_report_path})"
        )

    results: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def process(index: int, row: dict[str, Any]) -> None:
        candidate = get_candidate(row, args.candidate_field, args.candidate_from_reference)
        verdicts = call_grader(row, candidate, client, args.model, args.mock_grader, anchor_block)
        item_results = [
            build_result(
                row,
                candidate,
                verdicts,
                args.model,
                "routine",
                args.draft_only,
                args.mock_grader,
                str(config_path) if config_path else None,
            )
        ]
        if index < args.audit_first:
            audit_verdicts = call_grader(
                row,
                candidate,
                client,
                args.audit_model,
                args.mock_grader,
                anchor_block,
            )
            item_results.append(
                build_result(
                    row,
                    candidate,
                    audit_verdicts,
                    args.audit_model,
                    "audit",
                    args.draft_only,
                    args.mock_grader,
                    str(config_path) if config_path else None,
                )
            )
        with write_lock:
            results.extend(item_results)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, index, row): index for index, row in enumerate(rows)}
        for future in as_completed(futures):
            future.result()

    results.sort(key=lambda row: (str(row["id"]), str(row["grader_role"])))
    write_jsonl(Path(args.output), results)
    print(f"wrote {args.output}")
    print(f"items_loaded={len(rows)}")
    print(f"result_rows={len(results)}")
    print(f"routine_model={args.model}")
    print(f"audit_model={args.audit_model}")
    print(f"anchor_count={len(anchors)}")
    print(f"anchor_report={anchor_report_path}")
    print(f"anchor_drift_alert_count={anchor_report['drift_alert_count']}")
    print(f"mock_grader={args.mock_grader}")
    print(f"draft_only_non_reportable={args.draft_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
