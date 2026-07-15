#!/usr/bin/env python3
"""LLM rubric scorer for free-form eval draft/canonical items.

Smoke-test mode deliberately supports draft files and reference answers as
candidate answers. Any output from draft runs is non-reportable until #41 closes.

TODO(#18): add calibrated anchor examples and grader-drift alerts after the
human-reviewed canonical set exists.
"""

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

SYSTEM_PROMPT = """\
You are a strict but fair agricultural free-form evaluation judge.
Grade only whether the candidate answer satisfies each rubric point.
Return ONLY valid JSON, with no markdown fences or prose."""

USER_PROMPT = """\
Evaluate the candidate answer against each rubric point independently.

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
) -> list[dict[str, Any]]:
    if mock_grader:
        return mock_verdicts(row, candidate_answer)
    prompt = USER_PROMPT.format(
        question=row["question"],
        candidate_answer=candidate_answer,
        rubric_json=json.dumps(row["rubric"], ensure_ascii=False, indent=2),
    )
    raw = _call_judge(SYSTEM_PROMPT, prompt, client, model)
    return parse_verdicts(raw, row["rubric"])


def build_result(
    row: dict[str, Any],
    candidate_answer: str,
    verdicts: list[dict[str, Any]],
    model: str,
    model_role: str,
    draft_only: bool,
    mock_grader: bool,
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
        "mock_grader": mock_grader,
        "draft_only_non_reportable": draft_only,
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
    parser.add_argument("--candidate-field", default="candidate_answer")
    parser.add_argument(
        "--candidate-from-reference",
        action="store_true",
        help="Use reference_answer when candidate field is absent; for smoke tests only.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Routine OpenRouter grader model")
    parser.add_argument("--audit-model", default=DEFAULT_AUDIT_MODEL)
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

    rows = load_jsonl(Path(args.input), args.limit)
    if not rows:
        raise SystemExit(f"no rows loaded from {args.input}")
    for row in rows:
        validate_item(row)

    client = None
    if not args.mock_grader:
        api_key = _load_api_key(args.openrouter_key)
        client, _ = _get_client(api_key, args.model)

    results: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def process(index: int, row: dict[str, Any]) -> None:
        candidate = get_candidate(row, args.candidate_field, args.candidate_from_reference)
        verdicts = call_grader(row, candidate, client, args.model, args.mock_grader)
        item_results = [
            build_result(row, candidate, verdicts, args.model, "routine", args.draft_only, args.mock_grader)
        ]
        if index < args.audit_first:
            audit_verdicts = call_grader(row, candidate, client, args.audit_model, args.mock_grader)
            item_results.append(
                build_result(
                    row,
                    candidate,
                    audit_verdicts,
                    args.audit_model,
                    "audit",
                    args.draft_only,
                    args.mock_grader,
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
    print(f"mock_grader={args.mock_grader}")
    print(f"draft_only_non_reportable={args.draft_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
