#!/usr/bin/env python3
"""Score mean gold-token log-likelihood for MCQ validation rows.

This is the MCQ analog of a validation loss scalar: it runs a deterministic
forward pass, scores P(gold answer letter | forced-empty-<think> MCQ prompt),
and writes per-item gold-token log-likelihood plus aggregate means.

By default the script reuses the existing logit plumbing from the sibling
benchmark-eval-workspace diagnostics package:

    python autoresearch/score_mcq_ll.py \
        --model outputs/merged-model-v3-clean \
        --dataset data/val_v3.jsonl \
        --output results/mcq_gold_ll_v3_clean.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
DEFAULT_DIAGNOSTICS_DIR = Path("../benchmark-eval-workspace/diagnostics")

_OPTION_BLOCK_RE = re.compile(
    r"(?ms)"
    r"(?P<question>.*?)"
    r"(?:^|\n)\s*A\.\s*(?P<A>.*?)\n"
    r"\s*B\.\s*(?P<B>.*?)\n"
    r"\s*C\.\s*(?P<C>.*?)\n"
    r"\s*D\.\s*(?P<D>.*?)(?:\n\s*(?:Answer|Correct answer)\b|\s*$)"
)
_LETTER_RE = re.compile(r"\b([ABCD])\b")


class McqRecordError(ValueError):
    """Raised when a JSONL row cannot be normalized into a scorable MCQ."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def import_diagnostics_helpers(diagnostics_dir: Path):
    """Import and return the shared diagnostics helper functions."""
    resolved = diagnostics_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"diagnostics directory not found: {resolved}. "
            "Pass --diagnostics-dir pointing at benchmark-eval-workspace/diagnostics."
        )
    sys.path.insert(0, str(resolved))
    from utils import (  # type: ignore[import-not-found]
        format_mcq_prompt,
        get_letter_token_ids,
        load_benchmark,
        load_model_and_tokenizer,
        score_symbol,
    )

    return {
        "format_mcq_prompt": format_mcq_prompt,
        "get_letter_token_ids": get_letter_token_ids,
        "load_benchmark": load_benchmark,
        "load_model_and_tokenizer": load_model_and_tokenizer,
        "score_symbol": score_symbol,
    }


def _as_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _extract_gold_from_output(output: str) -> str:
    answer_region = output.rsplit("</think>", maxsplit=1)[-1]
    matches = _LETTER_RE.findall(answer_region)
    if not matches:
        matches = _LETTER_RE.findall(output)
    if not matches:
        raise McqRecordError("could not extract A/B/C/D gold letter from output")
    return matches[-1]


def _parse_instruction_mcq(instruction: str) -> dict[str, str]:
    matches = list(_OPTION_BLOCK_RE.finditer(instruction.strip()))
    if not matches:
        raise McqRecordError("instruction does not contain an A/B/C/D option block")

    match = matches[-1]
    raw_question = match.group("question").strip()
    question = raw_question.split("\n\n")[-1].strip()
    item = {"question": question}
    for letter in LETTERS:
        item[letter] = " ".join(match.group(letter).strip().split())
    if not item["question"]:
        raise McqRecordError("empty parsed question")
    return item


def normalize_mcq_record(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    """Normalize supported JSONL rows into diagnostics benchmark item shape."""
    if all(key in row for key in ("question", "A", "B", "C", "D", "answer")):
        item = {
            "question_id": str(row.get("question_id") or f"row_{row_number:06d}"),
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip().upper(),
        }
        for letter in LETTERS:
            item[letter] = str(row[letter]).strip()
    elif "instruction" in row and "output" in row:
        item = _parse_instruction_mcq(str(row["instruction"]))
        item["question_id"] = str(
            row.get("question_id")
            or row.get("_source_question_id")
            or f"row_{row_number:06d}"
        )
        item["answer"] = _extract_gold_from_output(str(row["output"]))
    else:
        raise McqRecordError("row is neither flat MCQ nor instruction/output MCQ")

    if item["answer"] not in LETTERS:
        raise McqRecordError(f"invalid gold answer letter: {item['answer']!r}")
    for key in ("question", "A", "B", "C", "D"):
        if not item[key]:
            raise McqRecordError(f"empty required MCQ field: {key}")

    for key in ("category", "topic", "difficulty"):
        if key in row:
            item[key] = row[key]
    return item


def normalize_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        try:
            records.append(normalize_mcq_record(row, row_number))
        except McqRecordError as exc:
            skipped.append({"row_number": row_number, "reason": str(exc)})
    return records, skipped


def _softmax_probs(logprobs: dict[str, float]) -> dict[str, float]:
    vals = [logprobs[letter] for letter in LETTERS]
    max_v = max(vals)
    exps = [math.exp(v - max_v) for v in vals]
    total = sum(exps)
    return {letter: exps[i] / total for i, letter in enumerate(LETTERS)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score mean MCQ gold-token log-likelihood without generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to merged model directory")
    parser.add_argument(
        "--dataset",
        default="data/val_v3.jsonl",
        help="JSONL dataset. Non-MCQ rows are skipped with a count in the output.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/mcq_gold_ll.json")
    parser.add_argument(
        "--diagnostics-dir",
        default=str(DEFAULT_DIAGNOSTICS_DIR),
        help="Directory containing benchmark-eval-workspace/diagnostics/utils.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N normalized MCQ rows; useful for smoke checks.",
    )
    parser.add_argument(
        "--full-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use diagnostics full-precision model loading instead of 4-bit NF4.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    helpers = import_diagnostics_helpers(Path(args.diagnostics_dir))
    rows = helpers["load_benchmark"](args.dataset)
    records, skipped = normalize_records(rows)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error(f"no scorable MCQ rows found in {args.dataset}")

    model, tokenizer = helpers["load_model_and_tokenizer"](
        args.model,
        full_precision=args.full_precision,
        device=args.device,
    )
    letter_ids = helpers["get_letter_token_ids"](tokenizer)

    per_item: list[dict[str, Any]] = []
    log.info("Scoring %d MCQ rows from %s", len(records), args.dataset)
    for item in records:
        prompt = helpers["format_mcq_prompt"](item, tokenizer, after_think=True)
        logprobs = helpers["score_symbol"](model, tokenizer, prompt, letter_ids, args.device)
        probs = _softmax_probs(logprobs)
        top_letter = max(LETTERS, key=lambda letter: logprobs[letter])
        gold = item["answer"]
        per_item.append(
            {
                "question_id": item["question_id"],
                "category": item.get("category", "unknown"),
                "topic": item.get("topic", "unknown"),
                "gold": gold,
                "gold_logprob": logprobs[gold],
                "gold_probability": probs[gold],
                "top_letter": top_letter,
                "top_logprob": logprobs[top_letter],
                "correct_by_argmax": top_letter == gold,
                "logprobs": {letter: logprobs[letter] for letter in LETTERS},
            }
        )

    total = len(per_item)
    mean_gold_ll = sum(row["gold_logprob"] for row in per_item) / total
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "diagnostics_dir": str(Path(args.diagnostics_dir)),
        "scored_mcq_rows": total,
        "skipped_non_mcq_rows": len(skipped),
        "mean_gold_logprob": mean_gold_ll,
        "mean_gold_nll": -mean_gold_ll,
        "argmax_accuracy": sum(1 for row in per_item if row["correct_by_argmax"]) / total,
        "full_precision": args.full_precision,
        "device": args.device,
    }
    result = {
        "summary": summary,
        "per_item": per_item,
        "skipped": skipped,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {_as_repo_relative(output_path)}")
    print(f"scored_mcq_rows={total}")
    print(f"skipped_non_mcq_rows={len(skipped)}")
    print(f"mean_gold_logprob={mean_gold_ll:.8f}")
    print(f"mean_gold_nll={-mean_gold_ll:.8f}")
    print(f"argmax_accuracy={summary['argmax_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
