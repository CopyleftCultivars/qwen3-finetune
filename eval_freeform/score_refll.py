#!/usr/bin/env python3
"""Teacher-forced reference-answer log-likelihood scorer for free-form eval.

This script is deterministic and grader-free. Draft-file smoke outputs are
non-reportable until #41 produces the human-reviewed canonical set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """\
Question:
{question}

Reference answer:
"""


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


def validate_item(row: dict[str, Any]) -> None:
    for key in ("id", "question", "reference_answer"):
        if key not in row or not row[key]:
            raise ValueError(f"line {row.get('_line_number')}: missing required key {key!r}")


def stable_tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def mock_score(row: dict[str, Any], model: str) -> dict[str, Any]:
    tokens = stable_tokenize(row["reference_answer"])
    if not tokens:
        raise ValueError(f"item {row['id']}: empty reference answer")
    token_logprobs: list[float] = []
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(f"{model}\0{row['id']}\0{index}\0{token}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
        token_logprobs.append(-0.25 - 5.0 * unit)
    total = sum(token_logprobs)
    return {
        "token_count": len(tokens),
        "total_logprob": total,
        "mean_logprob": total / len(tokens),
        "mean_nll": -total / len(tokens),
    }


def real_model_score(rows: list[dict[str, Any]], model_path: str, device: str) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Real scoring requires torch and transformers; use --mock-model for smoke tests") from exc

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            prompt = PROMPT_TEMPLATE.format(question=row["question"])
            reference = row["reference_answer"]
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            full_ids = tokenizer(prompt + reference, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            if full_ids.shape[1] <= prompt_ids.shape[1]:
                raise ValueError(f"item {row['id']}: reference answer produced no target tokens")

            labels = full_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100
            outputs = model(input_ids=full_ids, labels=labels)
            target_count = int((labels != -100).sum().item())
            total_nll = float(outputs.loss.item()) * target_count
            results.append(
                {
                    "id": row["id"],
                    "type": row.get("type"),
                    "source_doc": row.get("source_doc"),
                    "token_count": target_count,
                    "total_logprob": -total_nll,
                    "mean_logprob": -total_nll / target_count,
                    "mean_nll": total_nll / target_count,
                    "prompt_template": "question_reference_answer_prefix_v1",
                }
            )
    return results


def aggregate(per_item: list[dict[str, Any]]) -> dict[str, float | int]:
    total_tokens = sum(int(row["token_count"]) for row in per_item)
    total_logprob = sum(float(row["total_logprob"]) for row in per_item)
    return {
        "scored_items": len(per_item),
        "total_reference_tokens": total_tokens,
        "total_logprob": total_logprob,
        "mean_reference_logprob_per_token": total_logprob / total_tokens,
        "mean_reference_nll_per_token": -total_logprob / total_tokens,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score teacher-forced LL of reference_answer fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model path/name, or mock model label with --mock-model")
    parser.add_argument("--input", required=True, help="Free-form item JSONL")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock-model", action="store_true", help="No torch/transformers; deterministic smoke scorer")
    parser.add_argument("--draft-only", action="store_true", help="Mark output as draft-only/non-reportable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(0)

    rows = load_jsonl(Path(args.input), args.limit)
    if not rows:
        raise SystemExit(f"no rows loaded from {args.input}")
    for row in rows:
        validate_item(row)

    if args.mock_model:
        per_item = []
        for row in rows:
            score = mock_score(row, args.model)
            per_item.append(
                {
                    "id": row["id"],
                    "type": row.get("type"),
                    "source_doc": row.get("source_doc"),
                    "prompt_template": "mock_question_reference_answer_prefix_v1",
                    **score,
                }
            )
    else:
        per_item = real_model_score(rows, args.model, args.device)

    result = {
        "summary": {
            "model": args.model,
            "model_mode": "mock" if args.mock_model else "transformers",
            "input": args.input,
            "device": args.device,
            "draft_only_non_reportable": args.draft_only,
            "deterministic": True,
            **aggregate(per_item),
        },
        "per_item": per_item,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.output}")
    print(f"scored_items={result['summary']['scored_items']}")
    print(f"total_reference_tokens={result['summary']['total_reference_tokens']}")
    print(f"mean_reference_logprob_per_token={result['summary']['mean_reference_logprob_per_token']:.8f}")
    print(f"mock_model={args.mock_model}")
    print(f"draft_only_non_reportable={args.draft_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
