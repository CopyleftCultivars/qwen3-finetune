#!/usr/bin/env python3
"""Score a fixed local sample of the held-out general-instruction source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def token_f1(reference: str, prediction: str) -> float:
    expected = Counter(tokens(reference))
    actual = Counter(tokens(prediction))
    overlap = sum((expected & actual).values())
    if not expected or not actual:
        return 1.0 if not expected and not actual else 0.0
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalized(text: str) -> str:
    return " ".join(tokens(text))


def render_prompt(instruction: str) -> str:
    return f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n<|im_start|>assistant\n"


def choose_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(str(row["_source_id"]).encode()).hexdigest())
    return ordered[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    if not args.model.exists() or not args.dataset.exists():
        parser.error("model and dataset paths must exist")
    rows = choose_rows(load(args.dataset.resolve()), args.limit)
    if not rows:
        parser.error("dataset is empty")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    scored = []
    for row in rows:
        prompt = render_prompt(str(row["instruction"]))
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        reference = str(row["output"])
        scored.append({
            "source_id": row["_source_id"],
            "token_f1": token_f1(reference, prediction),
            "exact_text": normalized(reference) == normalized(prediction),
        })
    report = {
        "format": "general_instruction_validation_score_v1",
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "count": len(scored),
        "sample_rule": "sha256_source_id_v1",
        "aggregate": {
            "token_f1": sum(row["token_f1"] for row in scored) / len(scored),
            "exact_text": sum(float(row["exact_text"]) for row in scored) / len(scored),
        },
        "items": scored,
        "sealed_test_paths_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(scored), "aggregate": report["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
