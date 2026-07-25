#!/usr/bin/env python3
"""Score answer-plus-evidence predictions on an open grounded dev stream."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
EVIDENCE_RE = re.compile(r"^\s*evidence\s*:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
CHAT_PROMPT = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def reject_sealed_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "eval_freeform" in parts and path.name.casefold() == "test.jsonl":
        raise ValueError("sealed free-form test paths are forbidden")
    if "sealed" in path.name.casefold():
        raise ValueError("sealed paths are forbidden")


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def normalized_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def token_f1(reference: str, prediction: str) -> float:
    expected = Counter(normalized_tokens(reference))
    actual = Counter(normalized_tokens(prediction))
    overlap = sum((expected & actual).values())
    if not expected or not actual:
        return 1.0 if not expected and not actual else 0.0
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def parse_prediction(text: str) -> dict[str, Any]:
    answer_match = ANSWER_RE.search(text)
    evidence_match = EVIDENCE_RE.search(text)
    return {
        "answer": answer_match.group(1).strip() if answer_match else "",
        "evidence": evidence_match.group(1).strip() if evidence_match else "",
        "parse_valid": bool(answer_match and evidence_match),
    }


def generate(model: Any, tokenizer: Any, instruction: str, max_new_tokens: int) -> str:
    import torch

    prompt = CHAT_PROMPT.format(instruction=instruction.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    reject_sealed_path(args.dataset)
    if not args.model.exists():
        parser.error(f"model path not found: {args.model}")
    if not args.dataset.exists():
        parser.error(f"dataset path not found: {args.dataset}")
    rows = load(args.dataset.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error("dataset is empty")
    for row in rows:
        if row.get("grounded_task") != "open_book_grounded":
            parser.error("dataset contains a row for a different task")

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

    scored: list[dict[str, Any]] = []
    for row in rows:
        prediction_text = generate(model, tokenizer, str(row["instruction"]), args.max_new_tokens)
        parsed = parse_prediction(prediction_text)
        expected = parse_prediction(str(row["output"]))
        answer_f1 = token_f1(expected["answer"], parsed["answer"])
        answer_exact = (
            " ".join(normalized_tokens(expected["answer"]))
            == " ".join(normalized_tokens(parsed["answer"]))
        )
        evidence_exact = parsed["evidence"] == expected["evidence"]
        scored.append({
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "prediction": prediction_text,
            "parse_valid": parsed["parse_valid"],
            "answer_f1": answer_f1,
            "answer_exact": answer_exact,
            "evidence_exact": evidence_exact,
            "grounded_exact": bool(parsed["parse_valid"] and answer_exact and evidence_exact),
        })

    aggregate: dict[str, float] = {}
    metric_names = ("parse_valid", "answer_f1", "answer_exact", "evidence_exact", "grounded_exact")
    for name in metric_names:
        aggregate[name] = sum(float(row[name]) for row in scored) / len(scored)
    report = {
        "format": "open_book_grounded_score_v1",
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "count": len(scored),
        "aggregate": aggregate,
        "items": scored,
        "sealed_test_paths_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(scored), "aggregate": aggregate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
