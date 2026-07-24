#!/usr/bin/env python3
"""Score preserved multi-turn and deterministic reflection validation rows."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = ("multiturn", "flattened_control", "interrupted_counterfactual", "answer_audit")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


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


def reject_sealed(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "eval_freeform" in parts and path.name.casefold() == "test.jsonl":
        raise ValueError("sealed free-form test paths are forbidden")
    if "sealed" in path.name.casefold():
        raise ValueError("sealed path is forbidden")


def tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_RE.findall(value)


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


def render_hardcoded(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    rendered = "\n".join(
        f"<|im_start|>{turn['role']}\n{turn['content'].strip()}<|im_end|>"
        for turn in messages
    )
    if add_generation_prompt:
        rendered += "\n<|im_start|>assistant\n"
    return rendered


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    import torch

    prompt = render_hardcoded(messages, add_generation_prompt=True)
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


def corrected_answer(text: str) -> str:
    marker = "corrected answer:"
    lowered = text.casefold()
    index = lowered.find(marker)
    return text[index + len(marker):].strip() if index >= 0 else text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    reject_sealed(args.dataset)
    if not args.model.exists() or not args.dataset.exists():
        parser.error("model and dataset paths must exist")
    rows = load(args.dataset.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error("dataset is empty")
    for row in rows:
        if row.get("structure_task") != args.task:
            parser.error("dataset contains a different structure task")
        if not isinstance(row.get("prompt_messages"), list):
            parser.error("dataset row is missing prompt_messages")

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
        prediction = generate(model, tokenizer, row["prompt_messages"], args.max_new_tokens)
        reference = str(row["target_output"])
        item = {
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "prediction": prediction,
            "token_f1": token_f1(reference, prediction),
            "exact_text": normalized(reference) == normalized(prediction),
        }
        if args.task in {"interrupted_counterfactual", "answer_audit"}:
            expected_corrected = corrected_answer(reference)
            item["corrected_answer_token_f1"] = token_f1(expected_corrected, corrected_answer(prediction))
        scored.append(item)
    aggregate: dict[str, float] = {}
    metric_names = sorted({key for row in scored for key, value in row.items() if isinstance(value, bool | float | int)})
    for name in metric_names:
        aggregate[name] = sum(float(row[name]) for row in scored if name in row) / len(scored)
    report = {
        "format": "multiturn_reflection_validation_score_v1",
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "task": args.task,
        "count": len(scored),
        "aggregate": aggregate,
        "items": scored,
        "sealed_test_paths_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"task": args.task, "count": len(scored), "aggregate": aggregate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
