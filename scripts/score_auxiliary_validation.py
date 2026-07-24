#!/usr/bin/env python3
"""Score one auxiliary-task validation stream with deterministic generation.

This scorer is intentionally separate from the frozen MCQ objective. It reports
task-native quality for one open validation file without touching sealed eval
files: exact/JSON-valid rates where applicable and token F1 for all tasks.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = ("summarization", "ner", "structured_extraction", "open_book_qa")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
CHAT_PROMPT = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def reject_sealed_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "eval_freeform" in parts and path.name.casefold() == "test.jsonl":
        raise ValueError("sealed free-form test paths are forbidden")
    if "sealed" in path.name.casefold():
        raise ValueError("sealed paths are forbidden")


def normalized_tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_RE.findall(value)


def token_f1(reference: str, prediction: str) -> float:
    expected = Counter(normalized_tokens(reference))
    actual = Counter(normalized_tokens(prediction))
    overlap = sum((expected & actual).values())
    if not expected or not actual:
        return 1.0 if not expected and not actual else 0.0
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def normalized_text(text: str) -> str:
    return " ".join(normalized_tokens(text))


def extract_json(text: str) -> Any | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    decoder = json.JSONDecoder()
    for marker in ("{", "["):
        start = candidate.find(marker)
        if start < 0:
            continue
        try:
            value, _ = decoder.raw_decode(candidate[start:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_metrics(task: str, reference: str, prediction: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "token_f1": token_f1(reference, prediction),
        "exact_text": normalized_text(reference) == normalized_text(prediction),
    }
    if task in ("ner", "structured_extraction"):
        expected = extract_json(reference)
        actual = extract_json(prediction)
        metrics["json_valid"] = actual is not None
        metrics["exact_json"] = actual is not None and actual == expected
        if actual is not None and expected is not None:
            metrics["json_token_f1"] = token_f1(canonical_json(expected), canonical_json(actual))
        else:
            metrics["json_token_f1"] = 0.0
    return metrics


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
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    reject_sealed_path(args.dataset)
    if not args.model.exists():
        parser.error(f"model path not found: {args.model}")
    if not args.dataset.exists():
        parser.error(f"dataset path not found: {args.dataset}")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    rows = load(args.dataset.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error("dataset is empty")
    for row in rows:
        if row.get("auxiliary_task") != args.task:
            parser.error("dataset contains a row for a different auxiliary task")

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
        prediction = generate(model, tokenizer, str(row["instruction"]), args.max_new_tokens)
        metrics = task_metrics(args.task, str(row["output"]), prediction)
        scored.append({"id": row.get("id"), "source_id": row.get("source_id"), "prediction": prediction, **metrics})

    aggregate: dict[str, float] = {}
    metric_names = sorted({key for row in scored for key, value in row.items() if isinstance(value, bool | float | int)})
    for name in metric_names:
        aggregate[name] = sum(float(row[name]) for row in scored if name in row) / len(scored)
    report = {
        "format": "auxiliary_validation_score_v1",
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
