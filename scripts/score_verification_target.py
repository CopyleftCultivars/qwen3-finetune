#!/usr/bin/env python3
"""Score proposed-answer verification validation rows with deterministic parsing."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LETTERS = set("ABCD")
VERIFICATION_RE = re.compile(
    r"proposed_answer\s*:\s*([ABCD]).*?"
    r"verdict\s*:\s*(ACCEPT|REJECT).*?"
    r"verified_answer\s*:\s*([ABCD])",
    re.IGNORECASE | re.DOTALL,
)
FINAL_LETTER_RE = re.compile(r"(?:^|\n)\s*([ABCD])\s*$", re.IGNORECASE)


def reject_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    normalized = path.resolve().as_posix().casefold()
    if "spanish-benchmark-workspace" in parts:
        raise ValueError("Spanish benchmark paths are evaluation-only")
    if "sealed" in parts or "sealed" in path.name.casefold() or "/eval_freeform/test.jsonl" in normalized:
        raise ValueError("sealed evaluation path is forbidden")


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_prediction(text: str) -> dict[str, Any]:
    match = VERIFICATION_RE.search(text)
    if not match:
        return {"parse_valid": False, "proposed_answer": None, "verdict": None, "verified_answer": None, "final_answer": None}
    final_match = FINAL_LETTER_RE.search(text.strip())
    return {
        "parse_valid": True,
        "proposed_answer": match.group(1).upper(),
        "verdict": match.group(2).upper(),
        "verified_answer": match.group(3).upper(),
        "final_answer": final_match.group(1).upper() if final_match else None,
    }


def score_rows(rows: list[dict[str, Any]], model: Any, tokenizer: Any, device: str, max_new_tokens: int) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = []
    for row in rows:
        messages = [{"role": "user", "content": str(row["instruction"])}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        parsed = parse_prediction(generated)
        expected_verdict = str(row["expected_verdict"]).upper()
        expected_answer = str(row["correct_answer"]).upper()
        results.append({
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "expected_verdict": expected_verdict,
            "expected_answer": expected_answer,
            "prediction": generated,
            **parsed,
            "verdict_correct": parsed["verdict"] == expected_verdict,
            "verified_answer_correct": parsed["verified_answer"] == expected_answer,
            "final_answer_correct": parsed["final_answer"] == expected_answer,
            "fully_correct": (
                parsed["parse_valid"]
                and parsed["verdict"] == expected_verdict
                and parsed["verified_answer"] == expected_answer
                and parsed["final_answer"] == expected_answer
            ),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    reject_path(args.dataset)
    if not args.model.exists():
        parser.error(f"model path not found: {args.model}")
    rows = load(args.dataset.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        parser.error("dataset is empty")
    for row in rows:
        if row.get("verification_task") != "proposed_answer_audit":
            parser.error("dataset contains a different verification task")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    scored = score_rows(rows, model, tokenizer, args.device, args.max_new_tokens)
    metric_names = ("parse_valid", "verdict_correct", "verified_answer_correct", "final_answer_correct", "fully_correct")
    aggregate = {name: sum(float(row[name]) for row in scored) / len(scored) for name in metric_names}
    report = {
        "format": "verification_target_validation_score_v1",
        "task": "proposed_answer_audit",
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "count": len(scored),
        "aggregate": aggregate,
        "proposed_answer_distribution": dict(sorted(Counter(row["expected_answer"] for row in rows).items())),
        "items": scored,
        "sealed_test_paths_read": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(scored), "aggregate": aggregate}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
