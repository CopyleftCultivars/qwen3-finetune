#!/usr/bin/env python3
"""Generate P13 rationale targets locally with the cached Qwen model.

This keeps the source questions on the local machine.  It intentionally asks
for a short step-back explanation: state the relevant general agronomic
principle, then apply it to the question and distinguish the selected answer
from the distractors.  The wrapper/novelty checks happen in
``build_p13_novel_rationale.py`` after generation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


OPTION_RE = re.compile(
    r"^(.+?)\n\nA\. (.+)\nB\. (.+)\nC\. (.+)\nD\. (.+)$", re.DOTALL
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_row(row: dict) -> tuple[str, dict[str, str], str]:
    match = OPTION_RE.match(str(row["instruction"]).strip())
    if not match:
        raise ValueError(f"not an MCQ instruction: {row.get('_source_question_id')}")
    original_output = str(row["output"])
    letter_match = re.search(r"</think>\s*([A-D])\s*$", original_output)
    if not letter_match:
        raise ValueError(f"missing source answer letter: {row.get('_source_question_id')}")
    return match.group(1).strip(), {
        "A": match.group(2).strip(),
        "B": match.group(3).strip(),
        "C": match.group(4).strip(),
        "D": match.group(5).strip(),
    }, letter_match.group(1)


def clean_generation(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:Rationale|Step[- ]back rationale)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"loading model={args.model} device={device} dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    rows = load_rows(args.input)
    output_rows: list[dict] = []
    for index, row in enumerate(rows):
        question, options, letter = parse_row(row)
        prompt = (
            "Write a novel step-back rationale for this agricultural multiple-choice "
            "training target. First name the general agronomic principle or mechanism "
            "that matters, then apply it to this question and briefly distinguish the "
            "correct option from the distractors. Use 2-4 factual sentences. Do not "
            "copy a stock answer, do not begin with 'The correct answer is', do not "
            "mention this instruction, and return only the rationale text. Do not use "
            "<think> tags or a separate answer-letter line.\n\n"
            f"Question: {question}\n\n"
            f"A. {options['A']}\nB. {options['B']}\nC. {options['C']}\nD. {options['D']}\n\n"
            f"Correct option: {letter}. {options[letter]}"
        )
        messages = [
            {"role": "system", "content": "You are a careful agricultural education author."},
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            )
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, inputs.shape[-1]:]
        rationale = clean_generation(tokenizer.decode(continuation, skip_special_tokens=True))
        if len(rationale.split()) < 12:
            raise ValueError(f"short generated rationale at index {index}: {rationale!r}")
        output_rows.append({
            "instruction": row["instruction"],
            "output": f"<think>\n{rationale}\n</think>\n{letter}",
            "_src_idx": index,
        })
        print(f"generated {index + 1}/{len(rows)} id={row.get('_source_question_id')}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(output_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
