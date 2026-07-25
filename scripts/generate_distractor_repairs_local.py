#!/usr/bin/env python3
"""Locally propose distractor repairs for unused, judge-reviewed MCQs.

This authoring step never sends QMD text to a remote service. It selects
review-status MCQs that are not already in the v5clean QMD pool, optionally
adds already judge-kept unused QMD items to close a balance bracket, asks the
local v5clean checkpoint to rewrite only the incorrect options, and emits
candidate flat records for the separate quality-judge/audit stage.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LETTERS = ("A", "B", "C", "D")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def prompt_for(row: dict[str, Any]) -> str:
    options = "\n".join(f"{letter}. {str(row[letter]).strip()}" for letter in LETTERS)
    return (
        "You are editing one natural-farming multiple-choice item whose distractors "
        "were judged weak. Preserve the question, the correct answer letter, and the "
        "meaning of the correct option. Rewrite only the three incorrect options. "
        "Make each distractor a plausible near-miss: same domain, related to the "
        "question, factually wrong for a specific reason, and similar in length and "
        "style to the correct option. Avoid absurd answers, generic filler, absolute "
        "words such as 'always' or 'never', overlapping options, and clues based on "
        "length or grammar. Use only the supplied passage and do not invent named "
        "facts that the passage cannot support. Return JSON only with keys A, B, C, D.\n\n"
        f"Source passage:\n{str(row.get('source_passage', '')).strip()[:2600]}\n\n"
        f"Question:\n{str(row['question']).strip()}\n\n"
        f"Current options:\n{options}\n\n"
        f"Correct answer letter: {str(row['answer']).strip().upper()}"
    )


def parse_options(text: str) -> dict[str, str] | None:
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    options: dict[str, str] = {}
    for letter in LETTERS:
        candidate = value.get(letter)
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        options[letter] = candidate.strip()
    if len({normalize(options[letter]) for letter in LETTERS}) != 4:
        return None
    return options


def select_rows(
    judged_paths: list[Path],
    base_paths: list[Path],
    per_answer: int,
    supplemental_paths: list[Path],
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    def question_from_row(row: dict[str, Any]) -> str:
        if row.get("question"):
            return str(row["question"])
        instruction = str(row.get("instruction", ""))
        return instruction.split("\n\nA.", 1)[0]

    used_questions = {
        normalize(question_from_row(row))
        for path in base_paths
        for row in load_jsonl(path)
        if question_from_row(row)
    }
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in judged_paths:
        for row in load_jsonl(path):
            scores = row.get("quality_scores", {})
            answer = str(row.get("answer", "")).strip().upper()
            question_key = normalize(row.get("question", ""))
            if scores.get("verdict") != "review":
                continue
            if answer not in LETTERS or not question_key or question_key in used_questions:
                continue
            question_id = str(row.get("question_id", "")).strip()
            if not question_id or question_id in seen:
                continue
            if not str(row.get("source_passage", "")).strip():
                continue
            seen.add(question_id)
            candidates.append(row)

    # The new QMD pool is already judge-kept but was never included in the
    # v5clean training source. It supplies only the answer letters that the
    # unused review bracket cannot provide; its repaired versions still go
    # through the fresh quality judge and contamination audit.
    for path in supplemental_paths:
        for row in load_jsonl(path):
            answer = str(row.get("answer", "")).strip().upper()
            question_key = normalize(row.get("question", ""))
            question_id = str(row.get("question_id", "")).strip()
            if answer not in LETTERS or not question_key or question_key in used_questions:
                continue
            if not question_id or question_id in seen or not str(row.get("source_passage", "")).strip():
                continue
            seen.add(question_id)
            candidates.append(row)

    candidates.sort(
        key=lambda row: (
            float(row.get("quality_scores", {}).get("distractor_quality", 9)),
            float(row.get("quality_scores", {}).get("composite", 9)),
            str(row.get("question_id", "")),
        )
    )
    by_answer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_answer[str(row["answer"]).strip().upper()].append(row)
    missing = {letter: len(by_answer[letter]) for letter in LETTERS if len(by_answer[letter]) < per_answer}
    if missing and not allow_missing:
        raise ValueError(f"not enough unused review items for exact balance: {missing}")
    selected = [row for letter in LETTERS for row in by_answer[letter][:per_answer]]
    selected.sort(key=lambda row: hashlib.sha256(str(row["question_id"]).encode()).hexdigest())
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model path or cached Hugging Face model ID")
    parser.add_argument("--judged-glob", default="mcq_generation/quality/scores/*_judged.jsonl")
    parser.add_argument("--base", action="append", required=True, type=Path)
    parser.add_argument("--supplemental", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-answer", type=int, default=20)
    parser.add_argument(
        "--allow-source-imbalance",
        action="store_true",
        help="Use all available candidates when an exact flat-source balance is unavailable; cyclic training shifts still balance target letters.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()
    model_path = Path(args.model)
    if "/" not in args.model and not model_path.exists():
        parser.error(f"model not found: {args.model}")
    if args.device.startswith("cuda") and not __import__("torch").cuda.is_available():
        parser.error("CUDA is required for local authoring")

    judged_paths = [Path(path) for path in sorted(glob.glob(args.judged_glob))]
    if not judged_paths:
        parser.error(f"no judged files matched: {args.judged_glob}")
    try:
        selected = select_rows(
            judged_paths,
            [path.resolve() for path in args.base],
            args.per_answer,
            [path.resolve() for path in args.supplemental],
        )
    except ValueError:
        if not args.allow_source_imbalance:
            raise
        # Re-run selection at one item per answer to obtain the complete
        # available bracket without fabricating answer-position balance. The
        # downstream cyclic-shift builder creates the exact balanced target
        # surface used by v5clean.
        selected = select_rows(
            judged_paths,
            [path.resolve() for path in args.base],
            args.per_answer,
            [path.resolve() for path in args.supplemental],
            allow_missing=True,
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device)
    model.eval()

    output_rows: list[dict[str, Any]] = []
    failed = 0
    for index, row in enumerate(selected, 1):
        messages = [{"role": "user", "content": prompt_for(row)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        repaired = parse_options(text)
        if repaired is None:
            failed += 1
            preview = " ".join(text.split())[:260]
            print(f"  [{index}/{len(selected)}] {row['question_id']}: parse failed: {preview!r}")
            continue
        candidate = {
            "question_id": f"DQ-{str(row['question_id']).replace(' ', '_')}",
            "question": str(row["question"]).strip(),
            **repaired,
            "answer": str(row["answer"]).strip().upper(),
            "category": row.get("category"),
            "topic": row.get("topic"),
            "source_passage": row.get("source_passage"),
            "source_question_id": row.get("question_id"),
            "authoring_method": "local_v5clean_distractor_repair_v1",
            "authoring_model": args.model,
            "prior_quality_scores": row.get("quality_scores"),
            "prior_quality_justifications": row.get("quality_justifications"),
            "judge_verdict": "pending_quality_judge",
            "_contamination_status": "pending_audit",
        }
        output_rows.append(candidate)
        print(f"  [{index}/{len(selected)}] {row['question_id']}: repaired")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"selected": len(selected), "emitted": len(output_rows), "parse_failed": failed, "answer_counts": dict(Counter(row["answer"] for row in output_rows))}, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
