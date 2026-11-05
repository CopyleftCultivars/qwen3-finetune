#!/usr/bin/env python3
"""#239 G3: unified HF ladder scorer for Qwen3 and Qwen3.5 (transformers 5.5).

Same frozen protocol as scripts/ladder_score_local.py -- SYSTEM_PROMPT,
PROMPT_TEMPLATE and extract_answer are imported from
benchmark-workspace/evaluate.py, greedy temperature=0, seed 42,
max_new_tokens 8 -- but the model load path works for both Qwen3 and
Qwen3.5 under transformers>=5.2: each model's NATIVE apply_chat_template
with thinking OFF (enable_thinking=False).

The --load-4bit NF4 path is the labelled-non-equivalent deployable proxy
for the BLOCKED Cactus CQ leg (G1/G2): it measures whether the adapted
weights survive 4-bit quantization, not on-device quality.

Run with the isolated transformers-5.5 venv for every G3 row so the
environment is constant across the comparison.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BENCH_WS = Path(os.environ.get("BENCHMARK_ROOT", "."))
DEFAULT_BENCH = (BENCH_WS / "benchmarks/natural_fertilizers/"
                 "natural_fertilizers_benchmark_v1_graded.jsonl")


def import_protocol():
    sys.path.insert(0, str(BENCH_WS))
    import evaluate as ev  # noqa: E402

    return ev.SYSTEM_PROMPT, ev.PROMPT_TEMPLATE, ev.extract_answer


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_host(model_path: str, load_4bit: bool, seed: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant,
            device_map={"": 0}, dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={"": 0}, dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.eval()
    torch.manual_seed(seed)
    return model, tokenizer


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-4bit", action="store_true",
                        help="NF4 proxy for the blocked INT4 deployable leg")
    parser.add_argument("--out-prefix", default="results/g3_ladder210")
    args = parser.parse_args(argv)

    system_prompt, prompt_template, extract_answer = import_protocol()
    import torch

    rows = [json.loads(line) for line in open(args.benchmark, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} ladder items from {Path(args.benchmark).name}", flush=True)
    model, tokenizer = load_host(args.model, args.load_4bit, args.seed)
    print(f"loaded {args.model} "
          f"dtype={next(model.parameters()).dtype} 4bit={args.load_4bit}", flush=True)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    per_item: list[dict[str, Any]] = []
    out_path = Path(f"{args.out_prefix}_{args.label}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_truncated = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for i, row in enumerate(rows, start=1):
            user = prompt_template.format(
                question=row["question"], A=row["A"], B=row["B"],
                C=row["C"], D=row["D"])
            prompt = render(tokenizer, [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user}])
            torch.manual_seed(args.seed)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            started = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=pad)
            elapsed = (time.perf_counter() - started) * 1000.0
            new = out[0][inputs["input_ids"].shape[1]:]
            if int(new.shape[0]) >= args.max_new_tokens:
                n_truncated += 1
            raw = tokenizer.decode(new, skip_special_tokens=True)
            predicted = extract_answer(raw)
            gold = str(row["answer"]).strip().upper()
            rec = {
                "question_id": row["question_id"],
                "correct_answer": gold,
                "model_answer": predicted,
                "raw_response": raw[:400],
                "ambiguous": predicted is None,
                "correct": predicted is not None and predicted == gold,
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "latency_ms": round(elapsed, 1),
            }
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            per_item.append(rec)
            if i % 25 == 0:
                print(f"  [{args.label}] {i}/{len(rows)}", flush=True)

    n = len(per_item)
    correct = sum(r["correct"] for r in per_item)
    amb = sum(r["ambiguous"] for r in per_item)
    lo, hi = wilson(correct, n)
    summary = {
        "label": args.label,
        "model": args.model,
        "benchmark": args.benchmark,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "ambiguous": amb,
        "ambiguous_rate": round(amb / n, 4) if n else 0.0,
        "truncated_at_budget": n_truncated,
        "load_4bit_nf4_proxy": args.load_4bit,
        "enable_thinking": False,
        "temperature": 0.0,
        "seed": args.seed,
        "mean_latency_ms": round(statistics.mean(
            r["latency_ms"] for r in per_item), 1) if n else 0.0,
        "by_difficulty": {},
    }
    for diff in sorted({str(r["difficulty"]) for r in per_item}):
        sub = [r for r in per_item if str(r["difficulty"]) == diff]
        summary["by_difficulty"][diff] = {
            "n": len(sub),
            "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4),
        }
    Path(f"{args.out_prefix}_{args.label}_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
