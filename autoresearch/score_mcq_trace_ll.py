#!/usr/bin/env python3
"""score_mcq_trace_ll.py - Trace-conditioned (2-pass) MCQ scoring.

Chainlink issue #106 (I1), Epic #105, Milestone #1 (M1: Plan A eval). See
TRACE_EVAL_REASONING_SFT_RLM_PLAN.md, Part 1 / Plan A, for the full spec this
implements.

Why a new file: the frozen `score_mcq_ll.py` scores
P(gold letter | forced-EMPTY-<think> prompt) -- that objective removes
reasoning from the scored sequence entirely, so J-space has no verbalizable
content to read. This script restores a real reasoning surface without
touching any frozen_manifest.json file; it imports the frozen scorer's row
parsing (`normalize_records`) rather than duplicating it.

Two-pass shape:
  Pass 1 (reason):  question+options, think-ON, greedy decode (do_sample=False,
                     fixed seed), up to --max-trace-tokens new tokens.
                     Output = the model's own generated reasoning trace.
  Pass 2 (extract): question+options+Pass-1 trace baked into a *closed*
                     <think>...</think> block (think-OFF equivalent), scored
                     with the same score_symbol/get_letter_token_ids the
                     frozen scorer uses for its forced-empty-think score.
                     Output = option-set-normalized gold-letter log-likelihood.

This is a NEW, NOT-comparable objective (see TRACE_EVAL_REASONING_SFT_RLM_PLAN.md
assumption 5) -- do not diff mean_trace_option_set_gold_logprob against
score_mcq_ll.py's mean_option_set_gold_logprob. objective_weights_v3.json
(chainlink #109 / I4) is where that policy gets written down formally.

Explicitly OUT of scope for this file (separate, later chainlink issues):
  - Sealed-path denylist refusal + determinism verification (#107 / I2).
  - Empty-trace delta_trace ablation (#108 / I3) -- score_with_trace() below
    takes an explicit `trace` string precisely so I3 can call it a second
    time with trace="" without duplicating the scoring logic.

Usage:
  python autoresearch/score_mcq_trace_ll.py \
    --model outputs/merged-model-p6-lr5e5-r2 \
    --dataset data/val_v3.jsonl \
    --output results/mcq_trace_ll_p6-lr5e5-r2.json \
    --per-item-output results/mcq_trace_ll_p6-lr5e5-r2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_mcq_ll import (  # noqa: E402 -- reuse the frozen scorer's row parsing, do not duplicate it
    DEFAULT_DIAGNOSTICS_DIR,
    LETTERS,
    _as_repo_relative,
    _softmax_probs,
    import_diagnostics_helpers,
    normalize_records,
)

DEFAULT_MAX_TRACE_TOKENS = 256
DEFAULT_SEED = 42


def build_question_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    question_text = (
        f"{item['question']}\n\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}"
    )
    return [{"role": "user", "content": question_text}]


def extract_trace(generated: str) -> tuple[str, bool]:
    """Pull the reasoning content out of a think-on generation.

    Returns (trace_text, truncated) -- truncated=True means </think> was never
    reached within --max-trace-tokens, so `trace_text` is whatever partial
    reasoning was generated, not a complete thought.
    """
    if "</think>" in generated:
        trace = generated.split("</think>", maxsplit=1)[0]
        trace = trace.split("<think>", maxsplit=1)[-1]
        return trace.strip(), False
    trace = generated.split("<think>", maxsplit=1)[-1] if "<think>" in generated else generated
    return trace.strip(), True


def generate_trace(
    model,
    tokenizer,
    item: dict[str, Any],
    max_trace_tokens: int,
    device: str,
    seed: int,
) -> tuple[str, str, bool, int]:
    """Pass 1: think-ON greedy generation.

    Returns (raw_generated_text, extracted_trace, truncated, n_new_tokens).
    """
    import torch  # deferred: keep --help dependency-light, matches score_mcq_ll.py's style

    torch.manual_seed(seed)
    messages = build_question_messages(item)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_trace_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    trace, truncated = extract_trace(generated)
    return generated, trace, truncated, new_ids.shape[0]


def build_trace_prompt(item: dict[str, Any], tokenizer, trace: str) -> str:
    """Pass 2 prompt: same question turn, a *closed* <think> block containing
    the real Pass-1 trace -- the trace-conditioned analog of score_mcq_ll.py's
    format_mcq_prompt(after_think=True), which forces an EMPTY think block."""
    messages = build_question_messages(item)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + f"<think>\n{trace}\n</think>\n\n"


def score_with_trace(
    model,
    tokenizer,
    item: dict[str, Any],
    trace: str,
    letter_ids: dict[str, int],
    score_symbol,
    device: str,
) -> dict[str, float]:
    """Pass 2 scoring for an arbitrary trace string (real or empty -- the
    empty-trace call is I3's delta_trace ablation, reusing this unchanged)."""
    prompt = build_trace_prompt(item, tokenizer, trace)
    return score_symbol(model, tokenizer, prompt, letter_ids, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace-conditioned (2-pass) MCQ scoring -- reason then extract.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to merged model directory")
    parser.add_argument("--dataset", default="data/val_v3.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/mcq_trace_ll.json")
    parser.add_argument(
        "--per-item-output",
        default=None,
        help="JSONL path (default: --output with .jsonl suffix)",
    )
    parser.add_argument("--max-trace-tokens", type=int, default=DEFAULT_MAX_TRACE_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--diagnostics-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
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
        help="Use diagnostics full-precision (bf16) model loading instead of 4-bit NF4.",
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
        args.model, full_precision=args.full_precision, device=args.device
    )
    letter_ids = helpers["get_letter_token_ids"](tokenizer)
    score_symbol = helpers["score_symbol"]

    per_item: list[dict[str, Any]] = []
    log.info("Scoring %d MCQ rows (2-pass, trace-conditioned) from %s", len(records), args.dataset)
    for item in records:
        generated, trace, truncated, n_new_tokens = generate_trace(
            model, tokenizer, item, args.max_trace_tokens, args.device, args.seed
        )
        logprobs = score_with_trace(model, tokenizer, item, trace, letter_ids, score_symbol, args.device)
        probs = _softmax_probs(logprobs)
        top_letter = max(LETTERS, key=lambda letter: logprobs[letter])
        gold = item["answer"]
        per_item.append(
            {
                "question_id": item["question_id"],
                "category": item.get("category", "unknown"),
                "topic": item.get("topic", "unknown"),
                "language": item.get("language", "en"),
                "gold": gold,
                "trace": trace,
                "trace_raw_generation": generated,
                "trace_new_tokens": n_new_tokens,
                "trace_truncated": truncated,
                "gold_logprob": logprobs[gold],
                "gold_probability": probs[gold],
                "option_set_gold_logprob": math.log(probs[gold]),
                "top_letter": top_letter,
                "top_logprob": logprobs[top_letter],
                "correct_by_argmax": top_letter == gold,
                "logprobs": {letter: logprobs[letter] for letter in LETTERS},
            }
        )

    total = len(per_item)
    mean_gold_ll = sum(row["gold_logprob"] for row in per_item) / total
    mean_option_gold_ll = sum(row["option_set_gold_logprob"] for row in per_item) / total
    n_truncated = sum(1 for row in per_item if row["trace_truncated"])
    summary = {
        "format": "autoresearch_mcq_trace_ll_v1",
        "objective_note": (
            "Trace-conditioned (2-pass) option-set-normalized gold-letter log-likelihood. "
            "NOT comparable to score_mcq_ll.py's forced-empty-think score -- see "
            "TRACE_EVAL_REASONING_SFT_RLM_PLAN.md assumption 5 and objective_weights_v3.json "
            "(chainlink #109 / I4)."
        ),
        "chainlink_issue": 106,
        "model": args.model,
        "dataset": args.dataset,
        "diagnostics_dir": str(Path(args.diagnostics_dir)),
        "max_trace_tokens": args.max_trace_tokens,
        "seed": args.seed,
        "scored_mcq_rows": total,
        "skipped_non_mcq_rows": len(skipped),
        "mean_trace_gold_logprob": mean_gold_ll,
        "mean_trace_gold_nll": -mean_gold_ll,
        "mean_trace_option_set_gold_logprob": mean_option_gold_ll,
        "mean_trace_option_set_gold_nll": -mean_option_gold_ll,
        "argmax_accuracy": sum(1 for row in per_item if row["correct_by_argmax"]) / total,
        "traces_truncated_count": n_truncated,
        "traces_truncated_fraction": n_truncated / total,
        "full_precision": args.full_precision,
        "device": args.device,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "skipped": skipped}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    per_item_path = Path(args.per_item_output) if args.per_item_output else output_path.with_suffix(".jsonl")
    per_item_path.parent.mkdir(parents=True, exist_ok=True)
    with per_item_path.open("w", encoding="utf-8") as handle:
        for row in per_item:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"wrote {_as_repo_relative(output_path)}")
    print(f"wrote {_as_repo_relative(per_item_path)}")
    print(f"scored_mcq_rows={total}")
    print(f"skipped_non_mcq_rows={len(skipped)}")
    print(f"mean_trace_option_set_gold_logprob={mean_option_gold_ll:.8f}")
    print(f"argmax_accuracy={summary['argmax_accuracy']:.4f}")
    print(f"traces_truncated_fraction={summary['traces_truncated_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
