#!/usr/bin/env python3
"""Compare domain MCQ scores with and without a deployment system prompt.

This is a reported diagnostic, not a replacement for the frozen primary
objective. It keeps the model and MCQ rows fixed and changes only whether the
same explicit system message is supplied through the chat template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from autoresearch.score_mcq_ll import LETTERS, normalize_records
except ModuleNotFoundError:  # direct `python autoresearch/compare_system_prompt.py`
    from score_mcq_ll import LETTERS, normalize_records


def _load_helpers(diagnostics_dir: Path):
    try:
        from autoresearch.score_mcq_ll import import_diagnostics_helpers
    except ModuleNotFoundError:
        from score_mcq_ll import import_diagnostics_helpers

    return import_diagnostics_helpers(diagnostics_dir)


def _question_text(item: dict[str, Any]) -> str:
    return (
        f"{item['question']}\n\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}"
    )


def _prompt(tokenizer, item: dict[str, Any], system_prompt: str | None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _question_text(item)})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + "<think>\n</think>\n"


def _score_condition(
    model,
    tokenizer,
    helpers,
    records: list[dict[str, Any]],
    system_prompt: str | None,
    device: str,
) -> dict[str, Any]:
    letter_ids = helpers["get_letter_token_ids"](tokenizer)
    per_item = []
    for item in records:
        logprobs = helpers["score_symbol"](
            model,
            tokenizer,
            _prompt(tokenizer, item, system_prompt),
            letter_ids,
            device,
        )
        max_logprob = max(logprobs.values())
        probabilities = {
            letter: math.exp(logprobs[letter] - max_logprob) for letter in LETTERS
        }
        total = sum(probabilities.values())
        probabilities = {letter: value / total for letter, value in probabilities.items()}
        top_letter = max(LETTERS, key=lambda letter: logprobs[letter])
        per_item.append(
            {
                "question_id": item["question_id"],
                "gold": item["answer"],
                "top_letter": top_letter,
                "correct_by_argmax": top_letter == item["answer"],
                "option_set_gold_logprob": math.log(probabilities[item["answer"]]),
            }
        )
    total = len(per_item)
    return {
        "rows": total,
        "mean_option_set_gold_logprob": sum(
            row["option_set_gold_logprob"] for row in per_item
        ) / total,
        "argmax_accuracy": sum(row["correct_by_argmax"] for row in per_item) / total,
        "per_item": per_item,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--diagnostics-dir", default="../benchmark-eval-workspace/diagnostics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--full-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_rows = [
        json.loads(line)
        for line in Path(args.dataset).open(encoding="utf-8")
        if line.strip()
    ]
    records, skipped = normalize_records(dataset_rows)
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        parser.error("dataset contains no scorable MCQ rows")
    system_path = Path(args.system_prompt_file)
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    helpers = _load_helpers(Path(args.diagnostics_dir))

    results: dict[str, Any] = {}
    for label, model_path in (
        ("baseline", args.baseline_model),
        ("candidate", args.candidate_model),
    ):
        model, tokenizer = helpers["load_model_and_tokenizer"](
            model_path,
            full_precision=args.full_precision,
            device=args.device,
        )
        without = _score_condition(model, tokenizer, helpers, records, None, args.device)
        with_prompt = _score_condition(
            model, tokenizer, helpers, records, system_prompt, args.device
        )
        results[label] = {
            "model": model_path,
            "without_system_prompt": without,
            "with_system_prompt": with_prompt,
            "delta_with_minus_without": {
                "mean_option_set_gold_logprob": (
                    with_prompt["mean_option_set_gold_logprob"]
                    - without["mean_option_set_gold_logprob"]
                ),
                "argmax_accuracy": (
                    with_prompt["argmax_accuracy"] - without["argmax_accuracy"]
                ),
            },
        }
        del model, tokenizer
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    payload = {
        "format": "system_prompt_mcq_comparison_v1",
        "dataset": args.dataset,
        "scored_rows": len(records),
        "skipped_non_mcq_rows": len(skipped),
        "system_prompt_file": args.system_prompt_file,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "device": args.device,
        "results": results,
        "interpretation": (
            "reported diagnostic only; the frozen primary gate remains unchanged"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
