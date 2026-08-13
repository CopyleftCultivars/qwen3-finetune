#!/usr/bin/env python3
"""
score_grpo_primary.py - GRPO-tier primary objective: mean autoresearch/grpo_reward.py
`combined_reward` over a held-out slice of Agriculture-Agent-RL-Training-Data prompts,
generated fresh by the candidate model (not the dataset's own recorded completions).

Why this exists (levers.md Tier 4, "GRPO primary objective metric", PENDING-UNLOCK
until this script landed): the existing primary objective (objective_weights.json,
MCQ option-set gold-LL only) cannot measure what GRPO optimizes for --
generate_freeform_candidates.py has no tool-call generation support at all. Reusing
score_primary.py's MCQ number for a GRPO accept/reject decision would gate on a metric
blind to the capability actually being trained. The existing MCQ/freeform harness
still runs against a GRPO candidate, but only as a forgetting/regression guardrail
(gate.py's forgetting_guard/rubric_guard), never as this tier's primary signal.

The eval slice is the deterministic "eval" bucket from
autoresearch/grpo_dataset.py::holdout_bucket() -- same --holdout-seed/--holdout-fraction
as train_grpo.py's "train" bucket keeps the two disjoint, so this is a genuine held-out
score, not a train-set replay.

Usage:
  conda activate qwen3-grpo

  python autoresearch/score_grpo_primary.py \\
    --model outputs/merged-model-phase23-grpo-<name> \\
    --output results/primary_grpo_phase23-<name>.json

  # plan-only, no model load / no generation:
  python autoresearch/score_grpo_primary.py --model outputs/merged-model-phase23-grpo-<name> --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.grpo_dataset import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_HOLDOUT_SEED,
    EXCLUDED_GENERATOR_AGENTS,
    load_rollout_prompts,
)
from autoresearch.grpo_reward import (  # noqa: E402
    DEFAULT_WEIGHTS,
    citation_reward,
    combined_reward,
    content_quality_proxy_reward,
    ethics_redflag_reward,
    tool_use_reward,
)

DEFAULT_OUTPUT = "results/primary_grpo_reward.json"

# Matches Qwen3's chat_template.jinja tool-call convention exactly:
# <tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split a raw generation into (visible_text, tool_calls), matching the
    {"function": {"name": ..., "arguments": <json str>}} shape grpo_reward.py
    expects (same shape the training dataset's own completion turns use)."""
    tool_calls = []
    for i, match in enumerate(TOOL_CALL_RE.finditer(text)):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": payload.get("name"),
                    "arguments": json.dumps(payload.get("arguments", {})),
                },
            }
        )
    visible_text = TOOL_CALL_RE.sub("", text).strip()
    return visible_text, tool_calls


def load_reward_weights(path: Optional[Path]) -> dict[str, float]:
    if path is None:
        return DEFAULT_WEIGHTS
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["component_weights"]


def generate_completions(
    model,
    tokenizer,
    prompts: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Generate one assistant completion per prompt, wrapped as a single-turn
    completion list matching the shape grpo_reward.py's reward functions consume."""
    import torch

    torch.manual_seed(seed)
    completions = []
    device = next(model.parameters()).device
    for messages in prompts:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        text, tool_calls = parse_tool_calls_from_text(generated)
        completions.append([{"role": "assistant", "content": text, "tool_calls": tool_calls}])
    return completions


def build_result(
    prompts: list[Any],
    completions: list[list[dict[str, Any]]],
    tasks: list[Optional[str]],
    infos: list[Optional[dict]],
    weights: dict[str, float],
    model_path: str,
    n_holdout_total: int,
) -> dict[str, Any]:
    combined = combined_reward(prompts, completions, task=tasks, info=infos, weights=weights)
    tool = tool_use_reward(prompts, completions, task=tasks, info=infos)
    content = content_quality_proxy_reward(prompts, completions)
    citation = citation_reward(prompts, completions, task=tasks, info=infos)
    ethics = ethics_redflag_reward(prompts, completions)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "format": "autoresearch_grpo_primary_objective_result_v1",
        "objective_direction": "maximize",
        "role": "grpo_tier_primary_accept_revert_signal",
        "not_comparable_to": "autoresearch_primary_objective_result_v2 (MCQ gold-LL is a different tier's metric)",
        "primary": {
            "label": "grpo_mean_combined_reward_on_holdout",
            "score": mean(combined),
            "n_eval_prompts": len(combined),
            "n_holdout_pool_total": n_holdout_total,
            "components": {
                "tool_use_correctness": mean(tool),
                "content_quality_proxy": mean(content),
                "citation_presence": mean(citation),
                "ethics_redflags": mean(ethics),
            },
        },
        "reward_weights": weights,
        "inputs": {
            "model": model_path,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Merged candidate model directory (merge_adapter.py output)")
    parser.add_argument("--eval-data", default=DEFAULT_DATASET_ID)
    parser.add_argument("--exclude-sources", default=",".join(EXCLUDED_GENERATOR_AGENTS))
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--n-eval-samples", type=int, default=None, metavar="N",
                         help="Cap the eval slice to N prompts (default: use the full holdout bucket)")
    parser.add_argument("--max-new-tokens", type=int, default=768,
                         help="Match train_grpo.py's --max-completion-len; 128 was empirically too "
                              "short (see grpo_stage1_calibration.json)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reward-weights", type=Path, default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                         help="Load the eval prompt set + print the plan, skip model load and generation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    exclude_sources = frozenset(s.strip() for s in args.exclude_sources.split(",") if s.strip())
    weights = load_reward_weights(args.reward_weights)

    if args.dry_run:
        print(json.dumps({
            "model": args.model,
            "eval_data": args.eval_data,
            "exclude_sources": sorted(exclude_sources),
            "holdout_fraction": args.holdout_fraction,
            "holdout_seed": args.holdout_seed,
            "n_eval_samples": args.n_eval_samples,
            "max_new_tokens": args.max_new_tokens,
            "reward_weights": weights,
            "output": args.output,
        }, indent=2))
        return

    print(f"Loading held-out eval prompts (holdout_fraction={args.holdout_fraction}, "
          f"holdout_seed={args.holdout_seed}) ...", file=sys.stderr)
    eval_ds = load_rollout_prompts(
        dataset_id=args.eval_data,
        exclude_sources=exclude_sources,
        holdout="eval",
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
    )
    n_holdout_total = len(eval_ds)
    if args.n_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.n_eval_samples, len(eval_ds))))
    print(f"  Eval prompts: {len(eval_ds)} (of {n_holdout_total} in the full holdout bucket)", file=sys.stderr)

    print(f"\nLoading model: {args.model}", file=sys.stderr)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    prompts = eval_ds["prompt"]
    tasks = eval_ds["task"]
    infos = eval_ds["info"]

    print(f"\nGenerating {len(prompts)} completions (max_new_tokens={args.max_new_tokens}) ...", file=sys.stderr)
    completions = generate_completions(
        model, tokenizer, prompts, args.max_new_tokens, args.temperature, args.seed
    )

    result = build_result(prompts, completions, tasks, infos, weights, args.model, n_holdout_total)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", file=sys.stderr)
    print(f"primary.score = {result['primary']['score']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
