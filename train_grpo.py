#!/usr/bin/env python3
"""
train_grpo.py - Online GRPO training of Qwen3-0.6B (continuing from an existing
SFT LoRA adapter) on CopyleftCultivars/Agriculture-Agent-RL-Training-Data, using
Unsloth + TRL GRPOTrainer.

Verified compatible stack (2026-08-13, RTX 4070 Ti, `qwen3-grpo` conda env cloned
from the working `qwen3-train` SFT env so GRPO-specific package changes never risk
the SFT pipeline): torch 2.10.0+cu128, transformers 4.57.6, trl 0.24.0 (GRPOTrainer
present), peft 0.19.1, unsloth 2026.6.9 (PatchFastRL present).

Reward: autoresearch/grpo_reward.py's combined_reward -- a static reimplementation
of the dataset's info.reward_components, no live cultivars-mcp/PubChem calls in
the training loop. See that module's docstring for why.

Usage:
  conda activate qwen3-grpo

  # tiny smoke test (Stage 1 of the plan)
  python train_grpo.py \\
    --adapter outputs/lora-adapter-p6-lr5e5-r2/final \\
    --output-dir outputs/lora-adapter-grpo-smoke \\
    --max-steps 5 --num-generations 4 --batch-size 1 --max-samples 50

  # a real phase iteration (Stage 2), invoked by autoresearch/grpo_finetune.py
  python train_grpo.py \\
    --adapter outputs/lora-adapter-p6-lr5e5-r2/final \\
    --output-dir outputs/lora-adapter-grpo-phase23-<name> \\
    --max-steps 200 --num-generations 8

Output:
  <output-dir>/final/  -- LoRA adapter (PEFT format), ready for merge_adapter.py
"""

from __future__ import annotations

import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # must be set before unsloth import

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    from unsloth import FastLanguageModel, PatchFastRL  # must be FIRST before trl/transformers
    PatchFastRL(algorithm="GRPO", FastLanguageModel=FastLanguageModel)
else:
    FastLanguageModel = None
    PatchFastRL = None

from autoresearch.grpo_dataset import EXCLUDED_GENERATOR_AGENTS, DEFAULT_DATASET_ID, load_rollout_prompts  # noqa: E402
from autoresearch.grpo_reward import combined_reward, DEFAULT_WEIGHTS  # noqa: E402

DEFAULT_ADAPTER = "outputs/lora-adapter-p6-lr5e5-r2/final"
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


# ---------------------------------------------------------------------------
# Model loading -- continue from an existing LoRA adapter, not a fresh get_peft_model
# ---------------------------------------------------------------------------

def load_policy_with_lora(adapter_path: str, max_seq_len: int, load_in_4bit: bool = True) -> tuple:
    """Load a saved LoRA adapter (e.g. the v5clean SFT checkpoint) as the GRPO
    starting policy, same idiom merge_adapter.py already uses for the inference
    path, but kept trainable here via FastLanguageModel.for_training()."""
    if FastLanguageModel is None:
        raise RuntimeError("Unsloth is required for training")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_len,
        load_in_4bit=load_in_4bit,
        dtype=None,
    )
    FastLanguageModel.for_training(model)

    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Reward wiring
# ---------------------------------------------------------------------------

def load_reward_weights(path: Optional[Path]) -> dict[str, float]:
    if path is None:
        return DEFAULT_WEIGHTS
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["component_weights"]


def make_reward_fn(weights: dict[str, float]):
    def grpo_combined_reward(prompts, completions, **kwargs) -> list[float]:
        return combined_reward(prompts, completions, weights=weights, **kwargs)

    grpo_combined_reward.__name__ = "grpo_combined_reward"
    return grpo_combined_reward


# ---------------------------------------------------------------------------
# GRPO config
# ---------------------------------------------------------------------------

def build_grpo_config(
    output_dir: Path,
    num_generations: int,
    lr: float,
    beta: float,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    max_prompt_len: int,
    max_completion_len: int,
    temperature: float,
    use_wandb: bool,
    seed: int,
):
    import torch
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(output_dir),
        num_generations=num_generations,
        learning_rate=lr,
        beta=beta,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        temperature=temperature,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="no",
        optim="adamw_8bit",
        seed=seed,
        report_to="wandb" if use_wandb else "none",
        run_name="qwen3-0.6b-grpo-agri-rl",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online GRPO training on Agriculture-Agent-RL-Training-Data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B",
                         help="Base model ID, used only to resolve tokenizer defaults if --adapter is base-only")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, metavar="DIR",
                         help=f"LoRA adapter to continue from (default: {DEFAULT_ADAPTER})")
    parser.add_argument("--train-data", default=DEFAULT_DATASET_ID, metavar="HF_ID_OR_PATH",
                         help=f"HF dataset id or local JSONL snapshot (default: {DEFAULT_DATASET_ID})")
    parser.add_argument("--exclude-sources", default=",".join(EXCLUDED_GENERATOR_AGENTS), metavar="LIST",
                         help="Comma-separated generator_agent values to exclude (default: agy)")
    parser.add_argument("--max-samples", type=int, default=None, metavar="N",
                         help="Cap the rollout prompt set to N examples (for smoke tests)")
    parser.add_argument("--output-dir", default="outputs/lora-adapter-grpo", metavar="DIR")
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO group size (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-6, help="GRPO learning rate (default: 2e-6)")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient (default: 0.04)")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--reward-weights", type=Path, default=None, metavar="PATH",
                         help="Path to grpo_reward_weights.json (default: built-in DEFAULT_WEIGHTS)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                         help="Load dataset + print planned config, skip model load and training")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    exclude_sources = frozenset(s.strip() for s in args.exclude_sources.split(",") if s.strip())
    weights = load_reward_weights(args.reward_weights)

    print(f"Loading rollout prompts from {args.train_data} (excluding: {sorted(exclude_sources)}) ...",
          file=sys.stderr)
    local_path = Path(args.train_data) if Path(args.train_data).exists() else None
    dataset_id = args.train_data if local_path is None else DEFAULT_DATASET_ID

    if args.dry_run:
        print(json.dumps({
            "dataset_id": dataset_id,
            "local_path": str(local_path) if local_path else None,
            "exclude_sources": sorted(exclude_sources),
            "adapter": args.adapter,
            "output_dir": args.output_dir,
            "num_generations": args.num_generations,
            "lr": args.lr,
            "beta": args.beta,
            "max_steps": args.max_steps,
            "reward_weights": weights,
            "max_samples": args.max_samples,
        }, indent=2))
        return

    train_ds = load_rollout_prompts(
        dataset_id=dataset_id,
        local_path=local_path,
        exclude_sources=exclude_sources,
    )
    if args.max_samples is not None:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
    print(f"  Rollout prompts: {len(train_ds)}", file=sys.stderr)

    print(f"\nLoading policy from adapter: {args.adapter}", file=sys.stderr)
    model, tokenizer = load_policy_with_lora(args.adapter, max_seq_len=args.max_seq_len)

    output_dir = Path(args.output_dir)
    grpo_config = build_grpo_config(
        output_dir=output_dir,
        num_generations=args.num_generations,
        lr=args.lr,
        beta=args.beta,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_prompt_len=args.max_prompt_len,
        max_completion_len=args.max_completion_len,
        temperature=args.temperature,
        use_wandb=args.wandb,
        seed=args.seed,
    )

    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(weights)],
        args=grpo_config,
        train_dataset=train_ds,
    )

    print("\nStarting GRPO training ...\n", file=sys.stderr)
    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter -> {final_dir}", file=sys.stderr)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\nDone. Next step: merge_adapter.py to produce a mergeable checkpoint for eval.", file=sys.stderr)


if __name__ == "__main__":
    main()
