#!/usr/bin/env python3
"""grpo_finetune.py - Autoresearch harness entrypoint for the GRPO (Tier 4 /
RL second-stage) lever family, paralleling autoresearch/finetune.py's pattern
for the SFT lever families.

Deliberately invokes train_grpo.py and merge_adapter.py as subprocesses (not
imports) for the same reason finetune.py does: Unsloth's import-time patching
wants a clean Python process per script.

Chain: freeze_manifest.py check -> train_grpo.py -> merge_adapter.py

Usage:
  conda activate qwen3-grpo

  python autoresearch/grpo_finetune.py --run-name phase23-grpo-smoke \\
    --adapter outputs/lora-adapter-p6-lr5e5-r2/final \\
    --max-steps 5 --num-generations 4 --max-samples 50 --dry-run

  python autoresearch/grpo_finetune.py --run-name phase23-grpo-smoke \\
    --adapter outputs/lora-adapter-p6-lr5e5-r2/final \\
    --max-steps 5 --num-generations 4 --max-samples 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_ADAPTER = "outputs/lora-adapter-p6-lr5e5-r2/final"
DEFAULT_OUTPUT_ROOT = "outputs"


def _run(argv: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def _train_argv(args: argparse.Namespace) -> list[str]:
    lora_output_dir = Path(args.output_root) / f"lora-adapter-{args.run_name}"
    argv = [
        sys.executable,
        "train_grpo.py",
        "--adapter",
        args.adapter,
        "--train-data",
        args.train_data,
        "--exclude-sources",
        args.exclude_sources,
        "--output-dir",
        str(lora_output_dir),
        "--num-generations",
        str(args.num_generations),
        "--lr",
        str(args.lr),
        "--beta",
        str(args.beta),
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-prompt-len",
        str(args.max_prompt_len),
        "--max-completion-len",
        str(args.max_completion_len),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
    ]
    if args.max_samples is not None:
        argv.extend(["--max-samples", str(args.max_samples)])
    if args.reward_weights:
        argv.extend(["--reward-weights", args.reward_weights])
    if args.wandb:
        argv.append("--wandb")
    return argv


def _merge_argv(args: argparse.Namespace) -> list[str]:
    lora_output_dir = Path(args.output_root) / f"lora-adapter-{args.run_name}"
    merged_output_dir = Path(args.output_root) / f"merged-model-{args.run_name}"
    return [
        sys.executable,
        "merge_adapter.py",
        "--base",
        args.merge_base,
        "--adapter",
        str(lora_output_dir / "final"),
        "--output",
        str(merged_output_dir),
        "--dtype",
        args.merge_dtype,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True, help="e.g. phase23-grpo-<name>")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--train-data", default="CopyleftCultivars/Agriculture-Agent-RL-Training-Data")
    parser.add_argument("--exclude-sources", default="agy")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--max-completion-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward-weights", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-base", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--merge-dtype", default="float16")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Security boundary: checksum gate must be the first subprocess (same
    # discipline as finetune.py's SFT chain).
    subprocess.run([sys.executable, "autoresearch/freeze_manifest.py", "check"], check=True)

    commands = [_train_argv(args), _merge_argv(args)]
    if args.dry_run:
        print("Dry run: checksum passed; planned subprocesses:")
    for argv in commands:
        _run(argv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
