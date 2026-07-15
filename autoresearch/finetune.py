#!/usr/bin/env python3
"""Autoresearch harness entrypoint.

This script is the integration point for Phase 4 search runs. It deliberately
invokes the data prep, training, and merge scripts as subprocesses rather than
importing them. In particular, train.py and merge_adapter.py both rely on
Unsloth import-time patching, so each step gets a clean Python process.

Default data-source choice for this skeleton:
  - conversations: data/sources/gemma4/hf_dataset/train.jsonl
  - QMD MCQs:      data/sources/qmd_mcq_all_phase3.jsonl
  - MCQ rationale: disabled by default, because the historical default is
    benchmark-contaminated and prepare_dataset_v3.py now refuses it by name.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONVERSATIONS = "data/sources/gemma4/hf_dataset/train.jsonl"
DEFAULT_QMD_MCQ = "data/sources/qmd_mcq_all_phase3.jsonl"
DEFAULT_DATASET_OUTPUT_DIR = "data"
DEFAULT_OUTPUT_ROOT = "outputs"


def _run(argv: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def _dataset_suffix(run_name: str) -> str:
    return f"_{run_name}"


def _dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    suffix = _dataset_suffix(args.run_name)
    output_dir = Path(args.dataset_output_dir)
    return output_dir / f"train{suffix}.jsonl", output_dir / f"val{suffix}.jsonl"


def _prepare_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable,
        "prepare_dataset_v3.py",
        "--output-dir",
        args.dataset_output_dir,
        "--suffix",
        _dataset_suffix(args.run_name),
        "--split",
        str(args.split),
        "--seed",
        str(args.seed),
        "--min-output-chars",
        str(args.min_output_chars),
        "--max-total-chars",
        str(args.max_total_chars),
        "--sft-format",
        args.sft_format,
    ]
    if args.conversations:
        argv.extend(["--conversations", args.conversations])
    if args.mcq_rationale:
        argv.extend(["--mcq-rationale", args.mcq_rationale])
    if args.qmd_mcq:
        argv.extend(["--qmd-mcq", args.qmd_mcq])
    if args.conversation_ratio is not None:
        argv.extend(["--conversation-ratio", str(args.conversation_ratio)])
    if args.target_permutation_ids:
        argv.extend(["--target-permutation-ids", args.target_permutation_ids])
    return argv


def _train_argv(args: argparse.Namespace) -> list[str]:
    train_data, val_data = _dataset_paths(args)
    lora_output_dir = Path(args.output_root) / f"lora-adapter-{args.run_name}"
    argv = [
        sys.executable,
        "train.py",
        "--model",
        args.model,
        "--train-data",
        str(train_data),
        "--val-data",
        str(val_data),
        "--output-dir",
        str(lora_output_dir),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--lr-scheduler",
        args.lr_scheduler,
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    if args.early_stop_patience is not None:
        argv.extend(["--early-stop-patience", str(args.early_stop_patience)])
    if args.packing:
        argv.append("--packing")
    if args.wandb:
        argv.append("--wandb")
    if args.resume:
        argv.append("--resume")
    return argv


def _merge_argv(args: argparse.Namespace) -> list[str]:
    lora_output_dir = Path(args.output_root) / f"lora-adapter-{args.run_name}"
    merged_output_dir = Path(args.output_root) / f"merged-model-{args.run_name}"
    argv = [
        sys.executable,
        "merge_adapter.py",
        "--base",
        args.merge_base,
        "--adapter",
        str(lora_output_dir),
        "--output",
        str(merged_output_dir),
        "--dtype",
        args.merge_dtype,
        "--max-seq-len",
        str(args.max_seq_len),
    ]
    if args.force_peft:
        argv.append("--force-peft")
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checksum-gated autoresearch fine-tune harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Bookkeeping / control.
    parser.add_argument("--run-name", required=True, help="Run suffix for output paths")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="YAML config path; not implemented until chainlink #31",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checksum check, print subprocess argv, then stop",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-output-dir", default=DEFAULT_DATASET_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--min-output-chars", type=int, default=10)
    parser.add_argument("--max-total-chars", type=int, default=8192)

    # Data/source controls. The source defaults intentionally avoid the known
    # contaminated MCQ-rationale file; callers may pass a verified clean source.
    parser.add_argument("--conversations", default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--mcq-rationale", default=None)
    parser.add_argument("--qmd-mcq", default=DEFAULT_QMD_MCQ)

    # SEARCH DIMENSION: Tier 1 hyperparameters.
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "linear", "constant"),
        default="cosine",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--packing", action="store_true")

    # SEARCH DIMENSION: Tier 2 data/format knobs.
    parser.add_argument("--conversation-ratio", type=float, default=None)
    parser.add_argument("--target-permutation-ids", default=None, metavar="PATH")
    parser.add_argument(
        "--sft-format",
        choices=("bare-letter", "answer-prefix", "sentence"),
        default="bare-letter",
    )
    parser.add_argument(
        "--synth-item-count",
        type=int,
        default=None,
        help="Accepted search knob stub; data generation scaling is not wired yet",
    )

    # Pass-through model/merge controls; not search dimensions.
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B")
    parser.add_argument("--merge-base", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--merge-dtype",
        choices=("float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
        default="float16",
    )
    parser.add_argument("--force-peft", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.config is not None:
        parser.error("--config is not yet implemented; see chainlink #31")
    if args.synth_item_count is not None:
        parser.error(
            "--synth-item-count is accepted but not yet wired; "
            "QMD data-generation scaling lands in a later subissue"
        )
    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between 0 and 1 (exclusive)")
    if args.conversation_ratio is not None and not 0.0 < args.conversation_ratio < 1.0:
        parser.error("--conversation-ratio must be between 0 and 1 (exclusive)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    # Security boundary: checksum gate must be the first subprocess.
    subprocess.run(
        [sys.executable, "autoresearch/freeze_manifest.py", "check"],
        check=True,
    )

    commands = [_prepare_argv(args), _train_argv(args), _merge_argv(args)]
    if args.dry_run:
        print("Dry run: checksum passed; planned subprocesses:")
    for argv in commands:
        _run(argv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
