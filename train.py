#!/usr/bin/env python3
"""
train.py - QLoRA fine-tuning of Qwen3-0.6B using Unsloth + TRL SFTTrainer.

Usage:
  # Activate conda environment first:
  #   conda activate qwen3-finetune

  python finetune/train.py

  # Resume interrupted run:
  python finetune/train.py --resume

  # Override defaults:
  python finetune/train.py \\
    --model unsloth/Qwen3-0.6B \\
    --train-data finetune/data/train.jsonl \\
    --val-data finetune/data/val.jsonl \\
    --output-dir finetune/outputs/lora-adapter \\
    --epochs 3 \\
    --lr 2e-4 \\
    --batch-size 4 \\
    --grad-accum 4 \\
    --lora-r 32 \\
    --lora-alpha 64 \\
    --max-seq-len 2048 \\
    --wandb  # enable W&B logging

Output:
  finetune/outputs/lora-adapter/   — LoRA adapter (PEFT format)
  finetune/outputs/lora-adapter/final/  — final checkpoint after all epochs
"""

from __future__ import annotations

import os
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # must be set before unsloth import

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    from unsloth import FastLanguageModel  # must be FIRST before downstream training deps
else:
    FastLanguageModel = None



# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if "instruction" not in rec or "output" not in rec:
                raise ValueError(
                    f"{path}:{lineno}: record must have 'instruction' and 'output' fields"
                )
            records.append(rec)
    return records


def build_dataset(
    pairs: list[dict[str, str]],
    tokenizer,
    max_seq_len: int,
    packing: bool = False,
) -> Dataset:
    """Format pairs as ChatML strings and return a HuggingFace Dataset.

    The default path preserves the original behavior: each example is tokenized
    independently with truncation to max_seq_len. When packing is enabled, return
    raw text and let TRL SFTTrainer perform its built-in packing.
    """
    from datasets import Dataset

    texts = [
        CHAT_TEMPLATE.format(
            instruction=p["instruction"].strip(),
            output=p["output"].strip(),
        )
        for p in pairs
    ]

    raw_ds = Dataset.from_dict({"text": texts})
    if packing:
        return raw_ds

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )

    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def parse_target_modules(raw: str) -> list[str]:
    modules = [module.strip() for module in raw.split(",") if module.strip()]
    if not modules:
        raise argparse.ArgumentTypeError("--target-modules must contain at least one module")
    return modules


def load_model_with_lora(
    model_name: str,
    max_seq_len: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> tuple:
    """Load Qwen3-0.6B in 4-bit and attach LoRA adapters."""
    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)

    if FastLanguageModel is None:
        raise RuntimeError("Unsloth is required for training")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,  # auto-detect: bf16 if supported, else fp16
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")


    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Return the path to the most recent checkpoint in output_dir, or None."""
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return str(checkpoints[-1])
    return None


def build_training_args(
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum: int,
    use_wandb: bool,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    early_stop_patience: Optional[int] = None,
    packing: bool = False,
    max_seq_len: Optional[int] = None,
):
    import torch
    from trl import SFTConfig

    report_to = "wandb" if use_wandb else "none"
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        learning_rate=lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if early_stop_patience is not None else "no",
        optim="adamw_8bit",
        lr_scheduler_type=lr_scheduler,
        weight_decay=weight_decay,
        seed=42,
        report_to=report_to,
        run_name="qwen3-0.6b-nf-qlora",
    )

    if early_stop_patience is not None:
        kwargs.update(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    signature = inspect.signature(SFTConfig)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "packing" in parameters or accepts_kwargs:
        kwargs["packing"] = packing
    elif packing:
        raise RuntimeError("Installed TRL SFTConfig does not support packing=True")

    if packing and max_seq_len is not None:
        if "max_length" in parameters:
            kwargs["max_length"] = max_seq_len
        elif "max_seq_length" in parameters:
            kwargs["max_seq_length"] = max_seq_len

    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen3-0.6B on natural farming instruction pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-0.6B",
        help="HuggingFace model ID or local path (default: unsloth/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--train-data",
        default="finetune/data/train.jsonl",
        metavar="PATH",
        help="Training JSONL file (default: finetune/data/train.jsonl)",
    )
    parser.add_argument(
        "--val-data",
        default="finetune/data/val.jsonl",
        metavar="PATH",
        help="Validation JSONL file (default: finetune/data/val.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="finetune/outputs/lora-adapter",
        metavar="DIR",
        help="Output directory for checkpoints and final adapter (default: finetune/outputs/lora-adapter)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of training epochs (default: 6)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Peak learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "linear", "constant"),
        default="cosine",
        help="Learning-rate scheduler (default: cosine)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio (default: 0.05)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size (default: 4)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank r (default: 32)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha (default: 64)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default: 0)",
    )
    parser.add_argument(
        "--target-modules",
        type=parse_target_modules,
        default=",".join(DEFAULT_TARGET_MODULES),
        help=(
            "Comma-separated LoRA target modules "
            f"(default: {','.join(DEFAULT_TARGET_MODULES)})"
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        metavar="N",
        help="Enable eval-loss early stopping with N epochs patience (default: disabled)",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable TRL SFTTrainer sequence packing (default: disabled)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb login)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import torch
    from transformers import EarlyStoppingCallback
    from trl import SFTTrainer

    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    output_dir = Path(args.output_dir)

    for p in (train_path, val_path):
        if not p.exists():
            parser.error(f"Data file not found: {p}")

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available. Training on CPU will be extremely slow.",
            file=sys.stderr,
        )

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading datasets ...", file=sys.stderr)
    train_pairs = load_jsonl(train_path)
    val_pairs = load_jsonl(val_path)
    print(f"  Train: {len(train_pairs)} pairs", file=sys.stderr)
    print(f"  Val:   {len(val_pairs)} pairs", file=sys.stderr)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}", file=sys.stderr)
    model, tokenizer = load_model_with_lora(
        model_name=args.model,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    # ── Build HuggingFace datasets ─────────────────────────────────────────
    print("Tokenizing datasets ...", file=sys.stderr)
    train_ds = build_dataset(train_pairs, tokenizer, args.max_seq_len, packing=args.packing)
    val_ds = build_dataset(val_pairs, tokenizer, args.max_seq_len, packing=args.packing)

    # ── Configure training ─────────────────────────────────────────────────
    training_args = build_training_args(
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        use_wandb=args.wandb,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        packing=args.packing,
        max_seq_len=args.max_seq_len,
    )

    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    callbacks = []
    if args.early_stop_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        callbacks=callbacks,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None
    if args.resume:
        resume_from = find_latest_checkpoint(output_dir)
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}", file=sys.stderr)
        else:
            print("No checkpoint found in output dir, starting fresh.", file=sys.stderr)

    print("\nStarting training ...\n", file=sys.stderr)
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final adapter ─────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter → {final_dir}", file=sys.stderr)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\nDone. Next step: run finetune/merge.py to merge adapter into base model.", file=sys.stderr)


if __name__ == "__main__":
    main()
