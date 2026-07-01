#!/usr/bin/env python3
"""
merge_adapter.py - Merge a LoRA adapter into the base Qwen3-0.6B weights.

Produces a standalone HuggingFace checkpoint in 16-bit safetensors that can
be used for inference or converted to Cactus .cact format.

Usage:
  # Activate conda environment first:
  #   conda activate qwen3-finetune

  python finetune/merge_adapter.py

  # Override defaults:
  python finetune/merge_adapter.py \\
    --base Qwen/Qwen3-0.6B \\
    --adapter finetune/outputs/lora-adapter/final \\
    --output finetune/outputs/merged-model \\
    --dtype float16

Output:
  finetune/outputs/merged-model/   — merged model in safetensors + tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _try_unsloth_merge(
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    max_seq_len: int,
    dtype: torch.dtype,
) -> bool:
    """
    Attempt merge via Unsloth's save_pretrained_merged.

    Returns True on success, False if Unsloth is not available.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError:
        return False

    print("Using Unsloth merge path ...", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Unsloth merges and saves in one call; "merged_16bit" writes safetensors.
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    return True


def _peft_merge(
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    dtype: torch.dtype,
) -> None:
    """Merge via PEFT PeftModel.merge_and_unload (fallback when Unsloth absent)."""
    from peft import PeftModel  # type: ignore[import]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    print("Using PEFT merge path ...", file=sys.stderr)

    print(f"  Loading base model: {base_model}", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Loading LoRA adapter: {adapter_path}", file=sys.stderr)
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))

    print("  Merging weights ...", file=sys.stderr)
    merged = peft_model.merge_and_unload()

    print(f"  Saving merged model → {output_dir}", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))


def _resolve_adapter_path(adapter_arg: str) -> Path:
    """
    Prefer the 'final/' sub-directory written by train.py, fall back to the
    adapter root if 'final/' does not exist.
    """
    base = Path(adapter_arg)
    final = base / "final"
    if final.exists():
        return final
    if base.exists():
        return base
    raise FileNotFoundError(
        f"LoRA adapter not found at {base} or {final}. "
        "Run finetune/train.py first."
    )


def _dtype_from_str(s: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if s not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unknown dtype '{s}'. Choose from: {', '.join(mapping)}"
        )
    return mapping[s]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the Qwen3-0.6B base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace base model ID or local path (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--adapter",
        default="finetune/outputs/lora-adapter",
        metavar="DIR",
        help=(
            "LoRA adapter directory (default: finetune/outputs/lora-adapter). "
            "Automatically looks for 'final/' sub-directory first."
        ),
    )
    parser.add_argument(
        "--output",
        default="finetune/outputs/merged-model",
        metavar="DIR",
        help="Output directory for merged model (default: finetune/outputs/merged-model)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        type=_dtype_from_str,
        metavar="DTYPE",
        help=(
            "Weight dtype for the merged model: float16 (default), bfloat16, float32. "
            "float16 produces the smallest safetensors files."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max sequence length passed to Unsloth (ignored for PEFT path; default: 2048)",
    )
    parser.add_argument(
        "--force-peft",
        action="store_true",
        help="Skip Unsloth and use the PEFT merge path even if Unsloth is available",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        adapter_path = _resolve_adapter_path(args.adapter)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"WARNING: Output directory {output_dir} already exists and is non-empty. "
            "Existing files will be overwritten.",
            file=sys.stderr,
        )

    print(f"Base model : {args.base}", file=sys.stderr)
    print(f"Adapter    : {adapter_path}", file=sys.stderr)
    print(f"Output     : {output_dir}", file=sys.stderr)
    print(f"Dtype      : {args.dtype}", file=sys.stderr)
    print("", file=sys.stderr)

    merged = False
    if not args.force_peft:
        merged = _try_unsloth_merge(
            base_model=args.base,
            adapter_path=adapter_path,
            output_dir=output_dir,
            max_seq_len=args.max_seq_len,
            dtype=args.dtype,
        )

    if not merged:
        _peft_merge(
            base_model=args.base,
            adapter_path=adapter_path,
            output_dir=output_dir,
            dtype=args.dtype,
        )

    # Verify output
    safetensors_files = list(output_dir.glob("*.safetensors"))
    if not safetensors_files:
        print(
            "ERROR: No .safetensors files found in output directory after merge.",
            file=sys.stderr,
        )
        sys.exit(1)

    total_bytes = sum(f.stat().st_size for f in safetensors_files)
    total_mb = total_bytes / (1024 ** 2)
    print(
        f"\nMerge complete. {len(safetensors_files)} safetensors file(s), "
        f"{total_mb:.0f} MB total.",
        file=sys.stderr,
    )
    print(
        "\nNext step: run finetune/convert_cact.py to convert to Cactus .cact format.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
