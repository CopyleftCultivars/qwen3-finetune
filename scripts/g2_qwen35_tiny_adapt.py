#!/usr/bin/env python3
"""#238 G2: derive Qwen3.5-2B LoRA module targets and run a few real optimizer
steps on non-benchmark natural-farming training examples.

What this proves (and what it does not):
  * derives the ACTUAL module inventory of Qwen3.5-2B (18 Gated-DeltaNet blocks
    + 6 full-attention blocks + MLP + vision tower) instead of assuming the
    Qwen3 target recipe transfers;
  * attaches bf16 LoRA in the isolated transformers 5.5.0 + Unsloth 2026.6.9
    environment from the #236 retry and records which modules PEFT really
    adapted vs. which were intentionally left out (vision tower, embeddings,
    lm_head, and any GDN projections Unsloth filters away);
  * runs real optimizer steps: finite loss, finite gradients, adapter delta,
    wall time, peak VRAM;
  * saves the adapter + tokenizer for the separate reload/merge probe.

Run inside the #236 retry isolated venv:
  localllm-qwen3.5/retry-evidence/q1-isolated/venv/bin/python \
      scripts/g2_qwen35_tiny_adapt.py --metrics-out results/g2_tiny_adapt.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

import torch  # noqa: E402

from g2_training_rows import load_examples  # noqa: E402

# PEFT leaf names we ask for. Standard Qwen recipe plus the Gated DeltaNet
# projections so the delta layers are covered unless Unsloth filters them.
STANDARD_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")
GDN_TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
GDN_LEAVES = set(GDN_TARGETS)
FULL_ATTN_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LEAVES = {"gate_proj", "up_proj", "down_proj"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data", default="data/train.jsonl",
                        help="non-benchmark training JSONL (instruction/output)")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out-dir", default="outputs/lora-adapter-g2-qwen35-tiny")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--derive-only", action="store_true")
    return parser.parse_args()


def classify_module(name: str) -> str:
    """Group a module name: vision / full_attn / gdn_attn / mlp / head / other."""
    leaf = name.rsplit(".", 1)[-1]
    if name.startswith("model.visual"):
        return "vision"
    if name == "lm_head" or leaf == "embed_tokens":
        return "head_or_embed"
    if ".layers." not in name:
        return "other"
    if leaf in FULL_ATTN_LEAVES:
        return "full_attn"
    if leaf in GDN_LEAVES:
        return "gdn_attn"
    if leaf in MLP_LEAVES:
        return "mlp"
    return "other"


def module_inventory(model: torch.nn.Module) -> dict[str, Any]:
    linear_groups: Counter[str] = Counter()
    leaf_groups: dict[str, set[str]] = defaultdict(set)
    leaves: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        group = classify_module(name)
        leaf = name.rsplit(".", 1)[-1]
        linear_groups[group] += 1
        leaf_groups[leaf].add(group)
        if group not in ("vision", "head_or_embed"):
            leaves.append(leaf)
    return {
        "linear_total": sum(linear_groups.values()),
        "linear_by_group": dict(sorted(linear_groups.items())),
        "language_leaves": sorted(set(leaves)),
        "leaf_group_map": {k: sorted(v) for k, v in sorted(leaf_groups.items())},
    }


def is_lora_wrapped(module: torch.nn.Module) -> bool:
    lora_a = getattr(module, "lora_A", None)
    return hasattr(lora_a, "keys") and "default" in lora_a


def attach_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    from unsloth import FastLanguageModel

    targets = sorted(set(STANDARD_TARGETS) | set(GDN_TARGETS))
    return FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )


def adapted_coverage(model: torch.nn.Module) -> dict[str, Any]:
    per_leaf: Counter[str] = Counter()
    per_group: Counter[str] = Counter()
    adapted_names: list[str] = []
    for name, module in model.named_modules():
        if is_lora_wrapped(module):
            adapted_names.append(name)
            per_leaf[name.rsplit(".", 1)[-1]] += 1
            per_group[classify_module(name)] += 1
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    config = getattr(model, "peft_config", {}).get("default", None)
    declared = getattr(config, "target_modules", None)
    if isinstance(declared, (list, tuple, set)):
        declared = sorted(str(item) for item in declared)
    elif declared is not None:
        declared = str(declared)  # PEFT compiles a matching regex
    return {
        "declared_target_modules": declared,
        "adapted_module_total": len(adapted_names),
        "adapted_by_group": dict(sorted(per_group.items())),
        "adapted_by_leaf": dict(sorted(per_leaf.items())),
        "gdn_leaves_requested_but_missing":
            sorted(set(GDN_TARGETS) - set(per_leaf)),
        "trainable_param_count": int(sum(p.numel() for _, p in trainable)),
        "trainable_param_names_sample": sorted(n for n, _ in trainable)[:6],
        "examples": adapted_names[:4] + adapted_names[-2:],
    }


def trainable_named(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and "lora_" in n]


def snapshot_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().clone() for n, p in trainable_named(model)}


def snapshot_delta(before: dict[str, torch.Tensor],
                   model: torch.nn.Module) -> dict[str, float]:
    sq = 0.0
    changed = 0
    for name, param in trainable_named(model):
        delta = param.detach().float() - before[name]
        sq += float((delta * delta).sum())
        if float(delta.abs().max()) > 0:
            changed += 1
    return {"lora_delta_l2": math.sqrt(sq), "lora_tensors_changed": changed}


def render_example(tokenizer, example: dict[str, str], max_len: int) -> dict[str, list[int]]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["instruction"]}],
        tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["instruction"]},
         {"role": "assistant", "content": example["output"]}],
        tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False,
                          truncation=True, max_length=max_len)["input_ids"]
    labels = list(input_ids)
    mask_upto = min(len(prompt_ids), len(labels) - 1)
    labels[:mask_upto] = [-100] * mask_upto
    return {"input_ids": input_ids, "labels": labels}


def pad_batch(rendered: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(r["input_ids"]) for r in rendered)
    input_ids, attention, labels = [], [], []
    for row in rendered:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        attention.append([1] * len(row["input_ids"]) + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def check_gradients(model: torch.nn.Module) -> dict[str, Any]:
    total = 0
    finite = 0
    nonzero = 0
    sq = 0.0
    for _, param in trainable_named(model):
        if param.grad is None:
            continue
        total += 1
        grad = param.grad.detach().float()
        if bool(torch.isfinite(grad).all()):
            finite += 1
        if float(grad.abs().max()) > 0:
            nonzero += 1
        sq += float((grad * grad).sum())
    return {"grad_tensors": total, "grad_finite": finite, "grad_nonzero": nonzero,
            "grad_l2_norm": math.sqrt(sq)}


def run_training(model, raw_tokenizer, args: argparse.Namespace,
                 examples: list[dict[str, str]]) -> dict[str, Any]:
    trainable = trainable_named(model)
    optimizer = torch.optim.AdamW((p for _, p in trainable), lr=args.lr)
    pad_id = raw_tokenizer.pad_token_id or raw_tokenizer.eos_token_id
    rendered = [render_example(raw_tokenizer, ex, args.max_seq_len)
                for ex in examples]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    initial = snapshot_lora(model)
    step_records: list[dict[str, Any]] = []
    losses: list[float] = []

    for step in range(args.steps):
        offset = (step * args.batch_size) % len(rendered)
        batch_rows = rendered[offset:offset + args.batch_size]
        if len(batch_rows) < args.batch_size:
            batch_rows = batch_rows + rendered[:args.batch_size - len(batch_rows)]
        batch = pad_batch(batch_rows, pad_id)
        batch = {k: v.cuda() for k, v in batch.items()}

        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss_value = float(loss.detach())
        losses.append(loss_value)
        loss.backward()
        grad_stats = check_gradients(model)
        optimizer.step()
        torch.cuda.synchronize()
        step_records.append({
            "step": step + 1,
            "loss": loss_value,
            "loss_finite": bool(math.isfinite(loss_value)),
            "wall_s": round(time.time() - step_start, 3),
            **grad_stats,
        })
        print(f"step {step + 1}/{args.steps} loss={loss_value:.4f} "
              f"grad_l2={grad_stats['grad_l2_norm']:.4f} "
              f"finite={grad_stats['grad_finite']}/{grad_stats['grad_tensors']}")

    torch.cuda.synchronize()
    return {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "lr": args.lr,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "all_losses_finite": all(math.isfinite(v) for v in losses),
        "all_grads_finite": all(s["grad_finite"] == s["grad_tensors"]
                                for s in step_records if s["grad_tensors"]),
        "any_grad_nonzero": any(s["grad_nonzero"] > 0 for s in step_records),
        "adapter_delta": snapshot_delta(initial, model),
        "step_records": step_records,
        "wall_s_total": round(time.time() - started, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def write_metrics(metrics: dict[str, Any], path: str | None) -> None:
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(payload + "\n", encoding="utf-8")
    print(payload)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
    )
    # Unsloth hands back a Qwen3VLProcessor as `tokenizer`; calling it with a
    # positional string feeds the string to the image path. The inner text
    # tokenizer is the one that encodes prompts.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    metrics: dict[str, Any] = {
        "issue": 238,
        "stage": "g2_tiny_adapt",
        "model": args.model,
        "model_class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype),
        "tokenizer_class": type(tokenizer).__name__,
        "text_tokenizer_class": type(raw_tokenizer).__name__,
        "inventory": module_inventory(model),
    }

    model = attach_lora(model, args)
    metrics["lora_config"] = {"r": args.lora_r, "lora_alpha": args.lora_alpha,
                              "dropout": 0.0, "seed": args.seed}
    metrics["coverage"] = adapted_coverage(model)
    print("COVERAGE:", json.dumps(metrics["coverage"], indent=2))

    if args.derive_only:
        metrics["derive_only"] = True
        write_metrics(metrics, args.metrics_out)
        return 0

    examples = load_examples(Path(args.data),
                             args.steps * args.batch_size)
    metrics["data"] = {
        "path": args.data,
        "examples_used": len(examples),
        "benchmark_overlap_checked_separately": True,
        "first_instruction_preview": examples[0]["instruction"][:120],
    }
    metrics["training"] = run_training(model, raw_tokenizer, args, examples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    metrics["saved_adapter_to"] = str(out_dir)
    metrics["adapter_files"] = sorted(p.name for p in out_dir.iterdir())
    write_metrics(metrics, args.metrics_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
