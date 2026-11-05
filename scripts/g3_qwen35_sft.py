#!/usr/bin/env python3
"""#239 G3: full bf16 LoRA SFT of Qwen3.5-2B on the F6/F8 distillation corpus.

Frozen recipe (results/QWEN3_5_G3_FREEZE.json): F7 hyperparameters verbatim,
only the base model changes (Qwen/Qwen3.5-2B, bf16, no 4-bit per Unsloth's
Qwen3.5 guidance) and the LoRA leaf set expands to the G2-derived Qwen3.5
inventory (18 GDN blocks + 6 full-attention blocks + MLP). Training text
rendering is F7-identical hardcoded-v1 ChatML so the base model is the only
lever that moves.

Run with the isolated transformers-5.5 venv:
  localllm-qwen3.5/retry-evidence/q1-isolated/venv/bin/python \
    scripts/g3_qwen35_sft.py

Writes the adapter to outputs/lora-adapter-g3-qwen35-distill and metrics to
outputs/g3_sft_metrics.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

REPO = Path(__file__).resolve().parent.parent

# F7 hyperparameters, frozen.
LR = 5e-5
EPOCHS = 4
# Micro-batch 2 x grad-accum 8 = effective 16, identical to F7's 4x4.
# Halved after the 4x4 run OOMed at step 45 (12 GB card, vocab-248320
# logits materialization under torch inductor). Effective batch, and hence
# optimizer-step count (1612), is unchanged; recorded as a deviation.
BATCH = 2
GRAD_ACCUM = 8
WARMUP = 0.05
WD = 0.01
MAX_SEQ_LEN = 2048
SEED = 42
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.025
# G2-derived Qwen3.5 leaf set: GDN projections + full-attention + MLP.
TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
           "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]

TRAIN_DATA = REPO / "data/train_distill.jsonl"
VAL_DATA = REPO / "data/val_distill.jsonl"
OUT_DIR = REPO / "outputs/lora-adapter-g3-qwen35-distill"


def render_hardcoded(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"<|im_start|>{t['role']}\n{t['content'].strip()}<|im_end|>" for t in messages
    )


def main() -> int:
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel

    t0 = time.time()
    print(f"train rows: {sum(1 for _ in open(TRAIN_DATA))}", flush=True)
    print(f"val rows:   {sum(1 for _ in open(VAL_DATA))}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-2B",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
    )
    raw_tok = getattr(tokenizer, "tokenizer", tokenizer)
    print(f"model={type(model).__name__} proc={type(tokenizer).__name__} "
          f"tok={type(raw_tok).__name__} dtype={next(model.parameters()).dtype}",
          flush=True)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable}", flush=True)

    try:
        raw_tok.eos_token = "<|im_end|>"
        raw_tok.eos_token_id = raw_tok.convert_tokens_to_ids("<|im_end|>")
    except Exception as exc:  # noqa: BLE001 -- record, do not crash
        print(f"eos override skipped: {exc}", flush=True)

    def load_pairs(path: Path) -> list[dict]:
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def to_text(pair: dict) -> str:
        if "messages" in pair:
            return render_hardcoded(pair["messages"])
        return render_hardcoded([
            {"role": "user", "content": pair["instruction"].strip()},
            {"role": "assistant", "content": pair["output"].strip()},
        ])

    train_pairs = load_pairs(TRAIN_DATA)
    val_pairs = load_pairs(VAL_DATA)

    def build(pairs: list[dict]) -> Dataset:
        texts = [to_text(p) for p in pairs]
        ds = Dataset.from_dict({"text": texts})

        def tok(batch: dict) -> dict:
            return raw_tok(batch["text"], truncation=True,
                           max_length=MAX_SEQ_LEN, padding=False)

        return ds.map(tok, batched=True, remove_columns=["text"])

    train_ds = build(train_pairs)
    val_ds = build(val_pairs)
    print(f"tokenized train={len(train_ds)} val={len(val_ds)}", flush=True)

    from trl import SFTConfig, SFTTrainer

    def find_latest_checkpoint(outdir: Path) -> str | None:
        ckpts = sorted(outdir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        return str(ckpts[-1]) if ckpts else None

    args = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP,
        learning_rate=LR,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=WD,
        seed=SEED,
        report_to="none",
        run_name="g3-qwen35-distill",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=raw_tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
    )
    torch.cuda.reset_peak_memory_stats()
    resume_from = find_latest_checkpoint(OUT_DIR)
    if resume_from:
        print(f"resuming from {resume_from}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    torch.cuda.synchronize()
    wall_s = time.time() - t0
    peak_mib = torch.cuda.max_memory_allocated() / 2**20

    final_dir = OUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    log = trainer.state.log_history
    losses = [e["loss"] for e in log if "loss" in e]
    metrics = {
        "issue": 239,
        "stage": "g3_sft",
        "base": "Qwen/Qwen3.5-2B",
        "trainable_params": trainable,
        "train_rows": len(train_pairs),
        "val_rows": len(val_pairs),
        "optimizer_steps": trainer.state.global_step,
        "max_steps_expected": 4 * -(-len(train_pairs) // (BATCH * GRAD_ACCUM)),
        "wall_s": round(wall_s, 1),
        "wall_min": round(wall_s / 60, 1),
        "peak_vram_mib": round(peak_mib, 1),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_curve": [[e.get("step"), e["loss"]] for e in log if "loss" in e],
        "train_result": str(train_result),
    }
    out = REPO / "outputs/g3_sft_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_curve"},
                     indent=2), flush=True)
    print(f"wrote {out}; adapter -> {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
