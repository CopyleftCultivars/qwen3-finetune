# Qwen3 Natural Farming Finetune

This repository contains data preparation, training, evaluation, and autoresearch
code for Qwen3 natural-farming models.

## Current Champion

The promoted Qwen3-0.6B checkpoint is **v5clean**:

- model artifact: `outputs/merged-model-p6-lr5e5-r2` (local output, not tracked)
- reproduction config: `autoresearch/configs/v5-clean-baseline.yaml`
- base model: `unsloth/Qwen3-0.6B` for training, `Qwen/Qwen3-0.6B` for merge

The latest retained training/validation pair is the G6 Qwen3-8B trajectory SFT
dataset: `data/g6_rlm_sft_train.jsonl` and `data/g6_rlm_sft_val.jsonl`. Training
and preparation details are in `results/G6_LORA_SFT_QWEN3_8B.md`.

| Split | Rows | SHA256 |
| --- | ---: | --- |
| `data/g6_rlm_sft_train.jsonl` | 117 | `dd6896f5b5517b11abbfa0e8da83367ffe27d1f6d7b2d51e2a8b4dbbf3abadd8` |
| `data/g6_rlm_sft_val.jsonl` | 14 | `a4481f188ecc7bbd320527ff10ddcdf2f6c6482951bef6762da6091e9eca1f77` |

The current sealed evaluation fixtures are `eval_freeform/dev.jsonl` and
`eval_freeform/test.jsonl`. Keep the sealed test split out of training.

## Setup

Use a CUDA-capable Linux host for training. Unsloth training will not run
usefully on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some local judge/authoring pipelines used OpenRouter. Copy `.env.example` only
when running those judge steps; normal training does not require an OpenRouter
key.

```bash
cp .env.example .env
```

Never commit `.env`.

## Train the Latest Retained Split

```bash
python train.py \
  --model unsloth/qwen3-8b-unsloth-bnb-4bit \
  --train-data data/g6_rlm_sft_train.jsonl \
  --val-data data/g6_rlm_sft_val.jsonl \
  --output-dir outputs/lora-adapter-g6-rlm-8b \
  --epochs 4 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.025 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max-seq-len 2560 \
  --seed 42
```

## Rebuild the Historical v5clean Split

The generated `_v3` training snapshots are excluded from Git. Rebuild them from
the retained source datasets when needed:

```bash
python prepare_dataset_v3.py \
  --conversations data/sources/gemma4/hf_dataset/train.jsonl \
  --qmd-mcq data/sources/qmd_mcq_all_phase3.jsonl \
  --output-dir data \
  --suffix _v3 \
  --split 0.9 \
  --seed 42 \
  --min-output-chars 10 \
  --max-total-chars 8192 \
  --sft-format bare-letter
```

The autoresearch reproduction path is:

```bash
python autoresearch/finetune.py \
  --config autoresearch/configs/v5-clean-baseline.yaml
```

Use `--dry-run` to print the planned subprocesses without starting training.

## Merge Adapter

```bash
python merge_adapter.py \
  --base Qwen/Qwen3-0.6B \
  --adapter outputs/lora-adapter-v5-clean \
  --output outputs/merged-model-v5-clean \
  --dtype float16
```
