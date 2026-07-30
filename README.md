# Qwen3 Natural Farming Finetune

This repository contains the data preparation, training, evaluation, and
autoresearch loop used for the Qwen3-0.6B natural-farming finetune.

## Current Champion

The current promoted checkpoint is **v5clean**:

- model artifact: `outputs/merged-model-p6-lr5e5-r2` (local output, not tracked)
- clean reproduction config: `autoresearch/configs/v5-clean-baseline.yaml`
- original promoted run config:
  `autoresearch/configs/phase6-post-v4clean-confirm/p6-lr5e5-r2.yaml`
- base model: `unsloth/Qwen3-0.6B` for training,
  `Qwen/Qwen3-0.6B` for merge
- exact checked-in training payload: `data/train_v3.jsonl`
- exact checked-in validation payload: `data/val_v3.jsonl`

`autoresearch/configs/v5-clean-baseline.yaml` is the clean-named reproduction
config. It rebuilds the same source mix under `_v5` filenames
(`data/train_v5.jsonl` / `data/val_v5.jsonl`) before training.

The canonical training split for sharing to Hugging Face Datasets is:

| File | Rows | SHA256 |
| --- | ---: | --- |
| `data/train_v3.jsonl` | 4,400 | `76ca9bec47065b5e06480b8e05e203efb44e6533293bcbcb02b8760bdcee022e` |
| `data/val_v3.jsonl` | 489 | `2f63df7f7495a87bf1971b82d635e303432a380251b1e935310b6859a96df39f` |

Those files are derived from:

| Source | Rows | SHA256 |
| --- | ---: | --- |
| `data/sources/gemma4/hf_dataset/train.jsonl` | 3,998 | `ec4262b38d9a32d54e154e9a3df97f603e6e671ec105004793a26b68ae4f3e08` |
| `data/sources/qmd_mcq_all_phase3.jsonl` | 311 | `2f88dc7fd68d0e6a6df0530dd79d685961e36c2d620b1f339bc541d982f0ac1b` |

See `DATA_CARD.md` for source, contamination, and sharing notes.

## Setup

Use a CUDA-capable Linux host for training. Unsloth training will not run
usefully on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some local judge/authoring pipelines used OpenRouter. Copy `.env.example` only
when you are running those judge steps; champion reproduction and normal
training do not require an OpenRouter key.

```bash
cp .env.example .env
```

Never commit `.env`.

## Rebuild Champion Data

The checked-in champion data is already present as `data/train_v3.jsonl` and
`data/val_v3.jsonl`. To rebuild it from source:

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

## Train Exact Checked-In Split

Use this command when you want to train from the exact data files listed in the
data card:

```bash
python train.py \
  --model unsloth/Qwen3-0.6B \
  --train-data data/train_v3.jsonl \
  --val-data data/val_v3.jsonl \
  --output-dir outputs/lora-adapter-v5-clean \
  --epochs 4 \
  --lr 5e-5 \
  --lr-scheduler cosine \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --batch-size 4 \
  --grad-accum 4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.025 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max-seq-len 2048 \
  --seed 42
```

## Train Through Autoresearch Harness

The clean reproduction path is through the v5 config:

```bash
python autoresearch/finetune.py \
  --config autoresearch/configs/v5-clean-baseline.yaml
```

That command first rebuilds `data/train_v5.jsonl` and `data/val_v5.jsonl`, then
trains and merges the adapter. Use `--dry-run` to print the planned subprocesses
without starting training.

## Merge Adapter

```bash
python merge_adapter.py \
  --base Qwen/Qwen3-0.6B \
  --adapter outputs/lora-adapter-v5-clean \
  --output outputs/merged-model-v5-clean \
  --dtype float16
```

## Tests

The lightweight CPU tests cover data builders and guard helpers:

```bash
pytest
```

Full training and GPU scoring are intentionally not part of the CPU test suite.

## Repository Hygiene

- `outputs/`, model weights, W&B runs, local caches, and `.env` files are
  intentionally ignored.
- Share this repository through git, not by zipping the working directory.
- Publish selected datasets/checkpoints separately with checksums.
