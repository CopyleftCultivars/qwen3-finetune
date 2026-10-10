# Data Card

## Latest Retained Dataset

The latest retained training payload is the G6 Qwen3-8B trajectory SFT dataset.

| Split | File | Rows | SHA256 |
| --- | --- | ---: | --- |
| train | `data/g6_rlm_sft_train.jsonl` | 117 | `dd6896f5b5517b11abbfa0e8da83367ffe27d1f6d7b2d51e2a8b4dbbf3abadd8` |
| validation | `data/g6_rlm_sft_val.jsonl` | 14 | `a4481f188ecc7bbd320527ff10ddcdf2f6c6482951bef6762da6091e9eca1f77` |

Records use chat-style `messages`; the final assistant turn is the training
target. The G6 preparation and training details are in
`results/G6_LORA_SFT_QWEN3_8B.md`. Its intermediate G5 trajectory dump is not
retained in Git; the final train and validation splits are retained.

## Historical v5clean Sources

The v5clean champion split can be regenerated from these retained sources:

| Source | Rows | SHA256 | Role |
| --- | ---: | --- | --- |
| `data/sources/gemma4/hf_dataset/train.jsonl` | 3,998 | `ec4262b38d9a32d54e154e9a3df97f603e6e671ec105004793a26b68ae4f3e08` | Natural-farming conversation SFT source |
| `data/sources/qmd_mcq_all_phase3.jsonl` | 311 | `2f88dc7fd68d0e6a6df0530dd79d685961e36c2d620b1f339bc541d982f0ac1b` | Clean QMD-generated MCQ source, expanded with cyclic answer-position shifts |

Rebuild the historical split with:

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

Generated historical train/validation snapshots are ignored by Git.

## Evaluation Boundary

Sealed evaluation splits are not training data. The current free-form test split
is `eval_freeform/test.jsonl`; `eval_freeform/dev.jsonl` is for development.
The autoresearch program treats sealed tests as final-evaluation-only surfaces;
see `autoresearch/program.md` and `autoresearch/frozen_manifest.json`.

## Intended Use

The retained G6 pair is intended for supervised finetuning of the Qwen3-8B G6
trajectory adaptation. The historical v5clean split supports regeneration of
the earlier Qwen3-0.6B natural-farming champion data.

Do not mix validation or sealed evaluation splits into training.

## Sharing Notes

- Share the two latest split files above with this data card and the README
  training command.
- Source licensing and redistribution rights should be reviewed before public
  release. Internal coworker sharing through the git repo is a separate decision
  from publishing a public dataset.
- API keys belong in `.env` only and are not part of the dataset.
