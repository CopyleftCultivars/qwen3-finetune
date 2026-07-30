# Data Card

## Dataset

This repo's current share target is the **v5clean natural-farming SFT training
set** used by the promoted champion documented in `autoresearch/program.md`.

Recommended Hugging Face Datasets payload:

| Split | File | Rows | SHA256 |
| --- | --- | ---: | --- |
| train | `data/train_v3.jsonl` | 4,400 | `76ca9bec47065b5e06480b8e05e203efb44e6533293bcbcb02b8760bdcee022e` |
| validation | `data/val_v3.jsonl` | 489 | `2f63df7f7495a87bf1971b82d635e303432a380251b1e935310b6859a96df39f` |

Format: one JSON object per line. Records use either
`{"instruction": "...", "output": "..."}` or chat-style `messages` where the
last turn is the assistant target. The champion files use the instruction/output
shape expected by `train.py`.

## Sources

The v5clean split is rebuilt from:

| Source | Rows | SHA256 | Role |
| --- | ---: | --- | --- |
| `data/sources/gemma4/hf_dataset/train.jsonl` | 3,998 | `ec4262b38d9a32d54e154e9a3df97f603e6e671ec105004793a26b68ae4f3e08` | Natural-farming conversation SFT source |
| `data/sources/qmd_mcq_all_phase3.jsonl` | 311 | `2f88dc7fd68d0e6a6df0530dd79d685961e36c2d620b1f339bc541d982f0ac1b` | Clean QMD-generated MCQ source, expanded with cyclic answer-position shifts |

The preprocessing command is:

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

## Contamination And Evaluation Boundary

The QMD MCQ source was produced as the clean Phase 3 source and audited against
the English anchor benchmark before inclusion. The relevant audit artifacts are:

- `results/qmd_mcq_source_contamination_audit.json`
- `results/qmd_overlay_english_anchor_contamination_audit.json`
- `results/mcq_rationale_contamination_audit.json`

Sealed evaluation splits are not training data. The autoresearch program treats
sealed tests as final-evaluation-only surfaces; see `autoresearch/program.md` and
`autoresearch/frozen_manifest.json`.

## Intended Use

This dataset is intended for supervised finetuning and reproducibility of the
Qwen3-0.6B natural-farming champion experiments in this repository.

It is not a general benchmark. Do not mix validation or sealed evaluation splits
into training.

## Sharing Notes

- Publish the two split files above first; they are the exact current champion
  training payload.
- Include this data card and the README training command with any Hugging Face
  Datasets upload.
- Source licensing and redistribution rights should be reviewed before public
  release. Internal coworker sharing through the git repo is a separate decision
  from publishing a public dataset.
- Some historical local authoring and judge processes used OpenRouter; those
  keys belong in `.env` only and are not part of the dataset.
