# CROP-style auxiliary-task lever protocol

The repository now treats the four auxiliary tasks as four separate
autoresearch dimensions:

1. summarization
2. NER/entity extraction
3. structured extraction
4. domain open-book QA

The task records are deterministic templates over the natural-farming source
files under `data/sources/gemma4/spaces/data/`. The generic CROP dataset is not
ingested. The generated streams carry English language labels, source IDs,
source-file hashes, record hashes, authoring method, and contamination status.

## Build and audit

The fixed pilot uses 256 train rows and 64 validation rows per task. Splitting
happens by source document before capping, so a source ID cannot appear in both
parts. Each task is audited against the open English MCQ benchmark for exact or
high lexical text overlap. The audit is a question/option-text boundary check;
it does not claim semantic or source-document independence.

```bash
python scripts/build_crop_auxiliary_datasets.py \
  --benchmark [LOCAL_PATH] \
  --output-root data/auxiliary \
  --train-count 256 \
  --validation-count 64 \
  --split 0.8 \
  --seed 42
```

The four task manifests under `data/auxiliary/` are the auditable inputs. The
larger composed overlays under `data/auxiliary_levers/` are intentionally
ignored: `autoresearch/finetune.py` recreates one of them immediately before a
run from the strict v5clean base and the selected task stream.

## One-task-at-a-time evaluation

Run the configs serially because the machine has one GPU:

```bash
python autoresearch/finetune.py --config autoresearch/configs/phase17-crop-auxiliary/p17-aux-summarization.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase17-crop-auxiliary/p17-aux-ner.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase17-crop-auxiliary/p17-aux-structured-extraction.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase17-crop-auxiliary/p17-aux-open-book-qa.yaml
```

Every config preserves v5clean's settled training settings and changes only the
selected auxiliary stream. For each resulting model, record:

- the frozen MCQ primary and all existing guardrails;
- English free-form dev quality;
- task-native auxiliary validation quality, with exact-match/JSON validity for
  NER and structured extraction and grounded-answer quality for open-book QA;
- the source and composition manifest hashes.

Task-native scoring is run separately with `scripts/score_auxiliary_validation.py`.
For example, after the summarization run:

```bash
python scripts/score_auxiliary_validation.py \
  --model outputs/merged-model-p17-aux-summarization \
  --dataset data/auxiliary/summarization/validation.jsonl \
  --task summarization \
  --output results/auxiliary_validation_p17-aux-summarization.json
```

For NER and structured extraction, JSON validity and exact canonical JSON are
reported in addition to token F1. Open-book QA and summarization report token F1
and exact normalized-text rate. Score the v5clean model on the same validation
files first; these are auxiliary diagnostics, not replacements for the frozen
MCQ objective or free-form guardrails.

Do not combine auxiliary tasks until each single-task result is recorded. A
combination is a new lever and must use a controlled dose, with the same
contamination boundary and a fresh manifest.
