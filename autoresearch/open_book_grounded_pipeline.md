# Phase 22: context-grounded open-book training

Phase 17 already tested a generic domain `open_book_qa` auxiliary stream. It
learned its held-out format but regressed the closed-book MCQ anchor, so Phase
22 tests a materially different target rather than another dose of the same
overlay.

Each Phase 22 record contains three English natural-farming reference records:
one answer-bearing record and two same-domain hard negatives. The target has
two required lines:

```text
Answer: <answer>
Evidence: <SOURCE_ID>
```

The model must therefore identify the answer-bearing record and copy its
provenance ID. Context-present quality is scored separately from the frozen
closed-book MCQ primary.

## Source and contamination boundary

`scripts/build_open_book_grounded_source.py` reads only the open structured
natural-farming corpus under `data/sources/gemma4/spaces/data/`. It does not
read the generic CROP dataset, any Spanish benchmark path, or the sealed
free-form test. Source documents are assigned to train or validation before
selection and negative-context construction; every context record remains
within its partition. The manifest records source-file and record hashes.

The audit compares the instruction/context surface with the open English MCQ
anchor (`../benchmark-eval-workspace/benchmark_combined.jsonl`). It is a question/option lexical audit, not a
claim of semantic independence from all benchmark source documents.

```bash
python scripts/build_open_book_grounded_source.py \
  --source-root data/sources/gemma4/spaces/data \
  --benchmark [LOCAL_PATH] \
  --output-root data/sources/open_book_grounded \
  --train-count 256 \
  --validation-count 64 \
  --seed 42
```

## Training and evaluation

The config preserves v5clean's model, LoRA, optimizer, schedule, batch, and
epoch settings. `autoresearch/finetune.py` composes the audited source with
the strict v5clean base and treats the grounded stream as a mutually exclusive
lever. Run the one-GPU training job serially:

```bash
python autoresearch/finetune.py \
  --config autoresearch/configs/phase22-open-book-grounded/p22-open-book-grounded.yaml
```

Score the open-book dev stream on both v5clean and the candidate with:

```bash
python scripts/score_open_book_grounded.py \
  --model outputs/merged-model-p22-open-book-grounded \
  --dataset data/sources/open_book_grounded/validation.jsonl \
  --output results/open_book_grounded_validation_p22-open-book-grounded.json
```

The promotion decision remains the frozen English MCQ primary plus existing
free-form/forgetting/bias/anchor guards. A native grounded-task improvement
does not compensate for a closed-book regression. No Spanish benchmark file is
an input to this phase.
