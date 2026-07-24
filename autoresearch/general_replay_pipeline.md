# Phase 19: general-instruction replay

This phase tests one new data lever only: adding a clean English general-
instruction source to the canonical v5clean training pool. The QMD/new-MCQ
authoring track is intentionally excluded.

## Source boundary

The source is the pinned English Databricks Dolly 15k dataset
(`databricks/databricks-dolly-15k`, revision `bdd27f4`, CC BY-SA 3.0). It is
materialized under `data/sources/general_instruction/dolly15k/` with category,
source ID, dataset revision, license, and split metadata.

The audited source contains 15,011 records: 13,510 train and 1,501
validation. The contamination audit compares the instruction/context surface
and known answer fields against the full English benchmark and records the
v5clean-base prompt overlap separately. It found zero benchmark rejections and
zero train/validation source overlap. The two generic `Tell me a joke` prompt
matches against v5clean are retained because their answers differ and they are
not benchmark leakage.

The source is training-eligible only through
`data/sources/general_instruction/dolly15k/audited/manifest.json`. Raw source
downloads are reproducible from the pinned source manifest and are not used
directly by training.

## Ratio construction

The composer retains the complete v5clean base and adds general-instruction
rows until the selected general token budget reaches the requested fraction of
the v5clean base train token budget. The same deterministic source-ID order is
used for every ratio, and the held-out general source split is kept separate.

| Ratio | General train tokens | General train rows | Composed train rows |
|---:|---:|---:|---:|
| 0% control | 0 | 0 | 4,396 |
| 5% | 90,566 | 480 | 4,876 |
| 10% | 181,135 | 988 | 5,384 |
| 20% | 362,269 | 1,978 | 6,374 |

The 0% condition is the existing v5clean control; it does not require a
redundant GPU retraining run.

## Execution and gates

Run the 5%, 10%, and 20% configs serially on the one GPU:

```bash
python autoresearch/finetune.py --config autoresearch/configs/phase19-general-replay/p19-general-ratio005.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase19-general-replay/p19-general-ratio010.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase19-general-replay/p19-general-ratio020.yaml
```

For each merged candidate, score the frozen English MCQ primary first, then
the fixed local general-instruction validation sample and Spanish MCQ debug
diagnostic. The English primary, answer-position bias, forgetting, compute,
and contamination guardrails remain accept/revert gates. The sealed English
and Spanish tests are not read, and no external free-form judge is part of
this phase.

