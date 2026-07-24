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

## Phase 19 results

The three candidates were trained serially on the single CUDA device from the
same v5clean base and scored on the frozen English MCQ dev anchor, a fixed
256-example local Dolly validation sample, and the Spanish transfer dev
diagnostic. The 0% control is the already-scored v5clean champion.

| Condition | English primary | English argmax | General token-F1 | General exact | Spanish dev primary | Spanish argmax |
|---:|---:|---:|---:|---:|---:|---:|
| v5clean / 0% | -0.7752 | 72.6% | 0.2278 | 0.0% | -1.4337 | 52.3% |
| 5% | -1.9999 | 40.7% | 0.3225 | 2.0% | -1.5737 | 43.0% |
| 10% | -1.0717 | 59.3% | 0.3314 | 3.9% | **-0.9066** | **64.8%** |
| 20% | -1.4494 | 54.0% | **0.3567** | **5.1%** | -1.6632 | 41.4% |

The 10% dose is the best Spanish transfer-dev result, and the 20% dose is
the best local general-instruction result, but neither preserves the English
regression anchor. No ratio is promoted. The source remains provenance-tracked
and the generated ratio overlays remain ignored/reproducible; sealed English
and Spanish tests, Spanish free-form data, and external judging were not used.
