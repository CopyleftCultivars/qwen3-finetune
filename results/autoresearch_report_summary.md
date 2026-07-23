# Autoresearch report summary

This is the compact decision log for the current v5clean champion and the
post-Tier-1/Tier-2 data-pipeline work. The frozen primary remains the option-set
gold log-likelihood gate; free-form rubric results are a secondary quality
guard, not a replacement for that gate.

## Current decisions

| Lever | Isolated status | Evidence |
|---|---|---|
| MCQ versus cloze | **Search rejected at the tested 400-item dose** | `results/autoresearch_gate_p8-cloze400.json`: primary delta `-0.316` (`-1.58 SE`), compute 28.35 vs 28.00 minutes, bias `0.1121` vs `0.1097` allowed, and rubric `60.80% → 59.37%`; forgetting improved `41.67% → 45.83%`. The result is causal for this dose, not for all cloze designs. |
| Deployment system prompt | **A/B rejected for the current MCQ interface** | `results/system_prompt_compare_p9-clean-cloze400-refusal.json`: baseline argmax accuracy fell 73.28% → 55.17%; candidate fell 49.14% → 39.66%. Option-set gold log-likelihood also fell in both comparisons. |
| Embedded system/persona filtering | **Search rejected — isolated filter-only candidate** | `results/autoresearch_gate_p10-filter-only.json`: primary delta `+0.0586` (`+0.29 SE`, below the `+0.4` threshold), bias `0.1173` vs `0.1097` allowed, rubric `60.80% → 60.68%`, and forgetting improved `41.67% → 45.83%`. Filtering is now causally tested but is not a champion candidate. |
| Reviewed adversarial refusals | **Search rejected — isolated refusal-only candidate** | `results/autoresearch_gate_p10-refusal-only.json`: primary delta `+0.0497` (`+0.25 SE`, below threshold), rubric `60.80% → 59.13%`, forgetting improved `41.67% → 45.83%`, and the remaining guards were clean. Ten reviewed examples did not establish a champion benefit. |
| Chat template / prompt variant | **Search rejected — tokenizer-v1** | `results/autoresearch_gate_p10-chat-template-tokenizer.json`: primary delta `-1.213` (`-6.06 SE`), bias `0.3164`, forgetting drop 10%, rubric `60.80% → 49.81%`, and compute 28.18 vs 28.00 minutes. The free-form axis successfully exposed the severe quality regression; keep hardcoded-v1 as the current training interface. |
| Curriculum order | **Search rejected — three-repeat confirmation** | The fixed-data confirmation campaign produced primary deltas `+0.199` (seed 101), `+0.338` (seed 102), and `+0.340` (seed 103), all below the `+0.4` gate. All three kept the same 4,400 ordered training items and 489 validation items; hard guards stayed clean, forgetting improved to 62.5–66.67%, and rubric ranged 57.39–59.87%. The original +0.399 near miss was not robust enough to promote. |
| Rationale targets | **Search rejected — two category/dose runs; replay strategy closed** | The combined p11 overlay produced primary delta `-0.0732` (`-0.366 SE`) and bias `0.1645`; the natural-fertilizers-only p12 overlay produced delta `-0.0597` (`-0.298 SE`) and bias `0.1469`, both against v5clean and both below the `+0.4` gate. P12 kept rubric within guard (`60.80% → 60.53%`), forgetting unchanged at `41.67%`, and anchors clean, but still failed bias. All 311 clean rationale outputs exactly match the existing QMD outputs by `_source_question_id`, so this is metadata-clean replay rather than novel reasoning supervision. v5clean remains champion; stop widening this replay strategy. |

## Tier 2 completion campaign

The three remaining Tier-2 search dimensions were each given one isolated dose-refinement run against v5clean. No candidate cleared the frozen `+0.4` primary gate, and v5clean remains champion.

| Dimension | Candidate | Primary delta | Guard summary | Decision |
|---|---|---:|---|---|
| Answer-position augmentation | multiplier 3 on the 17 joined position-unstable QMD records | `-0.1461` (`-0.730 SE`) | Rubric `60.80% → 56.56%`; compute `27.10 min`; bias, forgetting, anchors, sealed clean | **reject** |
| Conversation/MCQ replay ratio | conversation fraction `0.55` | `+0.3121` (`+1.560 SE`) | Rubric drop `1.87 pp`; compute `15.46 min`; forgetting improved to `54.17%`; bias, anchors, sealed clean | **reject** |
| Synth-item count | 233 raw QMD items before cyclic expansion | `+0.2023` (`+1.011 SE`) | Rubric `60.80% → 59.37%`; compute `26.52 min`; forgetting improved to `58.33%`; bias, anchors, sealed clean | **reject** |

The position result closes the tested multiplier-2/3 targeted replay surface. The ratio result closes the current-source 0.40/0.50/0.55 dose bracket. The synth result closes the executable 155/233 down-dose search; moving above the 311-item source ceiling requires a separate MCQ authoring and contamination-audit pipeline. Distractor quality remains outside the training loop because it belongs to item authoring and review.

The initial position attempt was also recorded as an infrastructure failed-run when the loop was invoked with `/usr/bin/python` without `unsloth`; the scientific candidate was rerun unchanged in the `qwen3-train` CUDA environment before evaluation.

## Tier 3 rationale-target unlock

The first Tier 3 candidate was trained and rejected by the normal gate. The
clean rationale source is packaged at
`data/sources/qmd_mcq_rationale_clean.jsonl` from the existing QMD rationale
pool and audited against
`[LOCAL_PATH]`.
The audit covers 311 rows, removes 16 embedded persona prefixes, reports zero
exact question collisions and zero high-lexical-overlap collisions, and emits
`_contamination_status=passed` provenance required by `prepare_dataset_v3.py`.

The candidate is J-space-guided by the v5clean `late_weak` bucket, restricted
to `natural_fertilizers` and `soil_science_and_microbiology`.  It retains the
complete v5clean conversation/QMD pool and adds four cyclic answer-position
variants per selected rationale row, preserving A/B/C/D balance.  The source
and overlay are opt-in; baseline configs do not change. The run produced 5,281
pairs (4,752 train / 529 validation), including 392 overlay rows from 98
selected late-weak source IDs; the candidate MCQ targets were exactly
A/B/C/D-balanced. On the candidate validation file, the frozen primary moved
from `-0.77515` to `-0.84836` (`-0.07321`, `-0.366 SE`), while the model's
predicted-letter bias guard failed. The Codex-graded periodic rubric moved
from `60.80%` to `58.04%`; forgetting improved to `50.00%` accuracy and
anchor drift remained zero. The candidate recipe is
`autoresearch/configs/phase11-tier3/p11-rationale-overlay.yaml`. Do not promote
or repeat this exact overlay dose without a new rationale/bias hypothesis.

## Tier 3 rationale follow-up

The natural-fertilizers-only p12 ablation tested the smallest targeted follow-up
after p11. It selected 43 `late_weak` source records, added 172 rationale
overlay rows, and produced 5,061 post-filter pairs (4,554 train / 507
validation) with exact A/B/C/D balance. The frozen primary moved from
`-0.77515` to `-0.83482` (`-0.05966`, `-0.298 SE`), and the gate rejected it
for both missing the `+0.4` threshold and reintroducing bias (`0.1469` vs
`0.1097` allowed). Rubric was nearly flat (`60.80% → 60.53%`, `-0.27`
points), forgetting stayed at `41.67%`, anchor drift was zero, and the sealed
test was untouched. Training took `1,680.0 s` (~28.00 minutes), so it was
compute-borderline but not a substantive runtime failure.

The source postmortem is decisive for this lever shape: all 311 rationale
outputs in `data/sources/qmd_mcq_rationale_clean.jsonl` exactly match the
corresponding outputs in `data/sources/qmd_mcq_all_phase3.jsonl`. The source is
contamination-audited and safely opt-in, but it does not add independent
reasoning supervision. After p11 and p12 both showed negative/flat primary
effects with recurring bias, close the current rationale-replay strategy. A
future rationale experiment must author genuinely new rationales or use a new
target construction; do not widen this overlay unchanged.

## Tier 3 P13 novel-supervision candidate

P13 changed the target construction rather than widening the rejected replay
strategy. A local CUDA Qwen authoring pass produced 43 natural-fertilizers
step-back rationales, preserving the source answer letters while requiring a
general agronomic principle, application to the question, and distractor
distinction. The authoring audit reports zero exact output matches and zero
normalized rationale matches against the existing QMD outputs. The separate
contamination audit reports zero exact benchmark-question overlaps and zero
high-lexical-overlap collisions; no sealed-test path was read.

The opt-in source was assembled as 4,409 conversation pairs, 1,555
cyclic-expanded QMD pairs, and 172 four-way rationale-overlay pairs, for 6,136
pairs before filtering and 5,061 after filtering (4,554 train / 507
validation). The final MCQ pool was exactly A/B/C/D-balanced at 348 each.

The CUDA run completed 1,140 steps in 1,782.6 seconds (~29.70 minutes), then
merged successfully. Against v5clean, the candidate primary moved from
`-0.77515` to `-0.64442` (`+0.13073`, `+0.654 SE`), below the `+0.4` gate.
The gate rejected it because bias reappeared (`0.1202` vs `0.1097` allowed)
and the measured training runtime exceeded the 28-minute budget. The paired
75-item Codex rubric was essentially flat (`60.80% → 60.77%`), forgetting
improved (`41.67% → 45.83%`), anchor drift was zero, and the sealed test was
untouched. P13 therefore does not revive the rationale lever or the v5clean
champion; leave future work to a materially different authoring strategy or
dose rather than repeating this source.

## Confounded p9 reference

The combined `p9-clean-cloze400-refusal` candidate had primary delta
`-0.8899` (`-4.45 SE`), reintroduced answer-letter bias, and exceeded the
compute budget. Its rubric improved by 1.25 points and forgetting was
unchanged. This rejects the combination, not each component. The individual
follow-up configs are under `autoresearch/configs/phase10-tier2/`.

## System-prompt note

The deployment prompt is kept in
`autoresearch/system_prompts/natural_farming.txt`, outside training JSONL.
The A/B diagnostic uses the same validation records and the same model with
and without that system message. Since it degraded both baseline and candidate
on the current MCQ interface, it is not part of the default evaluation or
training path. A future free-form system-conditioned experiment would need its
own explicit hypothesis and prompt lineage.

## Pipeline and evaluation policy

Each phase-10 candidate changes one lever against v5clean. No candidate reads
`eval_freeform/test.jsonl`. A candidate is reportable only after the frozen
primary, forgetting, letter-bias, anchor, compute, and free-form guard checks
are available. A pass would require confirmation before champion promotion.

## Curriculum confirmation campaign

The confirmation repeats intentionally did not expose a possible curriculum
candidate to more data. Every repeat used the same seed-42 data split, the same
4,400-item ordered training set (3,293 conversations, 1,051 QMD MCQs, and 56
wrapped MCQs), and the same 489-item validation set. Only the training seed
changed:

| Training seed | Primary delta | Rubric | Forgetting | Decision |
|---:|---:|---:|---:|---|
| 101 | +0.199 (+0.995 SE) | 58.55% | 66.67% | reject |
| 102 | +0.338 (+1.692 SE) | 57.39% | 62.50% | reject |
| 103 | +0.340 (+1.702 SE) | 59.87% | 66.67% | reject |

This closes the current curriculum-order heuristic/dose for Tier 2. A future
curriculum hypothesis may change the difficulty signal or author a richer
training pool, but it should be a new lever rather than silently widening this
confirmation.
