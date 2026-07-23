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
