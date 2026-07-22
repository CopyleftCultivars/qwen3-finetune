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
| Embedded system/persona filtering | **Pipeline ready; isolated run pending** | `--filter-system-prompts` strips known composite prefixes and drops persona-only rows. The combined p9 result cannot identify its effect. |
| Reviewed adversarial refusals | **Pipeline ready; isolated run pending** | `data/sources/adversarial_refusals.jsonl` contains ten approved, stable-ID examples. The combined p9 result cannot identify their effect. |
| Chat template / prompt variant | **Unlocked; isolated run pending** | `measure_periodic_guards.py` supplies the free-form rubric axis. `tokenizer-v1` is now an explicit training knob; deployment system-prompt A/B remains separate. |
| Curriculum order | **Pipeline ready; isolated run pending** | `length-structure-v1` provides a deterministic text-only ranking signal and audit report; it is explicitly a heuristic, not a ground-truth difficulty label. |

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
