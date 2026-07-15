# C2 Metric Sanity Check

Date: 2026-07-15

Status: **BLOCKED / failed acceptance, do not close #23 or start #24 from this result.**

## Objective Under Test

Frozen config: `autoresearch/objective_weights.json`

Formula:

```text
primary = 0.5 * mcq_mean_gold_logprob + 0.5 * freeform_mean_reference_logprob_per_token
```

Objective direction: maximize.

## Real Primary Run Status

The full primary objective could not be measured in this environment because the real
free-form ref-LL scorer cannot load models here:

```text
python eval_freeform/score_refll.py --model outputs/merged-model-v3-clean --input eval_freeform/dev.jsonl --output /tmp/freeform_refll_v3_real_attempt.json --limit 1
python eval_freeform/score_refll.py --model outputs/merged-model-v2-clean --input eval_freeform/dev.jsonl --output /tmp/freeform_refll_v2_real_attempt.json --limit 1
```

Both fail with:

```text
ModuleNotFoundError: No module named 'torch'
ImportError: Real scoring requires torch and transformers; use --mock-model for smoke tests
```

The full `score_primary.py` real path fails for the same reason when it invokes
`eval_freeform/score_refll.py`. In addition, the local `outputs/merged-model-v2-clean`
and `outputs/merged-model-v3-clean` directories currently show tokenizer/config files
but no obvious `*.safetensors` or `*.bin` weight files, so a reproducible real run also
needs the actual model weights present.

## Available Actual MCQ Component

Existing actual MCQ scorer artifacts:

- `results/mcq_gold_ll_v3_clean.json`
- `results/mcq_gold_ll_v2_clean.json`

| model | MCQ mean gold logprob | frozen 0.5 contribution | MCQ argmax accuracy |
| --- | ---: | ---: | ---: |
| v3-clean | -14.7378318584 | -7.3689159292 | 0.2920353982 |
| v2-clean | -8.9633551403 | -4.4816775702 | 0.1769911504 |

MCQ raw LL ranks **v2-clean above v3-clean**, opposite the intended sanity direction.
The v3-minus-v2 MCQ LL delta is:

```text
-5.7744767181 logprob/item
```

With the frozen equal weights, v3 starts `-2.8872383590` primary-score points behind
v2 before the free-form component is included. Therefore, for the full primary to rank
v3-clean above v2-clean, the free-form ref-LL delta would need to satisfy:

```text
freeform_ref_ll(v3) - freeform_ref_ll(v2) > 5.7744767181 logprob/token
```

That is a very large per-token LL gap. It is not impossible without measuring it, but
this threshold is high enough that the current frozen raw-MCQ-LL objective should be
treated as suspect until the real free-form component is run.

## Variance Check On Available Paired MCQ Data

Paired rows: 113 shared MCQ question IDs.

Analytic paired estimates:

| metric delta, v3 - v2 | mean | sample SD | SE | z |
| --- | ---: | ---: | ---: | ---: |
| MCQ gold LL delta | -5.7744767181 | 5.2312036274 | 0.4921102419 | -11.7341120472 |
| MCQ argmax accuracy delta | 0.1150442478 | 0.5469430042 | 0.0514520698 | 2.2359498514 |

Bootstrap over paired question IDs, 20,000 resamples, seed 20260715:

| metric delta, v3 - v2 | bootstrap SD | q05 | q50 | q95 | opposite-sign probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| MCQ gold LL delta | 0.4911454359 | -6.5859396605 | -5.7731631119 | -4.9694435322 | 0.0000 |
| MCQ argmax accuracy delta | 0.0510980980 | 0.0353982301 | 0.1150442478 | 0.2035398230 | 0.0143 |

The continuous MCQ LL component is visibly more sign-stable than MCQ accuracy on this
paired comparison, but it is confidently stable in the **wrong** direction for the
v3-clean > v2-clean acceptance check.

For reference, binomial SEs from the known real-generation accuracy context are:

```text
p=0.7300, n=1603: SE=0.0110885958
p=0.4255, n=1603: SE=0.0123488933
```

## Conclusion

#35 does **not** pass acceptance in this environment.

Reasons:

1. The full primary objective cannot be measured because real free-form ref-LL scoring
   lacks `torch`/`transformers` and likely local model weight files.
2. The available actual MCQ component of the frozen raw-LL objective ranks v2-clean
   above v3-clean by a large margin.
3. Although MCQ LL has lower sign variance than MCQ accuracy in the paired bootstrap,
   that lower variance reinforces the wrong ranking rather than validating the objective.

Flag for #23: do not let C3/C4 climb this frozen objective until the real free-form
ref-LL component is run and the raw-MCQ-LL design issue is resolved. Candidate fixes
remain the ones already noted on #23: option-set-normalized MCQ LL, a robust aggregate,
or a different frozen weighting/normalization after explicit human approval.

## Follow-up: #49 Partial Objective Fix

A follow-up change updates the frozen MCQ component from raw full-vocabulary
gold-token LL to option-set-normalized gold-token LL:

```text
mcq_metric = mean_option_set_gold_logprob
primary = 0.5 * mcq_mean_option_set_gold_logprob + 0.5 * freeform_mean_reference_logprob_per_token
```

Existing MCQ artifacts already contain per-item four-letter probabilities, so the new
combiner can compute the normalized aggregate from those historical outputs.

With the same deterministic mock free-form refLL file held constant for both models,
the smoke primary now ranks in the intended direction:

| model | MCQ option-set mean gold logprob | smoke primary with held-constant mock free-form |
| --- | ---: | ---: |
| v3-clean | -3.0205043064 | -2.8926873242 |
| v2-clean | -3.8423283268 | -3.3035993344 |

This resolves the MCQ raw-LL pathology but does **not** close #35 or #49. A real #35
rerun still needs actual free-form refLL on both models with `torch`/`transformers` and
the real merged model weight files present.
