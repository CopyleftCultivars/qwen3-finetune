# C2 Metric Sanity Check

Date: 2026-07-15

Policy update, 2026-07-16 (#51): `autoresearch/objective_weights.json` now uses
`autoresearch_primary_objective_policy_v2`. The first C3 climbable primary is **MCQ
option-set LL only**:

```text
primary = 1.0 * mcq_mean_option_set_gold_logprob
```

Free-form ref-LL is **not** a cross-approach comparator and is excluded from the first
C3 primary (`freeform_ref_ll_weight = 0.0`). It may be reported only as a labelled
diagnostic / same-lineage proxy after chat-template scoring and calibration. Cross-
approach quality sanity must come from generation accuracy and/or the rubric guard, not
from raw v2-vs-v3 ref-LL.

Latest status after the #53 rerun: **PASS under the revised C2 criterion.** The C3-
climbable primary is MCQ option-set LL only and ranks v3-clean above v2-clean. Chat-
template free-form ref-LL still ranks v2-clean above v3-clean, so it is removed from
the primary and kept only as a labelled diagnostic / same-lineage proxy candidate.

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

---

# Rerun: 2026-07-16 (real free-form component)

Status: **STILL BLOCKED, new root cause.** The primary now ranks v3 > v2, but only by
accident of two large opposing components, and the free-form component is confidently
mis-signed because the scorer does not apply the models' chat template.

## The 2026-07-15 environment blocker was spurious

Both blocking claims above were environment artifacts, not facts about the repo:

| 2026-07-15 claim | Verified 2026-07-16 |
| --- | --- |
| `ModuleNotFoundError: No module named 'torch'` | `qwen3-diagnostics` env has torch 2.6.0+cu124, transformers 4.57.6 |
| "no obvious `*.safetensors` or `*.bin` weight files" | both merged models have `model.safetensors`, 1.2 GB each |

The original run used an env without torch. Per repo convention, all diagnostics and
autoresearch eval must run under `conda run -n qwen3-diagnostics`. The real scorer runs
fine there (`mock_model=False`), on an RTX 4070 Ti.

## Real free-form ref-LL (90 dev items, 9,799 reference tokens, identical tokenization)

```text
conda run -n qwen3-diagnostics python eval_freeform/score_refll.py \
  --model outputs/merged-model-{v3,v2}-clean --input eval_freeform/dev.jsonl \
  --output results/freeform_refll_{v3,v2}-clean.json --device cuda
```

| model | mean reference logprob / token |
| --- | ---: |
| v3-clean | -4.9926148700 |
| v2-clean | -4.3704151200 |

**The free-form axis ranks v2-clean above v3-clean** — the opposite of ground truth
(v3-clean 73.0% vs v2-clean 42.55% generation accuracy).

Paired over all 90 items, delta = v3 - v2:

| statistic | value |
| --- | ---: |
| mean per-item delta | -0.6369166600 |
| sample SD | 0.4271940400 |
| SE | 0.0450302100 |
| z | -14.1442090000 |
| bootstrap SD (20k resamples, seed 20260716) | 0.0452141500 |
| bootstrap q05 / q50 / q95 | -0.7117784200 / -0.6367938600 / -0.5629117400 |
| P(v3 better than v2) | 0.0000 |
| items where v3 > v2 | **1 / 90 (1.1%)** |

Every item type favors v2, worst on the two reasoning-heaviest types:

| type | n | mean delta | v3 wins |
| --- | ---: | ---: | ---: |
| multi_hop_factual | 18 | -1.108591 | 0/18 |
| diagnostic_reasoning | 18 | -1.061469 | 0/18 |
| free_form_explanation | 18 | -0.425460 | 0/18 |
| quantitative | 18 | -0.392061 | 0/18 |
| constraint_satisfaction | 18 | -0.197002 | 1/18 |

## The primary passes, but for the wrong reason

| model | MCQ option-set LL | free-form ref-LL | **primary** |
| --- | ---: | ---: | ---: |
| v3-clean | -3.0205043100 | -4.9926148700 | **-4.0065595900** |
| v2-clean | -3.8423283300 | -4.3704151200 | **-4.1063717300** |

```text
margin (v3 - v2)         = +0.0998121300
  mcq component delta    = +0.4109120100  (favors v3)
  freeform component delta = -0.3110998800  (favors v2)
```

The intended ranking survives only because the MCQ term outweighs the free-form term.
The Goodhart guard is not guarding — it is actively pulling toward the worse model.

## Root cause: score_refll.py never applies the chat template

`eval_freeform/score_refll.py::real_model_score` scores a bare prompt:

```python
PROMPT_TEMPLATE = "Question:\n{question}\n\nReference answer:\n"
prompt = PROMPT_TEMPLATE.format(question=row["question"])
prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)...
```

There is no `apply_chat_template` call anywhere in the file, and
`add_special_tokens=False` strips the rest. Both merged models ship a
`chat_template.jinja` (Qwen3 `<|im_start|>`/`<|im_end|>` + `<think>`), and v3-clean was
SFT'd hard onto that template with `<think>...\n</think>\n{letter}` bare-letter targets.

So this scorer measures **how close a model still is to base prose-continuation
behavior**, which anti-correlates with exactly the MCQ-format conditioning that makes
v3-clean better. v2-clean scores higher because it is *less* finetuned, not because it
knows more. The compounding-over-tokens pattern (worst on the longest, most reasoning-
heavy items) is consistent with a format/OOD mismatch rather than a knowledge gap.

This is a scorer defect, not a v3-clean regression.

## Consequence for C3/C4

Do **not** let the accept/revert gate climb this objective. As measured, the free-form
term rewards *undoing* finetuning, and a loop would discover that within a night.

Note also that `score_refll.py`, `score_rubric.py`, `score_primary.py`, and
`score_mcq_ll.py` are **not** in `frozen_manifest.json` — only their configs, splits,
and anchors are. C1's freeze does not currently cover the scorer code itself.

## Candidate fixes (need human decision, all touch a frozen-adjacent surface)

1. Apply each model's chat template in `score_refll.py` so the reference answer is
   scored as an assistant turn (add `--chat-template/--no-chat-template`, pin the choice
   in `scorer_config.json`). Most likely correct; changes all historical refLL numbers.
2. Adjudicate with the rubric grader (`score_rubric.py`) on the same 90 dev items. If
   v3 wins on rubric while losing on ref-LL, that confirms ref-LL is the broken proxy —
   the inverse of A3's stated "LL-up-while-rubric-down = surface-learning alarm". Costs
   OpenRouter/Haiku calls.
3. Add the scorer code to the frozen manifest once its prompt shape is settled.

Sealed test set (`eval_freeform/test.jsonl`, 60 items) was **not** touched by this rerun.

---

# Rerun: 2026-07-16 after #50/#51/#52

Status: **PASS for #35 under the revised objective policy.**

Environment caveat: this workspace exposes no CUDA device (`torch.cuda.is_available()`
is false), so the real rerun used `conda run -n qwen3-diagnostics` with `--device cpu`.
The scorer path is otherwise the same transformer path (`mock_model=false`), and output
JSON records the device, prompt mode, and chat-template metadata.

## Commands run

```text
conda run -n qwen3-diagnostics python eval_freeform/score_refll.py \
  --model outputs/merged-model-v3-clean \
  --input eval_freeform/dev.jsonl \
  --output results/freeform_refll_v3-clean.json \
  --device cpu

conda run -n qwen3-diagnostics python eval_freeform/score_refll.py \
  --model outputs/merged-model-v2-clean \
  --input eval_freeform/dev.jsonl \
  --output results/freeform_refll_v2-clean.json \
  --device cpu

python autoresearch/score_primary.py \
  --mcq-result results/mcq_gold_ll_v3_clean.json \
  --freeform-result results/freeform_refll_v3-clean.json \
  --output results/primary_objective_v3_clean.json

python autoresearch/score_primary.py \
  --mcq-result results/mcq_gold_ll_v2_clean.json \
  --freeform-result results/freeform_refll_v2-clean.json \
  --output results/primary_objective_v2_clean.json
```

No sealed free-form test file was read; both ref-LL commands used
`eval_freeform/dev.jsonl`.

## Cross-approach sanity: C3-climbable primary

The active policy is `autoresearch_primary_objective_policy_v2`:

```text
primary = 1.0 * mcq_mean_option_set_gold_logprob
freeform_ref_ll_weight = 0.0
cross_model_refll_comparison = invalid_unless_same_prompt_lineage
```

| model | MCQ option-set LL | revised primary | MCQ argmax accuracy |
| --- | ---: | ---: | ---: |
| v3-clean | -3.0205043064 | **-3.0205043064** | 0.2920353982 |
| v2-clean | -3.8423283268 | **-3.8423283268** | 0.1769911504 |

The revised climbable objective ranks v3-clean above v2-clean, matching the intended
direction and no longer rewarding base-prose/legacy behavior.

Paired MCQ option-set LL stats over 113 shared MCQ question IDs, delta = v3 - v2:

| statistic | value |
| --- | ---: |
| mean delta | +0.8218240204 |
| sample SD | 3.9852244097 |
| SE | 0.3748983767 |
| z | 2.1921247766 |
| bootstrap SD (20k resamples, seed 20260716) | 0.3723124711 |
| bootstrap q05 / q50 / q95 | +0.2020290353 / +0.8238781185 / +1.4242727335 |
| P(opposite sign or zero) | 0.01585 |
| items where v3 > v2 | 69 / 113 |

## Same-lineage proxy sanity: chat-template ref-LL diagnostic

The #50 scorer fix was active in both output JSON files:

| model | prompt mode | chat template used | mean reference logprob/token |
| --- | --- | --- | ---: |
| v3-clean | `chat_assistant_reference_v1` | true | -5.1820120413 |
| v2-clean | `chat_assistant_reference_v1` | true | -4.6051304144 |

The sign still does **not** flip after applying the chat template. Under #51 policy,
this activates the removal path: ref-LL is not part of the primary objective and must
not be used as a v2-vs-v3 quality comparator. It remains only a labelled diagnostic
that can be reconsidered later for same-lineage v3-family variants sharing prompt
template, training interface, and answer surface.

Paired free-form ref-LL stats over 90 dev items, delta = v3 - v2:

| statistic | value |
| --- | ---: |
| mean delta | -0.5959684902 |
| sample SD | 0.3797283865 |
| SE | 0.0400268865 |
| z | -14.8892043074 |
| bootstrap SD (20k resamples, seed 20260716) | 0.0399187416 |
| bootstrap q05 / q50 / q95 | -0.6618057119 / -0.5956070820 / -0.5311297867 |
| P(v3 better than v2) | 0.0000 |
| items where v3 > v2 | 1 / 90 |

## Guardrail status

Rubric grading was not run in this pass because API budget was not explicitly approved.
That does not block the revised #35 sanity result: the high-frequency climbable metric
is MCQ option-set LL, while rubric remains the hard periodic Goodhart guard for #24/C3.

## Conclusion

#35 passes under the revised criterion:

1. The metric C3 will actually climb is MCQ option-set LL only.
2. That metric ranks v3-clean above v2-clean with paired SE/variance reported above.
3. Ref-LL no longer participates in accept/revert decisions and is explicitly labelled
   invalid for cross-approach comparison unless the same-lineage condition is met.
4. The sealed free-form test split was not read.

This resolves the #49 scorer/objective semantics blocker for C2 and unblocks the next
gate/guardrail work, subject to keeping rubric as the periodic quality guard.
