# Free-form rubric grades already exist, and they change the RAG prior (#130)

Date: 2026-09-03. Source: `results/slice_v1_freeform_*_rubric.jsonl`, produced by
commit `3a9afef` (a parallel agent's free-form implementation), graded with
`grader_model: codex/gpt-5.6-luna` against `eval_freeform/scorer_config.json`.
75 paired items.

## Rubric weighted score -- the actual deploy target

| Arm | Mean weighted score | Paired delta vs A | t | better / worse / same |
| --- | --- | --- | --- | --- |
| **A** bare | **0.5569** | - | - | - |
| R retrieval | 0.4804 | -0.0765 | -2.57 | 12 / 27 / 36 |
| C self-refine | 0.4079 | -0.1491 | -4.66 | 11 / 36 / 28 |
| B RLM REPL | 0.0875 | -0.4695 | -14.35 | 1 / 68 / 6 |

**Arm A wins the free-form track too, and by a significant margin over every
elaboration.** This is the same ordering the MCQ track produced, on the metric
that actually matters for the product.

## The corpus gap is not masking a retrieval benefit

The obvious objection to the table above is that 49/75 of Arm R's items
retrieved nothing, so R is mostly just A with noise. Restricting to the **26
items where retrieval actually fired**:

    delta vs A = -0.1885, better = 5, worse = 15

Retrieval is **worse where it works**, and worse than its own all-items average.
That is the opposite of "the corpus gap is hiding a gain". The most likely
reading: retrieving from a corpus that does not contain the source material
injects plausible-but-irrelevant agronomy text, and the 0.6B follows it away
from a better unaided answer.

This does not make the re-run pointless -- retrieval fired against the *wrong*
corpus, and indexing `qtmd_workspace/markdown_corpus` could flip the sign. But
the prior going in should now be that RAG hurts this model, not that it helps,
and the deployment-upper-bound (Config A) number should be read with that in
mind.

## Two confirmations requested by the review

1. **codex CLI is authenticated.** `codex` resolves, `codex exec` runs, and the
   existing rubric rows carry `grader_model: codex/gpt-5.6-luna` -- the backend
   already produced real grades. OPENROUTER_API_KEY is genuinely not required,
   and my earlier claim that grading was blocked on it was wrong.
2. **QMD can isolate an index.** `qmd --index <name>` ("Use a named index,
   default: index") is a global option. Phase 3 therefore does **not** need to
   modify the shared store and does not need owner escalation -- the free-form
   corpus can be indexed into a slice-local named index.

## Duplicate implementation to resolve

Two parallel free-form paths now exist:

* `rlm_slice/freeform_arm.py` + `rlm_slice/eval_slice_freeform.py` (`3a9afef`,
  parallel agent) -- all four arms, already rubric-graded. The review's line
  references (219/256) point here.
* `rlm_slice/eval_freeform_slice.py` (`2c45a58`, mine) -- base/R only, plus a
  local reference-LL proxy.

The first is strictly further along. Keeping both invites two divergent
free-form numbers, so mine should be retired and its one distinct feature (the
local reference-LL proxy) folded in only if wanted.
