# v3-clean Evaluation Summary (Phase 5)

Status: **Complete** (Stage A + Stage B, same three diagnostics run for v2-clean —
calibration, logit lens, linear probes — all against the full 1603-question benchmark).
`selection_bias.py` was not run for either checkpoint (v2-clean's own eval skipped it
too — its live report predates v2-clean's training); see chainlink #13 for a follow-up
that runs it identically on both checkpoints for a purpose-built letter-bias comparison.

## What v3-clean is

Per `MCQ_DIAGNOSTIC_PLAN.md` Phase 4, trained on gemma4 conversations (4409 pairs) +
the Phase 3 clean MCQ source (`data/sources/qmd_mcq_all_phase3.jsonl`, 311 questions,
judge-approved + benchmark-deduped, cyclic-shift balanced to an exact 25/25/25/25
A/B/C/D split via `prepare_dataset_v3.py --qmd-mcq`). 4400 train / 489 val pairs total.
Trained 31:11, default hyperparameters (6 epochs, lr 2e-4, LoRA r=32/alpha=64), merged
to `outputs/merged-model-v3-clean` (1137MB). Generation-accuracy eval used
`--max-new-tokens 220` (matching the setting that fixed a think-block truncation bug
in the original, contaminated v3 run).

## Stage A results

| Metric | v2-clean | v3-clean |
|---|---|---|
| Generation accuracy | 42.55% (682/1603) | **73.0%** (1170/1603, 2 ambiguous) |
| Logprob accuracy (forced empty `<think></think>`) | 31.8% (509/1603) | 32.6% (523/1603) |
| Mean entropy | 0.645 bits | 0.864 bits |
| Error buckets | correct 31.8%, confident_wrong 0%, uniform 13.0%, biased_letter 55.2% | correct 32.6%, confident_wrong 43.0%, uniform 24.3%, biased_letter 0% |
| Letter win distribution (logprob top choice) | A 1237 (77.2%), B 271, C 31, D 64 | A 876 (54.6%), B 181 (11.3%), C 370 (23.1%), D 176 (11.0%) |
| logprob✓ but gen✗ / gen✓ but logprob✗ | n/a (no cross-ref in v2-clean write-up) | 137 / 784 |

Baselines (unchanged benchmark composition): random chance 25.0%, majority-class
('A' at 26.1%) 26.1%. v3-clean beats v2-clean's generation accuracy by +30.5pp and beats
majority-class by +46.8pp.

### Reasoning

**Generation accuracy is the headline result and it moved a lot: +30.5pp.** This
directly confirms the Phase 2 diagnosis — v2-clean's dominant failure mode was lack of
MCQ-format exposure entirely, not a rationale-quality problem, and the Phase 3/4 fix
(balanced, cyclic-shifted MCQ training data) targeted exactly that.

**The logprob probe barely moved (31.8%→32.6%), and that's expected, not concerning.**
`calibration.py` scores the model under a forced *empty* `<think>\n</think>\n` block —
skipping the reasoning step entirely. v3-clean was trained to produce genuine CoT before
answering, so forcing it to skip that step is an increasingly out-of-distribution prompt
shape for this checkpoint specifically. The 784 questions correct in generation but not
via the logprob shortcut confirm v3-clean's real strength lives in its reasoning path,
not a reflexive single-token guess — a different (and arguably healthier) reliance than
a checkpoint whose logprob and generation numbers track closely.

**Letter bias changed shape rather than simply shrinking.** A's win share fell from 77.2%
to 54.6% and the `biased_letter` bucket collapsed to 0%, exactly the direction the Phase
3/4 cyclic-shift design targeted (D marginal was 16.9% vs a 25% target under the original,
contaminated pipeline; this is the same mechanism, now on a clean source). But under this
narrow forced-empty-think probe, those errors didn't disappear — they moved into
`confident_wrong` (0%→43.0%) and `uniform` (13.0%→24.3%). The model still often backs a
specific wrong letter confidently, or is genuinely uncertain, when denied its reasoning
step. This is a real, different failure mode, not the same one just smaller.

## Stage B: logit lens (complete)

50 correct + 50 wrong questions sampled from v3-clean's own generation results:

| Split | v2-clean median first-correct layer | v2-clean never-correct | v3-clean median first-correct layer | v3-clean never-correct |
|---|---|---|---|---|
| correct | 1/27 | 17/50 | 24/27 | 21/50 |
| wrong | 28/27 (never, by definition) | 26/50 | 23/27 | 19/50 |

### Reasoning

**This is not simply worse — it's a different and more informative pattern.** v2-clean's
near-instant correct-split decoding (layer 1) was mostly an artifact of its severe A-bias:
the model already "knew" its answer from the first layer because it always leaned A, and
A happens to be gold 26.1% of the time (flagged explicitly in the v2-clean write-up as
likely correct-by-coincidence). With that bias substantially reduced in v3-clean, early
layers no longer trivially leak an answer — the network now needs to go almost to the
final layer (24/27) to settle on the correct letter, consistent with real content-based
computation replacing a letter-preference shortcut.

**The wrong-split tells a genuinely new story.** v2-clean's wrong-split never resolves to
correct by definition of the metric (median layer 28, i.e. "never" for at least half the
sample) — diagnosed as knowledge genuinely absent. v3-clean's wrong-split median
first-correct layer is 23/27, nearly matching the correct-split's 24 — meaning for a large
share of wrong answers, the correct letter *does* become decodable deep in the network,
just loses the final routing step. That's a **present-but-barely-not-surfaced** pattern:
distinct from both v2-clean's "knowledge absent" verdict and the original contaminated
checkpoints' "present by layer 8" story. v3-clean's remaining failure looks more like a
late-stage decision problem than a knowledge gap.

## Stage A follow-up: selection bias / PriDe (complete, chainlink #13)

Neither checkpoint's original eval ran `selection_bias.py` (v2-clean's live report predated
its training). Ran it dedicated, on both checkpoints, for a purpose-built letter-bias
comparison — full 1603-question benchmark, cyclic permutation test + PriDe debiasing
(Zheng et al. 2309.03882).

| Metric | v2-clean | v3-clean |
|---|---|---|
| Standard accuracy (shift=0) | 31.8% | 32.6% |
| Mean permutation accuracy | 31.3% | 32.2% |
| PriDe debiased accuracy | 38.7% | 33.9% |
| PriDe improvement over standard | +6.9pp | +1.3pp |
| Mean per-question spread | 0.982 | 0.947 |
| Letter marginal | A 77.4%, B 16.8%, C 1.9%, D 3.9% | A 54.1%, B 11.0%, C 23.1%, D 11.8% |
| Marginal entropy | 1.009 bits (max 2.0) | **1.683 bits** (max 2.0) |
| Decision guidance | skewed marginal + spread flagged; apply PriDe | only spread flagged |

### Reasoning

**This resolves the ambiguity calibration.py's side-metrics left open, and it's good news:
selection bias was genuinely reduced, not just reshaped.** Marginal entropy jumped from
1.009 to 1.683 bits — more than 60% of the way to the 2.0-bit uniform ceiling — meaning the
letter marginal is now much closer to flat. PriDe's benefit shrank from +6.9pp to +1.3pp,
consistent with the raw model no longer leaning heavily on a position-based prior to begin
with; there's much less bias left for debiasing to remove. This confirms, with a dedicated
purpose-built diagnostic, what calibration.py's error-bucket reshaping only hinted at
(A-win 77.2%→54.6%, biased_letter 55.2%→0%).

**One thing that did *not* improve: mean per-question spread (0.982→0.947), essentially
flat.** Spread measures whether an individual question flips between correct and incorrect
depending on which position the gold answer lands in — i.e., per-question position
sensitivity, distinct from the aggregate letter marginal. Even though v3-clean's *aggregate*
letter preference is much more uniform, individual questions are still nearly as likely to
flip right/wrong under permutation as they were for v2-clean. This is a genuine residual
weakness: v3-clean isn't yet close to fully position-invariant on a per-question basis, even
though its aggregate selection bias improved substantially.

## Stage B: linear probes (complete)

`diagnostics/linear_probes.py` against `unsloth/Qwen3-0.6B` (base) and
`outputs/merged-model-v3-clean` (finetuned), full 1603-question benchmark.

**Correction note:** the first run of this diagnostic was invalidated by an invocation
mistake — `--finetuned-results` was omitted and silently defaulted to a stale results
file from the original (contaminated) checkpoint, corrupting the prediction-probe and
gap numbers (though not the knowledge-probe numbers, which don't depend on that file).
Re-run with `--finetuned-results results/qwen3-0_6b-v3-clean_results.json`; the invalid
first run is preserved as `results/probe_results_v3clean_INVALID_wrong_gen_labels.json`
for the record.

| Model | Peak knowledge accuracy | Peak layer | Max knowledge-prediction gap |
|---|---|---|---|
| base (Qwen3-0.6B) | 30.7% | 17 | v2-clean run: −0.077 / v3-clean run: +0.018 |
| v2-clean finetuned | 25.8% | 24 | −0.131 |
| v3-clean finetuned | 23.0% | 17 | **−0.006** |

Generation accuracy as measured by this script: 73.1% (matches evaluate.py's 73.0%,
confirming the corrected run loaded the right data).

### Reasoning

**Base model reproduced identically for a third time** (30.7% at layer 17) — continues to
be a solid reproducibility anchor for the pipeline.

**Peak knowledge accuracy for v3-clean (23.0%) is modestly below both the base model
(30.7%) and v2-clean (25.8%).** Taken alone this looks like a regression, and the script's
own decision guidance says "knowledge is ABSENT — invest in domain data." That heuristic
doesn't have access to the 73% real generation accuracy, so don't take it as the practical
verdict — it measures something narrower: linear decodability of the *gold* letter from
residual stream activations under the same forced-empty-think prompt shape discussed
above, which remains somewhat out-of-distribution for a checkpoint trained to reason
before answering.

**The max knowledge-prediction gap essentially closed: −0.131 (v2-clean) → −0.006
(v3-clean).** This gap is `knowledge_acc − prediction_acc`, where "prediction" probes
decodability of the *model's own eventual answer* (right or wrong), not the gold letter.
A strongly negative gap (v2-clean) means the model's own output was far more decodable
than the correct answer — a real disconnect between what the model was about to say and
what was actually right, consistent with knowledge effectively decoupled from behavior
under this probe. A near-zero gap (v3-clean) means those two are now almost equally
decodable — the model's residual stream has moved toward a regime where predicting its
own answer and predicting the correct answer are nearly the same problem. Combined with
the 73% generation accuracy and the logit lens's "present-but-barely-not-surfaced"
wrong-split story, this is a coherent picture: v3-clean's remaining gap is much smaller
and looks like a late-stage surfacing/routing shortfall, not the "knowledge absent, format
never seen" story that described v2-clean.

## What this means going forward

- The core Phase 3/4 hypothesis — that v2-clean's dominant failure was lack of
  MCQ-format exposure, fixable with a clean, balanced MCQ source — is now confirmed by a
  fair, uncontaminated v2→v3 comparison for the first time in this project's history.
- v3-clean's remaining weaknesses are narrower and later-stage than v2-clean's: under the
  forced-empty-think probes, letter bias persists in a different shape
  (`confident_wrong`/`uniform` rather than `biased_letter`), and logit lens shows a
  present-but-not-quite-surfaced pattern on the wrong split rather than absent knowledge.
- Chainlink #13 (dedicated `selection_bias.py`/PriDe run on both checkpoints) is now
  complete and confirms the letter-bias reshaping above is a genuine net reduction in
  selection bias (marginal entropy 1.009→1.683 bits), not just a redistribution — though
  per-question position spread remains a residual weakness (0.982→0.947, essentially flat).
- Tool-calling (deferred, chainlink #10) remains out of scope until these MCQ-focused
  phases are considered complete.
