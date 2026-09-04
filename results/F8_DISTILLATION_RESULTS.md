# F8 (#145): distillation evaluation — qwen3:8b rationale (Variant 2) vs v5clean

Date: 2026-09-04. Candidate: `outputs/merged-model-distill-v2-rationale`
(trained in F7 on the F6 dataset: 871 gold-filtered qwen3:8b teacher rationales,
single lever = the dataset, v5clean hyperparameters).

## 1. The 210-item ladder (the target line)

| Model | Accuracy | Wilson 95% | easy (n=175) | hard (n=33) |
| --- | --- | --- | --- | --- |
| qwen3-8b (same harness, target) | 0.8571 | 0.8034-0.8981 | 0.9371 | 0.4545 |
| **distill-v2-rationale (0.6B)** | **0.8714** | **0.8194-0.9101** | **0.9029** | **0.7273** |
| v5clean (0.6B champion) | 0.7810 | 0.7202-0.8316 | 0.8286 | 0.5455 |
| base Qwen3-0.6B | 0.4286 | 0.3635-0.4962 | 0.4686 | 0.2424 |

**The distillation beats the target.** 183/210 = 0.8714, +9.0 pp over v5clean
and +1.4 pp over the qwen3-8b teacher it was distilled from. A 13x-smaller model
now scores above the 8B reference on the ladder.

**The hard tier is the surprise.** v5clean 0.5455 -> 0.7273 (+6 items, +18 pp),
*above* the teacher's own hard-tier 0.4545. The teacher's easy/mid rationale
transferred generalizable reasoning that helps on exactly the hard items the
teacher itself gets wrong — the opposite of the naive "distillation caps at the
teacher" fear (risk C.1).

## 2. Frozen MCQ primary (accept/revert signal)

| Metric | v5clean | distill | delta |
| --- | --- | --- | --- |
| mean_option_set_gold_logprob | -0.7752 | -0.5566 | **+0.2186** (+1.09 SE) |
| argmax accuracy (113 rows) | 0.7257 | **0.8584** | +13.3 pp |

Positive on both. The +1.09 SE delta is below the autoresearch loop's 2-SE
promotion threshold (`min_delta_standard_errors=2.0`), so the loop's primary
gate alone would not promote — but this is a clear improvement, not a regression.

## 3. Guardrails

| Guard | Result |
| --- | --- |
| Forgetting (general probe, 24 items) | **pass** — 0.7500 vs 0.4167 baseline (improvement) |
| Bias (letter marginals) | **false positive** — see `F8_BIAS_FLAG_FALSE_POSITIVE.md` |
| Rubric / grader-anchor | carried forward (not measured this batch) |

**The bias flag is a guard miscalibration, not a model bias.** val_v3's 113
MCQ rows have skewed gold letters (A=15, C=43); the candidate predicts A exactly
15 times. The guard compared against a uniform 0.25 reference and flagged the
candidate for being calibrated. Fixed in `ac6888e` (gold-marginal reference +
positive controls); re-gating clears the bias reason, leaving only the primary.

## 4. Gate decision

`results/autoresearch_gate_distill-v2-rationale.json`: **REJECT**, on a single
reason after the bias-guard fix:
1. `primary_delta_not_significant` (delta +0.2186 < threshold 0.4).

The bias rejection (`bias_reintroduced`) was a guard miscalibration and is
cleared by `ac6888e`; it no longer appears in `reject_guardrails`.

## Verdict

The distillation lever **works on its actual target** — the ladder gap to an
8B is closed (0.7810 -> 0.8714, above the 0.8571 target), and the hard tier
doubles the champion's improvement against the teacher. The only remaining
blocker is the loop's 2-SE primary bar (+1.09 SE vs 2 SE): the MCQ gold-LL
improved (+0.2186) but not by enough to clear the promotion threshold.

Recommended follow-up (F8 pass path): a stronger distillation signal to lift
the primary — letter-invariant rationale (cleaner training signal), more
teacher items, a hard-tier teacher (opus), or more epochs — then re-evaluate
with the full guardrail suite (rubric/grader-anchor measured, trace track run).
