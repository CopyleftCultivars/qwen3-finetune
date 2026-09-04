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
| Bias (letter marginals) | **flag** — A under-predicted: 0.1327 vs 0.25 reference |
| Rubric / grader-anchor | carried forward (not measured this batch) |

**The bias flag is real and marginal.** Candidate A-deviation 0.1173 vs the
allowed 0.1097 (baseline 0.0597 + 0.05 tolerance). The training data is
letter-balanced (25/25/25/25 via cyclic shift), so the A-aversion most likely
comes from the teacher rationale text referencing option letters ("Option A
is…") that go stale after the cyclic shift rotates the options.

## 4. Gate decision

`results/autoresearch_gate_distill-v2-rationale.json`: **REJECT**, for two
independent reasons:
1. `primary_delta_not_significant` (delta +0.2186 < threshold 0.4).
2. `bias_reintroduced` (observed 0.1173 > allowed 0.1097).

## Verdict

The distillation lever **works on its actual target** — the ladder gap to an
8B is closed (0.7810 -> 0.8714, above the 0.8571 target), and the hard tier
doubles the champion's improvement against the teacher. But it does **not**
pass the autoresearch gate as-is: the loop's 2-SE primary bar is not cleared
(+1.09 SE), and a letter-bias side effect (A-aversion) trips the bias guard.

Recommended follow-up (F8 pass path): regenerate the teacher rationale to be
letter-invariant (no "Option A/B/C/D" references), rebuild the distillation
dataset, and retrain — this targets the bias flag directly and should leave
the ladder gain intact.
