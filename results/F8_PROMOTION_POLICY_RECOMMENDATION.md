# F8 promotion policy: which metric gates the distillation lever? (#160)

Date: 2026-09-04. Decision request from #160. Deliverable: a recommendation with
evidence, not a code change.

## The three questions, answered from source

### 1. Has the 2-SE primary bar been real in practice?

Yes — strictly. Tabulating `primary_delta_standard_errors` across every recorded
gate decision (`results/autoresearch_gate_*.json`, n=72):

| Decision | n | dse range |
| --- | --- | --- |
| accept | 10 | **2.186 … 7.845** |
| reject | 61 | −10.03 … 3.542 |
| pause | 1 | −0.018 |

Every accepted lever cleared 2 SE; the lowest accepted dse is 2.186
(`p5real6-d01-r4`), and the champion promotion (`p6-lr5e5*`) sat at 7.0–7.8 SE.
The primary bar is **necessary but not sufficient**: no accept sits below 2 SE,
yet six rejects exceeded 2 SE (dse 2.118–3.542 — `p5accept-ep4_full` 3.542,
`p5real6-d01-r3` 3.428, `p5real6-d01` 2.765, `p5real2-epochs` 2.410,
`p5real6-d025` 2.262, `p5real6-d01-r2` 2.118) and were all killed by
**guardrails** (rubric drop, compute budget, bias drift), not the primary.
The bar is noise-limited, not n-limited: commit `45413bc` (2026-07-18, #72)
set `standard_error = 0.20` from the measured paired item SE 0.152 plus
run-to-run SD 0.128 (combined 0.198), so `min_delta_standard_errors = 2.0`
→ threshold 0.40. **The distill candidate's +0.2186 (+1.09 SE) is genuinely
below every historically-accepted move.**

### 2. Is the ladder sealed w.r.t. the distill training data?

Lexically, yes — strongly. Max 8-gram Jaccard over all 210 × 6437 pairs is
**0.091 (question stems)** and **0.034 (full instruction+output)**, versus the
0.60 alarm; 0 exact matches. The single highest pair (NF-0011 vs a comfrey
item) shares only the boilerplate of short questions and a source book title.

Provenance, no — the shared-source signal never ran. The ladder rows *do*
carry `source_document` (`corpus:Natural Fertilizers Benchmark/…`), but the
training pool uses a different field (`source_passage`), so the F2 source check
matched nothing. Shared source documents are real (e.g. the "Manual of On-Farm
Vermicomposting" feeds both a ladder item and a training MCQ). The questions
differ, so this is domain-coverage overlap, not question leakage — but it is
unverified, and both v5clean and the distill model share it (the base 0.6B
scores 0.4286, so the ladder demonstrably measures learned domain knowledge).

### 3. Does `mcq_argmax_accuracy` move for this candidate?

Yes — and the magnitudes disagree in a way that localizes the divergence to the
letter surface:

| metric | v5clean | distill | delta |
| --- | --- | --- | --- |
| `mcq_argmax_accuracy` | 0.7257 | 0.8584 | **+13.3 pp** |
| `mean_gold_logprob` (unnormalized) | −2.1003 | −0.5566 | **+1.544** |
| `mean_option_set_gold_logprob` (the primary) | −0.7752 | −0.5566 | +0.219 |

The primary is `log(softmax(gold) over A/B/C/D)` (`score_mcq_ll.py
_softmax_probs`), i.e. the gold letter's **share of probability among the four
letters**. The unnormalized gold logprob — the model's raw confidence in the
answer token — rose +1.54, and argmax accuracy rose +13 pp, but the
option-set-normalized share rose only +0.22. The normalization that the gate
scores is exactly the letter surface the lever had to remediate (the teacher's
original gold letters are 58% B, which is why the cyclic shift is mandatory).

## Decision: (B), with a provenance precondition

Recommend **option (B)** — the generative ladder becomes the promotion gate for
deploy-shaped levers, and the frozen MCQ primary is demoted to a non-regression
guard — **gated on closing the provenance hole first**.

Grounding:

1. **The policy text already routes cross-approach quality to generation
   accuracy.** `objective_weights.json` rationale: *"Cross-approach quality is
   guarded by generation accuracy and/or the free-form rubric, not by ref-LL."*
   The option-set-LL primary is named `first_c3_climbable_objective` and
   `high_frequency_primary_accept_revert_signal` — a same-lineage loop-climb
   signal. Distillation is a cross-approach change (new data source, new
   teacher). Gating it on the letter-surface primary contradicts the policy's
   own routing.

2. **(1) and (3) together rule out (A) as currently justified.** The primary bar
   is real, so (A) cannot be fixed by fudging it — but (3) shows the primary is
   suppressed on exactly the letter surface this lever remediates. Rejecting a
   model that matches the 8B teacher on the ladder, on a metric the policy says
   shouldn't govern cross-approach quality, is not defensible.

3. **(2) rules out (B) being adopted blindly.** The ladder is lexically sealed
   but provenance-unverified. The precondition is cheap and feasible: map
   ladder `source_document` corpus labels to the training pool's
   `source_passage`, and confirm no ladder item is answerable by memorising a
   same-source training question.

4. **(C)'s seed replication stays as a supporting diagnostic, not the decision.**
   Even a 2×item-SE bar (0.304) still misses +0.2186, so (C) would only confirm
   a primary reject it cannot express either. Replication instead settles
   whether the +9.0 pp ladder gain and the +0.219 primary gain are stable, which
   informs the (B) non-regression guard's noise bound.

## Non-regression guard under (B)

Deploy-shaped levers gate on the ladder; the primary becomes a guard that
rejects only on a **negative** primary delta beyond its noise bound (the
candidate's +0.2186 is positive, so it passes). Same-lineage loop promotions
(v5clean-style hyperparameter climbs) keep the existing 2-SE primary bar
unchanged.

## Acceptance follow-through

- Record the policy in `LADDER_AND_DISTILLATION_PLAN.md` (this document).
- Close the ladder provenance gap (mapping script + audit), then re-record F8:
  ladder pass (0.8714 > 0.8571) + primary non-regression + clean guardrails →
  **accept**, and unblock F9 (#146).
- Run the queued 8-epoch retrain and a 2–3-seed replication as diagnostics that
  confirm the divergence direction and the noise bound.
