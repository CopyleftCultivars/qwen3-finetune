# Which metric should gate the distillation lever's promotion?

**Type:** research / policy decision
**Blocks:** F9 (#146, Cactus conversion) — F9 is gated on an F8 accept
**Depends on:** F8 (#145)
**Priority:** high — nothing downstream of F8 can move until this is answered

## The decision

F8 produced a result that the current gate cannot express: the distill candidate
**matches or beats the 8B teacher on the ladder (0.8714 vs 0.8571)** while
**failing the frozen MCQ primary gate**. Both numbers come from the same model.
Only one of them can be the right promotion criterion for this lever, and the
epic never said which.

This issue exists to answer that question *before* more compute is spent, not to
argue for a particular answer.

## Evidence on the table

- Ladder (210-item `natural_fertilizers_benchmark_v1_graded.jsonl`, temperature 0,
  generative scorer): distill candidate **0.8714**, qwen3:8b target line **0.8571**.
- Frozen MCQ primary: `primary_delta = +0.2186` = **+1.09 SE** against a bar of exactly
  **0.40** (`primary_gate.min_delta_standard_errors = 2.0` x `standard_error = 0.2`,
  `autoresearch/gate_config.json`). Direction positive; magnitude ~55% of the bar.
- Guardrails: **clean**. The bias rejection in the first F8 pass was a false positive in
  the guard itself (uniform reference instead of the eval's gold marginals); fixed in
  ac6888e / 1966a7a. The primary is the sole remaining blocker.

### What the primary actually is (verified)

Per `autoresearch/objective_weights.json`:

    primary = 1.0 * mcq_mean_option_set_gold_logprob

i.e. the mean **option-set-normalized gold answer-LETTER logprob** per MCQ item on
`data/val_v3.jsonl` (n = 489). `freeform_ref_ll_weight` is 0.0 -- diagnostic only.
`mcq_argmax_accuracy` already exists in `secondary_metrics_reported_only`.

### Where the 0.2 SE came from (verified)

`standard_error = 0.2` is **not** an arbitrary constant. Commit 45413bc (human-approved
policy change 2026-07-18, chainlink #72) corrected it 0.3749 -> 0.20:

> the measured paired candidate-vs-champion item SE is 0.152 plus run-to-run SD 0.128,
> combined 0.198. The 2-SE acceptance bar becomes 0.40.

This matters for how this issue is argued -- see below.

## Why this is a policy question and not a "train harder" question

**1. The bar is noise-limited, not n-limited.** An earlier framing of this issue called
the test "underpowered at its current n". That was wrong and is corrected here: the SE is
empirically calibrated, and more than half of it (run-to-run SD 0.128 vs item SE 0.152) is
*training-run reproducibility*, not eval sampling error. Enlarging val_v3 would therefore
barely move the bar. The honest statement is narrower: at +1.09 SE the candidate's effect
is real in direction but **smaller than this lever's own run-to-run variance**, so the gate
cannot distinguish it from a lucky seed. "We cannot distinguish it from noise" and "it is
not there" remain different findings, and the gate reports them identically.

**2. The primary and the lever disagree about what the answer is.** The primary scores the
gold *answer-letter* logprob. The ladder scores generative accuracy. This is not a
generic "different metrics" objection -- it is the same letter-surface dependency that
closed the letter-invariant arm below: the cyclic shift is mandatory because the teacher's
gold letters are skewed 58% B, and the primary is measured on exactly that letter surface.
A lever whose data required letter-position remediation is being gated on a
letter-position likelihood.

Notably, `objective_weights.json`'s own rationale already anticipates this:

> Cross-approach quality is guarded by generation accuracy and/or the free-form rubric,
> not by ref-LL.

The distillation lever is a cross-approach change (new data source, new teacher). The
policy text arguably already routes it to generation accuracy -- which is the ladder.

## The options

- **(A) Primary stands as the gate.** The ladder win is treated as a
  generative-format artifact; the candidate is rejected; the lever needs a real
  primary-mover. Cost: F9 stays blocked, and we accept an underpowered test as a
  negative result.
- **(B) Ladder becomes the promotion gate for deploy-shaped levers**, with the
  primary demoted to a non-regression guard (e.g. reject only on a *negative*
  primary delta beyond noise). Under this policy the current candidate is
  promotable today and F9 unblocks. Cost: changes the gate contract mid-epic;
  needs to be written down so it isn't relitigated per-lever.
- **(C) Both must pass; separate the two components of the SE.** Keep both metrics, but
  split `standard_error` into its item-sampling (0.152) and run-to-run (0.128) parts and
  decide which the acceptance bar should be built on. A bar at 2x item-SE alone is 0.304,
  which the current +0.2186 still misses -- so this is not a backdoor accept. It would
  instead be settled by seed replication: run the distill config at 2-3 seeds and see
  whether the delta is stable. Cost: most defensible, most compute.

## What is explicitly NOT on the table

**The letter-invariant rationale arm is closed.** It was investigated before
committing ~3h of regeneration and the premise does not survive contact:

1. The teacher ignores the instruction. A smoke test prompting "refer to each
   option only by content, never by letter" still emitted "Option A/B/C/D" and
   "the correct answer should be B" in 5/5 items.
2. Naive post-processing destroys the text — dangling fragments
   ("the correct answer. They build structures...") and "one option says X" x4.
   Degraded, not usable.
3. Cyclic shift cannot be dropped. The teacher data's original gold letters are
   skewed A 10.8% / B 58.1% / C 29.0% / D 2.1%; without the shift we would train a
   58%-B answer bias. The shift is load-bearing, and the shift is precisely what
   makes every rationale's letter reference stale.

Achieving letter-invariance would therefore require a dedicated LLM rewrite pass
(~2.5h of 8B inference) whose output quality has no gate, in service of a metric
this issue is questioning. Recorded here so the arm is not re-proposed.

## Queued behind this decision (diagnostic, not a promotion attempt)

An 8-epoch retrain (4 -> 8) of the same `phaseNN-distill` config. Framing matters:
doubling epochs on distillation data tends to sharpen teacher-mimicry rather than
generalization, and the primary is a held-out likelihood measure. A plausible
outcome is that the ladder goes **up** and the primary goes **down** — which would
be strong evidence for option (B)/(C) rather than a primary push. Run it as a
diagnostic that informs this decision; do not run it as the promotion arm until
this issue says the primary is the thing to push.

## Acceptance

- One of (A) / (B) / (C) chosen and recorded in `LADDER_AND_DISTILLATION_PLAN.md`
  as the promotion policy for deploy-shaped levers.
- `autoresearch/gate.py` config updated to match the chosen policy (or explicitly
  confirmed unchanged under (A)).
- F8's accept/reject verdict re-recorded under the chosen policy, and F9 (#146)
  either unblocked or formally held.
