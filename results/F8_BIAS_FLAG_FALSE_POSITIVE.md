# F8 follow-up: the bias guard flag is a false positive

Date: 2026-09-04. Re-examination of the F8 bias rejection
(`bias_reintroduced observed=0.1173 baseline=0.0597 allowed=0.1097`).

## Finding

The bias guard compares the candidate's predicted-letter marginals on
`data/val_v3.jsonl` against a **uniform** reference (0.25 per letter). But
val_v3's 113 scorable MCQ rows have a **skewed gold distribution**:

| Letter | val_v3 gold (n=113) | v5clean argmax | distill argmax |
| --- | --- | --- | --- |
| A | **15 (13.3%)** | 26 | **15** |
| B | 26 (23.0%) | 23 | 33 |
| C | 43 (38.1%) | 35 | 36 |
| D | 29 (25.7%) | 29 | 29 |

The distill candidate predicts **A exactly 15 times** — identical to the gold
count. Its A-marginal (0.1327) matches the gold A-fraction (0.1327) to four
decimals. The guard flags this as "A-aversion" only because it expects 25%
uniform, but A is genuinely rare in this val sample.

**The baseline (v5clean) was the actually-biased one on A**: it over-predicted
A 26 times against 15 gold (deviation +0.097 from gold). The distill model
*corrected* this to 15 = 15, and the guard interpreted that correction as a
reintroduced bias because v5clean happened to sit closer to the uniform
reference (0.230) than to the truth.

The candidate's remaining off-diagonal mass is a B/C confusion (33 vs 26 and
36 vs 43), not an A-aversion.

## Conclusion

1. The F8 bias rejection is a **guard miscalibration**, not a model bias:
   the reference distribution should be the val's gold marginals (or the val
   split should be letter-balanced), not uniform.
2. F10's original premise — "stale option-letter references in the
   cyclic-shifted rationale cause an A-aversion" — is **not supported**. The
   A-marginal is perfectly calibrated.
3. The **only** real F8 blocker is the primary delta: +0.2186 = +1.09 SE,
   below the loop's 2-SE (0.4 raw) promotion bar.

## What this changes

- F10 should be re-scoped from "clear the bias flag" to a **training-signal
  cleanliness** question: the rationale's stale "Option X" references still
  create internally-inconsistent training rows after cyclic shift, which may
  cap the primary — but this is a hypothesis to test, not a measured bias to fix.
- The bias guard reference (uniform) should be recalibrated against val_v3's
  actual gold letter distribution before it is used to reject any candidate.
