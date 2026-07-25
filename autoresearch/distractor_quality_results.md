# Phase 21 result: distractor-quality MCQ authoring

The isolated candidate was trained from the strict v5clean recipe plus the
62 judge-kept distractor-quality questions. Training completed normally on
one RTX 4070 Ti in 28m49s: 1,152/1,152 optimizer steps, four epochs, final
reported training loss 1.2013. No CUDA, OOM, NaN, or gradient-health failure
occurred.

## Decision

**Rejected. v5clean remains the champion.** The frozen English MCQ primary
fell well below the promotion gate, and the answer-letter bias guard also
failed.

| Metric | v5clean | Phase 21 candidate |
|---|---:|---:|
| English MCQ primary | -0.7752 | **-2.2282** |
| Primary delta | — | **-1.4531 (-7.27 SE)** |
| English MCQ argmax accuracy | — | 30.09% |
| Candidate distractor-quality validation option-set LL | -0.7670 | -1.4610 |
| Candidate distractor-quality validation argmax | 69.44% | 41.67% |
| Full 90-item English free-form rubric | 46.0667% | **46.7222%** |
| Full free-form rubric delta | — | +0.6556 points |
| Periodic 75-item rubric | 47.5600% | 48.0800% |
| Periodic rubric delta | — | +0.5200 points |
| Periodic anchor drift | — | 0.0000 |
| General-knowledge probe argmax | 33.33% | 33.33% |

The free-form gains are real dev-only secondary diagnostics, but they cannot
override the primary MCQ failure. The candidate also reintroduced selection
bias: observed letter-bias distance 0.1659 versus the 0.1097 allowed limit.
The gate decision is recorded in
`results/autoresearch_gate_p21-distractor-quality.json`.

## Interpretation

The authoring and semantic quality gates succeeded: 62/170 repaired items
were judged keep, the source was contamination-audited, split atomically, and
exactly balanced after equal rotations. The failure happened after training:
the candidate did not convert that source quality into better closed-book MCQ
behavior. Its free-form style/quality improved slightly, while the MCQ answer
surface became substantially less calibrated and more position-biased.

This closes the current distractor-quality dose. Do not combine it with
another lever or promote it into the autoresearch champion. A future attempt
would need a materially different recipe or a much tighter source dose, and
would still have to clear the same English MCQ anchor and bias guards.

The paired free-form evaluation used the fixed 90-item English dev subset and
the authorized OpenRouter judge; it transmitted only those generated English
dev answers. No Spanish benchmark or sealed English test path was read.
