# Phase 22: context-grounded open-book result

Phase 22 was trained as one isolated lever on the strict v5clean base. It did
not include Spanish data, previous auxiliary/reflection streams, verification,
general replay, or distractor-quality data.

## Training

| Item | Value |
|---|---:|
| Composed train / validation rows | 4,652 / 553 |
| Grounded train / validation rows | 256 / 64 |
| Steps / epochs | 1,164 / 4 |
| Training runtime | 29:55 |
| Final training loss | 1.1433 |
| Model | `outputs/merged-model-p22-open-book-grounded` |

The external CUDA run completed without OOM, NaN, or trainer failure. The
20-minute checkpoint was step 582/1,164; the later step-873 checkpoint also
remained healthy.

## Native grounded validation

The fixed 64-item context-present validation stream was scored on both models.

| Metric | v5clean | Phase 22 |
|---|---:|---:|
| Answer token F1 | 9.17% | 91.43% |
| Answer exact | 1.56% | 68.75% |
| Evidence-source exact | 10.94% | 81.25% |
| Fully grounded exact | 0.00% | 67.19% |
| Parse-valid target | 9.38% | 84.38% |

This confirms that the new target is learnable and materially different from
the earlier generic CROP open-book auxiliary diagnostic.

## Closed-book promotion gate

| Metric | v5clean | Phase 22 | Delta |
|---|---:|---:|---:|
| English MCQ option-set gold logprob | -0.7751537 | -0.6751628 | +0.0999909 |
| English MCQ argmax | 72.57% | 77.88% | +5.31 points |

The improvement is only `+0.50` standard errors against the frozen `+2 SE`
(`+0.4`) promotion threshold. Bias remained within the existing guard, with
candidate letter marginals A/B/C/D = 0.2389/0.2389/0.2832/0.2389. The frozen
gate therefore rejects the candidate for promotion.

The fixed general-knowledge forgetting probe was not run because the external
execution approval policy rejected that invocation. The OpenRouter free-form
guard was also not run because the current policy rejected transmitting model-
generated English dev answers. No Spanish or sealed test path was read.

## Disposition

Keep v5clean as the champion. Do not mix Phase 22 with NER or any other
unpromoted lever yet. Retain the Phase 22 source builder, source manifest, and
native grounded scorer as a useful context-present diagnostic; a future dose or
retrieval-policy experiment would need a new hypothesis and a fresh gate.

Artifacts: `results/primary_p22-open-book-grounded.json`,
`results/autoresearch_guardrails_p22-open-book-grounded.json`, and
`results/autoresearch_gate_p22-open-book-grounded.json`.
