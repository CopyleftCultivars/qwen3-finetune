# P13 novel rationale postmortem

## Decision

Reject `p13-novel-rationale-nf` and keep v5clean as champion. Do not repeat
this source or widen the rejected QMD replay overlay.

## Hypothesis and isolation

P13 held the v5clean/p12 recipe fixed and changed only the rationale target
construction. The source contained 43 natural-fertilizers items selected from
the v5clean `late_weak` routing category. A local CUDA Qwen authoring pass
created step-back rationales from the MCQ question and options while
preserving the source answer letter. The source was opt-in only and expanded
the selected rows to 172 cyclic answer-position variants.

Pre-training checks all passed:

- 0 exact output matches and 0 normalized rationale matches against the QMD
  rationale outputs;
- 0 exact benchmark-question overlaps and 0 high-lexical-overlap collisions;
- 4,409 conversation pairs + 1,555 QMD pairs + 172 rationale overlays;
- 6,136 pairs before filtering, 5,061 after filtering, 4,554 train / 507
  validation;
- final MCQ A/B/C/D counts 348 each;
- sealed-test paths read: 0.

## Gated result

| Metric | v5clean | P13 | Decision |
|---|---:|---:|---|
| Primary option-set gold LL | -0.77515 | -0.64442 | `+0.13073`, below `+0.4` gate |
| Primary delta in SE | — | +0.654 SE | below +2 SE requirement |
| Bias deviation | 0.0597 | 0.1202 | reject; 0.1097 allowed |
| General-knowledge accuracy | 41.67% | 45.83% | improved |
| Periodic rubric | 60.80% | 60.77% | within 5-point guard |
| Anchor drift | — | 0.0000 | clean |
| Sealed test | untouched | untouched | clean |
| Training runtime | — | 29.70 min | reject; 28.00 min budget |

The primary lift is directionally positive but too small, and the answer-prior
shift is the same failure family seen in p11/p12. The flat rubric means this
was not a catastrophic free-form-quality failure; it simply did not provide
enough reliable primary improvement to offset the bias and compute costs.

## Interpretation

P13 proves the source was textually novel, but not that it supplied reliable
new reasoning supervision. The local 0.6B authoring pass produced short,
generic explanations, so the next rationale hypothesis must change authoring
quality, supervision structure, or dose. Do not treat “different text” alone
as sufficient evidence for a new champion.
