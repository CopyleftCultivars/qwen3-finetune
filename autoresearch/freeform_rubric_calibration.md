# Free-form rubric calibration

The first external rubric calibration is complete using the paired dev-only
pipeline and the v5clean model as both baseline and candidate. The judge was
the pinned OpenRouter routine grader from `eval_freeform/scorer_config.json`.
The fixed English dev split contributed all 90 items: 18 per rubric type.

## Calibration result

| Measure | Baseline | Candidate | Difference |
|---|---:|---:|---:|
| Overall rubric score | 46.0667% | 46.0667% | 0.0000 pp |
| Reference-LL / token | -3.66685 | -3.66685 | 0.00000 |
| Maximum grader-anchor drift | 0.0000 | 0.0000 | no alerts |

Rubric score by type for this v5clean/OpenRouter calibration:

| Type | Score |
|---|---:|
| Constraint satisfaction | 45.7778% |
| Diagnostic reasoning | 39.1667% |
| Free-form explanation | 26.0556% |
| Multi-hop factual | 65.3889% |
| Quantitative | 53.9444% |

The historical `codexv5` rubric result is not numerically comparable: changing
judge backend/model invalidates direct rubric comparisons. Future baseline and
candidate runs must use this same OpenRouter configuration, fixed dev IDs, and
the same five anchors.

## Disposition

The evaluation path is now operational and externally calibrated. The rubric
is still a secondary guardrail, not a weighted primary objective. Before using
it for accept/revert decisions, the next candidate comparison must confirm the
same-backend paired behavior and establish the practical rubric-drop threshold
alongside the existing English MCQ primary gate.

Calibration metadata:

- pipeline report: `results/freeform_eval_freeform-calibration-v5clean.json`
- routine judge: `anthropic/claude-haiku-4-5` via OpenRouter
- scorer config SHA-256: `74a9101089d5341624f0e8c9588ba7c509ed48629e8948d05595846356366b38`
- sealed test paths read: none
- Spanish benchmark paths read: none
