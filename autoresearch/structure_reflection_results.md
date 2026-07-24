# Phase 18: multi-turn and reflection results

All four pilots started from the same strict v5clean recipe and used the same
140-train / 40-validation source dose. Runs were serial on the single GPU.
The frozen English MCQ primary was the accept/revert signal; the sealed
Spanish benchmark and sealed English free-form test were not read, and no
external free-form judge was used.

The v5clean reference is MCQ primary `-0.77515369` and MCQ argmax accuracy
`0.72566372`.

| Variant | Native validation token-F1 | Native corrected-answer F1 | MCQ primary | Primary delta | MCQ argmax | Disposition |
|---|---:|---:|---:|---:|---:|---|
| Flattened control | `0.3113` → `0.3146` (`+0.0033`) | n/a | `-1.7294` | `-0.9543` | `0.3982` | Reject; source replay does not protect the anchor |
| Preserved multi-turn | `0.3469` → `0.3739` (`+0.0270`) | n/a | `-1.7036` | `-0.9284` | `0.4071` | Reject; best closed-book candidate, still far below anchor |
| Interrupted/counterfactual | `0.5077` → `0.5493` (`+0.0417`) | `0.5093` → `0.5409` (`+0.0316`) | `-2.2033` | `-1.4282` | `0.2920` | Reject; native reflection gain does not transfer |
| Answer-audit/self-critique | `0.4185` → `0.5993` (`+0.1808`) | `0.4315` → `0.5734` (`+0.1419`) | `-1.7892` | `-1.0141` | `0.3717` | Reject; strongest native gain, unacceptable anchor regression |

## Interpretation

The local task metrics demonstrate that the targets were learnable, but they
also expose a large closed-book interference cost. The flattened control
rules out conversation structure alone as the explanation. Preserving
multi-turn context is directionally better than flattening, but not enough to
retain the English benchmark. Reflection formats are even more specialized:
they improve audit/correction behavior on their own validation surface while
damaging the primary MCQ capability.

No candidate passed the bilingual acceptance gate, so no candidate is promoted
to a proportion sweep or combination experiment. The next reflection attempt
needs a materially different recipe, such as a much smaller token-matched dose,
replay protection, or a separate auxiliary head; it should first be tested as
an individual lever against the same frozen anchor.

The source contained no explicit step-back examples. A future step-back pilot
therefore requires authored data and must remain a separate lever from this
phase.
