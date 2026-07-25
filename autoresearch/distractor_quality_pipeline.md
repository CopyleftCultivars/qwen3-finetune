# Phase 21: MCQ distractor-quality authoring

This phase tests one new lever: improving the plausibility of incorrect MCQ
options and making them represent realistic domain misconceptions. It keeps
the v5clean optimizer, model, conversation pool, MCQ interface, contamination
rules, and English evaluation gates unchanged.

## Source construction

The authoring input was the previously judge-reviewed but unused QMD MCQ
pool. A local Qwen3-0.6B checkpoint proposed replacements for incorrect
options only. The local pass produced 181 structurally parseable candidates;
the contamination audit accepted 170 and rejected 11 malformed/bare-letter
records. The accepted candidates were sent to the authorized OpenRouter judge
(`anthropic/claude-haiku-4.5`) for semantic quality review.

The judge returned 62 `keep`, 107 `review`, and 1 `reject`. Only the 62 keep
items enter training. Their original correct-letter distribution is A=3,
B=41, C=17, D=1; this is recorded as provenance, not used as a training
target. Each kept question is split atomically and expanded into four equal
cyclic option rotations, producing an exactly balanced added stream:

| split | questions | expanded records | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|
| train | 53 | 212 | 53 | 53 | 53 | 53 |
| validation | 9 | 36 | 9 | 9 | 9 | 9 |

The source manifest records the hashes, judge threshold, question-level split,
quality metadata, and contamination status. No Spanish benchmark or sealed
test path was read. The raw local-generation, audit-input, and full judge
cache files are ignored as runtime artifacts; the clean source and compact
judge/audit manifests are the reproducibility boundary.

## Isolated training composition

`p21-distractor-quality.yaml` composes the clean source with the strict
v5clean base (`data/train_p6lrr2_strict_clean.jsonl` and its validation
counterpart). The composed dataset is generated immediately before training
and is ignored because it duplicates the base rows:

- train: 4,396 base + 212 distractor-quality = 4,608 records;
- validation: 489 base + 36 distractor-quality = 525 records.

No auxiliary, reflection, verification, general-replay, Spanish, or sealed
evaluation stream is included. This is a single-GPU, serial experiment.

## Evaluation protocol

After training and merge, score the frozen English MCQ primary first. Then run
the ordinary guardrails and the fixed English free-form dev rubric. The
distractor-quality validation slice is a diagnostic for source retention, not
a replacement for the frozen anchor. The candidate is promotable only if it
clears the existing primary improvement threshold and all guardrails; until
then v5clean remains the champion.

Training is monitored at least every 20 minutes for progress, loss/gradient
health, GPU utilization/memory, NaN/OMP/OOM failures, and completion time.

## Result

The isolated run completed normally but was rejected. The candidate primary
was `-2.2282` versus the v5clean anchor `-0.7752` (delta `-1.4531`), and the
answer-bias guard also failed. Free-form dev quality improved modestly, but
that secondary result cannot override the closed-book MCQ anchor. See
`autoresearch/distractor_quality_results.md` for the complete comparison.
