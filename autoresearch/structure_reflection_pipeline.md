# Multi-turn and reflection lever protocol

This phase follows the CROP-derived recommendation to preserve conversational
structure and measure reflection targets independently. It uses the English
natural-farming ShareGPT source and the strict v5clean base. Spanish data is
not included, and the sealed Spanish benchmark and sealed English free-form
test are never read.

## Source and controls

The source audit found 180 eligible multi-turn threads after removing
persona-only openings, malformed records, and non-alternating turns. Each
stream uses the same deterministic 140 train / 40 validation thread dose, a
0.78 source split, and seed 42. The preserved stream keeps a bounded recent
window of four user/assistant pairs so it fits the 2048-token training surface.

The flattened control uses the same source IDs and final answers but only the
last user/assistant pair. This isolates conversation structure from source
content. There were no explicit step-back labels in the source, so this phase
does not claim to test a native step-back corpus; a separately authored
step-back stream is a later lever.

## Independent streams

- `multiturn`: preserved recent multi-turn messages;
- `flattened_control`: same threads flattened to the final pair;
- `interrupted_counterfactual`: a deterministic interrupted draft plus an
  explicit audit of an unsupported universal claim and corrected answer;
- `answer_audit`: a deterministic incomplete draft followed by an audit and
  corrected answer.

Every row contains messages, a local validation prompt, a target output,
source record hashes, authoring method, reflection error type where relevant,
and English MCQ contamination status. No QMD benchmark rationale replay is
used.

## Serial execution and scoring

Run one config at a time on the single GPU:

```bash
python autoresearch/finetune.py --config autoresearch/configs/phase18-structure-reflection/p18-multiturn.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase18-structure-reflection/p18-flattened-control.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase18-structure-reflection/p18-interrupted-counterfactual.yaml
python autoresearch/finetune.py --config autoresearch/configs/phase18-structure-reflection/p18-answer-audit.yaml
```

After each merge, score the task's held-out validation stream with
`scripts/score_multiturn_reflection_validation.py`, then run the frozen MCQ
primary and the existing local guard checks. External free-form rubric
grading is not part of this run because it would transmit generated answers
to a judge without separate authorization.

Reflection proportion is a follow-up sweep, not a simultaneous combination.
Only the best individual reflection format can advance to a token-matched
5% / 10% / 20% proportion test.

## Phase 18 result

The individual pilots are complete. Every candidate improved its local native
validation score, but every candidate regressed the frozen English MCQ anchor
relative to v5clean (`-0.7752`). The best closed-book candidate was preserved
multi-turn at `-1.7036`; it remained `-0.9284` below the anchor. The strongest
native reflection gain was answer-audit, whose token-F1 rose from `0.4185` to
`0.5993`, but its MCQ primary fell to `-1.7892`.

Because no individual format passed the no-English-regression gate, the
token-matched reflection-proportion sweep is locked pending a materially
different recipe. The phase also does not claim a step-back result: the source
audit found no explicit step-back examples.

See `autoresearch/structure_reflection_results.md` for the complete table and
disposition.
