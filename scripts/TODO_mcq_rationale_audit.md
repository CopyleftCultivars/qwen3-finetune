# TODO: Audit MCQ rationale dataset for refusal/low-quality rationales

(Preserved from chainlink issue #1, filed 2026-07-09, before the chainlink DB was reset.)

Priority: medium
Labels: data-quality, mcq-dataset

## Context

While evaluating the v3 fine-tune, we spot-checked
`data/sources/mcq_with_rationale.repaired.jsonl` (output of
`scripts/repair_mcq_rationales.py`, 9456 records after cleanup) for
refusal-style rationales using a crude keyword search ("i cannot", "as an
ai", etc.) across the whole 9459-record file. Of 46 initial hits, only 8
were actually MCQ-formatted (the rest were legitimate non-MCQ conversational
refusal examples, correctly left alone). Of those 8, 5 were false positives
(e.g. "fungi cannot photosynthesize" substring-matches "i cannot") and 3
were genuine broken rationales (idx 8605, 8711, 8949 in the pre-cleanup file
— two psilocybin-cultivation refusals and one source-verification refusal).
Those 3 were removed; backup of the pre-cleanup file kept at
`data/sources/mcq_with_rationale.repaired.pre_cleanup_backup.jsonl`.

v3 was already trained on the un-cleaned file, so this fix only benefits a
future v4 run.

## Open follow-up

The keyword search was crude and untargeted at MCQ structure specifically,
so it's plausible there are other quality issues in the ~9456 MCQ rationales
that this pass didn't catch (e.g. hallucinated citations like idx 8949's
book-chapter claim, incomplete reasoning, wrong-answer-letter mismatches).
Worth a more thorough pass before v4 training — possibly via
`scripts/judge_training_data.py` (already exists) or a dedicated LLM-judge
sweep over all MCQ-formatted records rather than a keyword grep. Deferred
for now.
