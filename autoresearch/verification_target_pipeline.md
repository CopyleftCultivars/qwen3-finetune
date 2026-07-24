# Phase 20: proposed-answer verification target

This phase tests one new reasoning/verification supervision surface. It does
not mix any previously tested lever.

## Target design

The source is the 228-record, judge-approved English QMD MCQ pool in
`data/sources/qmd_mcq_generated.jsonl`. It is not the 311-record rationale
replay source. Each source question emits four records, one for each proposed
answer letter. The model must audit the proposal and return a structured
verification ledger plus the settled bare final letter:

```text
<verification>
proposed_answer: B
verdict: REJECT
verified_answer: C
verification_basis: ...
</verification>
C
```

This is a proposed-answer audit task, not a direct MCQ rationale, interrupted
reflection, self-critique, or answer-first output-format experiment. Proposed
letters are exactly balanced A/B/C/D across the full source. The verdict mix
is explicit: 228 ACCEPT and 684 REJECT records.

## Audits and composition

The source builder rejects duplicate IDs, malformed options, non-approved
records, and exact English benchmark question/full-option overlap. The current
audit passed all 228 source questions with zero benchmark overlap and zero
exact instruction overlap with the strict v5clean base. Source questions are
split atomically so variants cannot cross train/validation.

```bash
python scripts/build_verification_target.py \
  --source data/sources/qmd_mcq_generated.jsonl \
  --benchmark [LOCAL_PATH] \
  --base-train data/train_p6lrr2_strict_clean.jsonl \
  --base-validation data/val_p6lrr2_strict_clean.jsonl \
  --output-dir data/verification_target/proposed_answer_audit
```

The prepared source contains 728 train and 184 validation rows. The
composed, reproducible v5clean-plus-verification set contains 5,124 train and
673 validation rows. The composition is rebuilt by `autoresearch/finetune.py`
immediately before training and remains ignored; the audited source and
manifest are tracked.

## Training and evaluation

Run only this config on the single GPU:

```bash
python autoresearch/finetune.py \
  --config autoresearch/configs/phase20-verification/p20-proposed-answer-audit.yaml
```

Monitor training every 20 minutes. After merge, score the frozen English MCQ
primary first, then the verification validation task, and then the paired
English free-form rubric on the fixed dev IDs. Spanish and sealed English
tests remain untouched. The verification task is a reported diagnostic; the
English MCQ primary and existing guardrails decide promotion.

## Phase 20 result

The single isolated run was rejected. It used the v5clean recipe unchanged
apart from adding the composed verification stream, trained for four epochs
and 1,284 optimizer steps on one RTX 4070 Ti (28m34s; reported training loss
1.1536). The merged candidate produced:

| Metric | v5clean anchor | Proposed-answer audit |
|---|---:|---:|
| English MCQ primary | -0.7752 | **-3.0322** |
| English MCQ argmax accuracy | — | 30.97% |
| Verification validation parse-valid | — | **0/184 (0%)** |
| Verification verdict / verified-answer / final-answer accuracy | — | **0% / 0% / 0%** |
| OpenRouter free-form rubric | 46.0667% | 45.8333% |
| Free-form rubric delta | — | -0.2333 points |
| Same-lineage reference-LL | -3.6668/token | -3.6991/token |

The candidate emitted bare letters on the verification validation prompts,
so it did not learn the structured ledger target at this recipe. It also
failed the frozen English primary by a wide margin; it is not a promotion
candidate and must not be combined with another lever. The paired rubric used
the fixed 90-item English dev subset, the same OpenRouter judge/config for
both models, and recorded zero anchor drift. No Spanish or sealed test path
was read. Raw runtime evaluation artifacts remain ignored; the compact
result record is `autoresearch/verification_target_results.md`.
