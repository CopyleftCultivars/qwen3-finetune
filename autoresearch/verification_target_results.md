# Phase 20 proposed-answer verification result

Status: **rejected; do not promote or combine**.

## Run

- Config: `autoresearch/configs/phase20-verification/p20-proposed-answer-audit.yaml`
- Parent: strict v5clean base (`4,396` train / `489` validation)
- Verification overlay: `728` train / `184` validation
- Composed data: `5,124` train / `673` validation
- Recipe: v5clean settings, four epochs, `1,284` optimizer steps
- Hardware: one host RTX 4070 Ti; training runtime `28m34s`
- Reported training loss: `1.153574`
- Merged model: `outputs/merged-model-p20-proposed-answer-audit`

## Gate results

| Metric | v5clean | Phase 20 candidate | Decision |
|---|---:|---:|---|
| English MCQ option-set primary | `-0.7752` | `-3.0322` | reject: severe regression |
| English MCQ argmax accuracy | — | `30.97%` | diagnostic |
| Verification parse-valid | — | `0/184` | reject: target not learned |
| Verification verdict accuracy | — | `0%` | reject |
| Verification verified-answer accuracy | — | `0%` | reject |
| Verification final-answer accuracy | — | `0%` | reject |
| OpenRouter free-form rubric | `46.0667%` | `45.8333%` | reject: `-0.2333` points |
| Same-lineage reference-LL | `-3.666848/token` | `-3.699113/token` | reported diagnostic |

The candidate generated a bare answer letter rather than the required
`<verification>` ledger on all inspected validation outputs. This means the
new supervision surface was not acquired under this recipe, and the English
MCQ anchor was badly damaged. Keep the audited source and builder for future
analysis, but close this exact dose/target run and leave v5clean as champion.

The free-form comparison used the fixed 90-item English dev subset (18 per
type), the same OpenRouter routine judge/config for baseline and candidate,
and had zero anchor-drift alerts. It was dev-only; no Spanish benchmark or
sealed English test path was read. OpenRouter received the generated English
dev answers for rubric grading, not benchmark test content.

## Artifacts

- Primary report: `results/primary_p20-proposed-answer-audit.json`
- MCQ report: `results/mcq_gold_ll_p20-proposed-answer-audit.json`
- Verification report: `results/verification_target_validation_p20-proposed-answer-audit.json`
- Paired free-form report: `results/freeform_eval_freeform-p20-proposed-answer-audit.json`
- Reproducibility commit: `8daeb79` plus the scorer-report fix commit
