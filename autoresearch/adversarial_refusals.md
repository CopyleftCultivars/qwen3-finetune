# Reviewed adversarial refusals

`data/sources/adversarial_refusals.jsonl` is a small, deliberately authored
source for refusal behavior. It is separate from the natural-farming chat
corpus because the corpus contains incidental prompt-injection-looking rows and
some answers that disclose provenance or hidden-context claims.

Every row must carry:

- `_source: adversarial_refusal`
- `_review_status: approved`
- a stable `_refusal_id` and `_category`

`prepare_dataset_v3.py` validates those fields before adding the rows. The
source is opt-in and can be deterministically capped with
`--adversarial-refusal-count`. Refusal examples should remain concise,
consistent, and non-disclosing; adding this source is a training hypothesis,
not a claim that incidental adversarial rows are automatically beneficial.
