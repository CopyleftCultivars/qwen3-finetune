# Autoresearch Program

This is the human-edited search manual for the Phase 4 loop. The loop should be
hypothesis-driven: one lever, one smallest viable edit, one measured verdict.

## Editable and frozen surfaces

- The candidate search edit is limited to `autoresearch/finetune.py` parameters/configuration and
  the explicit search dimensions listed in `autoresearch/levers.md`.
- Frozen files are guarded by `autoresearch/frozen_manifest.json`; changing them invalidates the
  run before training starts.
- `autoresearch/program.md` is the human instruction surface. `autoresearch/journal.jsonl` is the
  append-only learning log.
- Sealed test splits must not be read during loop iterations. They are touched exactly once at
  final evaluation.

## Objective and gate rule

- High-frequency primary: MCQ option-set-normalized gold-answer log-likelihood.
- Free-form ref-LL: reported diagnostic only; not a cross-approach comparator unless same model
  family, prompt template, training interface, and answer surface are established.
- Rubric score: hard periodic Goodhart guard.
- Accept a candidate only when the primary delta clears the frozen gate threshold and no guardrail
  fires. Otherwise revert the candidate.
- Keep compute fixed per iteration; do not let longer training masquerade as a discovery.

## Guardrails

- Bias: reject answer-letter marginal drift away from uniform.
- Forgetting: reject general/non-farming score drops beyond threshold.
- Generation quality: reject free-form rubric drops beyond threshold.
- Grader drift: pause for human review if anchor drift exceeds threshold.
- Sealed tests: reject if any sealed test split is read.

## Diagnosis priors from `levers.md`

- Tier 3 data/format levers are higher-ceiling than Tier 2, which are higher-ceiling than Tier 1
  hyperparameters.
- Letter balance is a settled constant. Do not reintroduce answer-position bias.
- Position/order augmentation is the first prioritized experiment: target residual position-spread
  weakness rather than blindly changing all data.
- CoT/rationale targets are pending-unlock: they may help the weak 0.6B internal workspace, but
  require free-form/rubric confirmation because MCQ can be blind to answer quality.
- Every proposal must state: hypothesis, lever tier, prior cited from `levers.md`, smallest edit,
  predicted metric direction, and what result would falsify the hypothesis.

## First experiment

Train-time option-order shuffling augmentation targeting the residual position-spread weakness,
compared against the v3-clean baseline rescored on the dev split.
