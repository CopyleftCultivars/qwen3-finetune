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

## J-space diagnostic and SAE policy (Phase 5)

- J-space (Jacobian-lens) item signals are the active-loop diagnostic: cheap, item-level, and
  directly actionable. They are proposer guidance for choosing a lever — never a primary
  objective, gate threshold, or score to optimize.
- The diagnostic surface is frozen: `autoresearch/jspace_annotator.py` +
  `autoresearch/jspace_config.json` (manifest-guarded). Artifacts live at
  `results/jspace_item_signals.jsonl` / `results/jspace_category_summary.json` and must be
  regenerated whenever the champion model changes (the lens is model-specific).
- SAE is deferred, not deleted: it becomes relevant only after the loop plateaus, J-space signals
  become ambiguous, or a compatible low-cost Qwen3-0.6B SAE exists and can be used without slowing
  the loop. SAE is not a first-pass autoresearch lever.
- Sealed test splits are never read for lens fitting, annotation, routing, or stream selection.

### J-space routing rules for the proposer (Phase 5, #57)

Bucket evidence routes which lever a proposal targets (mapping frozen in
`jspace_config.json`):

- `absent` → expand category/source data for that category.
- `late_weak` → rationale targets (confirm on the free-form/rubric axis; MCQ can be blind).
- `present_unsurfaced` → rationale or answer-surface experiments (`--sft-format`).
- `format_sensitive` → format / chat-template experiments.
- `position_unstable` → targeted permutation augmentation.
- `robust` → do not target, except replay to protect against forgetting.

Every J-space-guided hypothesis must cite: the artifact path
(`results/jspace_item_signals.jsonl` or `results/jspace_category_summary.json`), the
category and bucket driving it, and the expected primary-metric movement. J-space evidence
picks the lever; the frozen primary objective and gate still decide acceptance.

## Multi-stream batch proposals (Phase 5, #61)

A batch is 2-4 candidate streams run through `run_loop.py --journal-mode batch`
(sequential; at most one accepted stream; losers journaled). A batch must not be N
copies of the same tweak.

Every stream must state, in its spec/hypothesis:

1. one falsifiable hypothesis (what result would refute it);
2. one primary lever (a single `levers.md` row it moves);
3. expected primary-metric movement;
4. guardrail risk (which guardrail is most likely to fire and why that is acceptable);
5. evidence source: a `levers.md` row, a J-space artifact path + bucket, or a prior
   journal row.

Diversity rules:

- No two streams in a batch may change the same primary lever, unless the batch is an
  explicit micro-sweep: `batch_id` prefixed `sweep-` and every hypothesis naming the
  single swept lever.
- The first allowed stream families are: (a) Tier-2 position/order augmentation;
  (b) Tier-1 conservative hyperparameter (lower LR, fewer epochs, LoRA dropout);
  (c) J-space-guided category-data expansion (`--jspace-*` selection flags);
  (d) J-space-guided rationale/answer-surface (`--sft-format`, rationale targets once
  a clean source exists — see chainlink #65).
- The first real batch must include at least one non-hyperparameter stream (satisfied by
  phase5-real-1). Later batches should too, except when the batch's stated purpose is
  hyperparameter/efficiency calibration directed by the human — cite the directive in the
  batch issue.
- A batch may pin a journal-validated setting shared across its streams (cite the journal
  row that validated it); the per-stream primary lever is what differs between streams.
- Failed, rejected, and defeated streams must be interpreted in the next `program.md`
  update or cited in later hypotheses — never silently re-proposed unchanged. Check the
  journal for a stream's prior outcome before re-running its lever.
- Batch mode changes nothing about acceptance: the frozen gate and objective thresholds
  decide, and a guardrail pause blocks the whole batch from auto-accepting.

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
