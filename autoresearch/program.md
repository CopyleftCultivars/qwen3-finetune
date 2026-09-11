# Autoresearch Program

This is the human-edited search manual for the Phase 4 loop. The loop should be
hypothesis-driven: one lever, one smallest viable edit, one measured verdict.

## Deployed (on-device) champion

- **distill-v2-rationale, calibrated conversion** (`outputs/merged-model-distill-v2-rationale`,
  trained in F7 on the F6 dataset: 871 gold-filtered qwen3:8b teacher rationales, v5clean's
  hyperparameters) replaces v5clean as the model shipped in the Natural Farming Chat app, per
  chainlink #163 (decided 2026-09-08) and deployed in #169. This is a **deployment-track**
  promotion, separate from the GPU autoresearch-loop's lever-search promotions below: it did not
  go through the rubric/primary-objective gate (F5-F9 is its own distillation lever, milestone
  #6 M2), and it is judged on the on-device MCQ ladder rather than the loop's usual criteria.
- Promotion basis: GPU-side, distill-v2-rationale scored 0.8714 on the 210-item ladder vs
  v5clean's 0.7810 (+9.0pp, `results/F8_DISTILLATION_RESULTS.md`), but that edge did not survive
  quantized on-device deployment at first measurement (statistically indistinguishable from
  v5clean, `results/F9_ON_DEVICE_LADDER_RESULTS.md`). A calibrated GPTQ-Hessian conversion
  (chainlink #165) recovered most of the gap: calibrated distill scores 0.8381 on-device vs
  calibrated v5clean's 0.7810 (+5.7pp, two-proportion z=1.49). The residual -3.3pp vs its own GPU
  score is attributed to irreducible INT4 precision loss (ruled out termination/verbosity via
  #166/#167's constrained-decoding check). Root-cause analysis: #164,
  `results/QUANT_SENSITIVITY_ROOTCAUSE_164.md`.
- **v5clean remains the base/champion for the GPU autoresearch training-lever loop** (see below)
  -- this deployment promotion does not reset the rubric sample-size growth policy or the lever
  search's champion lineage. Future lever-search candidates are still compared against v5clean
  until a lever-search promotion (not a distillation/deployment promotion) supersedes it.

## Current champion (GPU autoresearch training-lever loop)

- **v5clean** (`outputs/merged-model-p6-lr5e5-r2`) supersedes v4clean as of chainlink #90,
  2026-07-21. Settings: v4clean's mix, `epochs=4`, `lora_dropout=0.025`, **`lr=5e-5`** (see
  `autoresearch/configs/v5-clean-baseline.yaml`). Promoted on chainlink #86's repeat-confirmed
  result: primary delta +1.478 +/- 0.071 SD (n=4 identical runs — an order of magnitude tighter
  than any other lever's run noise measured in this program, on an effect ~20x that noise
  floor), rubric IMPROVED in every run (never regressed), forgetting/anchor/bias/compute/sealed
  clean throughout. The strongest, cleanest confirmed result the program has produced.
- New champion-reference artifacts: `results/primary_objective_v5_clean.json`,
  `results/mcq_gold_ll_v5_clean.json`, `results/general_knowledge_v5_clean.json`. New rubric
  baseline: `codexv5` label (`results/periodic_rubric_baseline_codexv5.jsonl`, mean 60.8% at
  items-per-type=15 per the [[rubric sample-size growth policy]] below, graded under the codex
  judge backend). J-space lens/annotations regenerated against v5clean
  (`autoresearch/jspace_config.json`, `results/jspace_category_summary.json`,
  `results/jspace_item_signals.jsonl`) — lens SHA256 changed (`4570a349...` → `89b0805...`),
  confirming a genuine refit, not a stale-checkpoint no-op.
- v4clean's history (below) is preserved as the record of how the prior champion was reached;
  it is superseded, not deleted. Future streams should set `--baseline-label codexv5` (not
  `codexv4`) and reference the v5clean baseline files above.

### Prior champion: v4clean (superseded 2026-07-21, see #90)

- **v4clean** (`outputs/merged-model-p5real6-d025-r2`) supersedes v3-clean as of chainlink #85,
  2026-07-20. Settings: v3-clean's mix, `epochs=4`, `lora_dropout=0.025` (see
  `autoresearch/configs/v4-clean-baseline.yaml`). Promoted on chainlink #84's repeat-confirmed
  result: primary delta +0.519 +/- 0.149 SD (n=4 identical runs), rubric drop 2.85 +/- 0.58pp
  (all 4 comfortably under the 5.0pp limit — unlike the falsified `lora_dropout=0.01` point,
  see `levers.md`), forgetting/anchor/bias/compute/sealed clean throughout.
- Champion-reference artifacts: `results/primary_objective_v4_clean.json`,
  `results/mcq_gold_ll_v4_clean.json`, `results/general_knowledge_v4_clean.json`. Rubric
  baseline: `codexv4` label (`results/periodic_rubric_baseline_codexv4.jsonl`, mean 53.17% at
  items-per-type=12, graded under the codex judge backend).
- v3-clean's history (below) is preserved as the record of how that prior champion was reached;
  it is superseded, not deleted.

## Rubric sample-size growth policy (human-directed, 2026-07-21)

- The rubric guard's `--items-per-type` grew once already under pressure: 6 (30 items) was
  proven too noisy (chainlink #82 — a 2/4 pass rate on `lora_dropout=0.01` reversed to 0/4 fail
  at higher precision), so v4clean's `codexv4` baseline was built at 12 (60 items). The dev pool
  (`eval_freeform/dev.jsonl`) has a hard ceiling of 18/type (90 items total) — there is no more
  headroom without authoring new dev items, which is out of scope for the loop itself.
- Going forward, **every successful champion promotion also grows the rubric sample size by
  +3/type**, up to that ceiling: v4clean is baselined at 12/type; the *next* promoted champion's
  baseline must be built at 15/type; the one after that at 18/type (the ceiling — stays there
  for all subsequent promotions once reached).
- Rationale: a promotion already requires building a fresh judge baseline for the new champion
  (new model → new reference generations → new grading), so bumping N at that exact moment adds
  no extra re-baseline event — it rides the one that would happen anyway. Growing preemptively
  mid-tournament, by contrast, would invalidate whatever comparison is currently in flight against
  the existing baseline for no such win (see chainlink #86's LR-sweep stream, which correctly
  stays at 12/type against `codexv4` — this policy does not apply retroactively or mid-flight).
- Do not skip a step: always exactly +3/type per promotion until the 18/type ceiling, not a
  jump straight to 18. This keeps each re-baseline small relative to the one before it, and
  gives another checkpoint to notice if noise is still a problem before committing to the max.

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

### Trace-conditioned parallel track (Epic E, chainlink #105, 2026-09-01)

- The primary above scores `P(gold letter | forced-empty-<think>)`, which removes reasoning
  from the scored sequence entirely -- J-space then has no verbalizable content to read.
  `autoresearch/score_mcq_trace_ll.py` (chainlink #106-108 / I1-I3) restores a reasoning surface:
  Pass 1 generates a real trace (think-on, greedy, fixed seed), Pass 2 scores the gold letter
  conditioned on that trace (think-off equivalent, closed `<think>` block).
- This is a **new, NOT-comparable objective** (`autoresearch/objective_weights_v3.json`,
  chainlink #109 / I4) running as a **PARALLEL track** -- it does not replace the primary above,
  and never will without an explicit promotion decision recorded in both files.
- Champion (v5clean) trace-LL baseline: `results/mcq_trace_ll_v5_clean.json` /
  `.jsonl` (chainlink #109 / I4). Measured 2026-09-01 on the full 113 scorable rows in
  `data/val_v3.jsonl`: `mean_trace_option_set_gold_logprob=-0.535`, `mean_delta_trace=+0.167`,
  `delta_trace_positive_fraction=66.4%` (75/113) -- clears the "clear majority positive" half
  of the step 8 gate. Trace-conditioning also lifted argmax accuracy (78.8% vs. 71.7%
  empty-trace) and truncation was negligible (1/113 items hit `max_trace_tokens=256`).
- Promotion requires (a) `delta_trace > 0` on a clear majority of items and (b) the free-form
  rubric guard staying clean (Plan A step 8, `TRACE_EVAL_REASONING_SFT_RLM_PLAN.md`). Phase24
  (`outputs/merged-model-p24-trace-sft`) was evaluated on 2026-09-01/02 (chainlink #115 / I10):
  trace primary delta was `-0.187` versus v5clean, mean `delta_trace=-0.041`, and only 46.9%
  (53/113) of items had positive delta. Its rubric guard stayed clean (59.79% versus 60.80%,
  a 1.01-point drop), and the general-knowledge probe improved (45.83% versus 41.67%), but the
  required trace-benefit majority failed. The standard primary delta was only `+0.078`, below
  the existing `+0.4` gate. Decision: reject Phase24, keep v5clean champion, and leave this
  objective as a non-gating parallel track. Evidence: `results/autoresearch_gate_p24-trace-sft.json`.

## Guardrails

- Bias (relative rule, human-directed 2026-07-18): the guard prevents *reintroducing*
  letter bias. A candidate whose predicted-letter deviation from uniform is <=0.10 always
  passes; otherwise it must not exceed the champion baseline's deviation plus a 0.05 noise
  tolerance (~1.2 SE at n=113). Context: v3-clean measured 0.272 deviation (A=0.522), so
  absolute-uniform gating rejected candidates that improved balance; v4clean measured 0.219
  (A=0.469) -- an improvement. Current champion v5clean measures 0.060 deviation (C=0.310) --
  the most balanced champion yet -- and is the baseline future candidates are compared against
  as of chainlink #90.
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
- Position/order augmentation was the first prioritized Tier-2 experiment. The targeted
  17-record QMD overlap dose and a stronger multiplier-3 follow-up both rejected; do not
  silently reopen this exact replay surface.
- CoT/rationale targets are pending-unlock: they may help the weak 0.6B internal workspace, but
  require free-form/rubric confirmation because MCQ can be blind to answer quality.
- Every proposal must state: hypothesis, lever tier, prior cited from `levers.md`, smallest edit,
  predicted metric direction, and what result would falsify the hypothesis.

## First experiment

Train-time option-order shuffling augmentation targeting the residual position-spread weakness,
compared against the v3-clean baseline rescored on the dev split.
