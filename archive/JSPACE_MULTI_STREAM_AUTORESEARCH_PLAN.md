# Plan: J-space-guided + parallel multi-stream autoresearch extensions

Date: 2026-07-17

## 0. Current state and why this is new scope

The original Phase 2.6/2.7/4 autoresearch infrastructure is complete as infrastructure:

- Phase 2.6 free-form eval exists: canonical items, dev/test split, contamination wall, scorer config, rubric scorer, ref-LL scorer, anchors.
- Phase 2.7 lever inventory exists: `autoresearch/levers.md` is the action-space source of truth.
- C1 frozen harness exists: `autoresearch/finetune.py`, configs, frozen manifest, and checksum gate.
- C2 objective policy is settled: MCQ option-set LL is the climbable primary; free-form ref-LL is diagnostic-only because it is not an apples-to-apples cross-approach comparator.
- C3/C4 gate and guardrails exist: `autoresearch/gate.py`, `gate_config.json`, and 8 self-test fixtures pass.
- C5 loop scaffold exists: `autoresearch/run_loop.py`, `program.md`, and `journal.jsonl` smoke path.
- Chainlink epic #14 and blocker chain #49/#35/#23/#24/#25 are closed; only deferred #9/#10 remain open.

Therefore this plan is **not** remaining infrastructure repair. It is the next research-extension scope:

1. add J-space-guided item/category routing;
2. add parallel multi-stream proposal/evaluation;
3. document SAE as deferred/off-critical-path, not part of the active loop.

The only caveat on the completed loop is that the last loop journal row was a dry-run smoke. The first real non-dry-run augmentation experiment is still the first use of the loop, not a missing infrastructure component.

## 0.1 Local source inventory now available

The user has added the two external sources this phase depends on:

- `jacobian-lens/` — local clone of the companion repo for **"Verbalizable Representations Form a
  Global Workspace in Language Models"**. The repo README defines the Jacobian lens as a readout
  that linearly transports residual-stream vectors into the final-layer basis and decodes them with
  the model's own unembedding. Important local APIs:
  - `jlens.from_hf(hf_model, tokenizer)` wraps a loaded HuggingFace decoder.
  - `jlens.fit(model, prompts=..., checkpoint_path=...)` fits a Jacobian lens.
  - `jlens.JacobianLens.load(...)` / `.from_pretrained(...)` loads a fitted lens.
  - `lens.apply(model, prompt, layers=..., positions=...)` returns per-layer lens logits.
  - `jlens.vis._ranks_of(...)` is a useful tested helper for full-vocab target-token ranks.
  - README caveat: this is a **reference implementation**, not maintained/optimized. Fit cost is
    dominated by backward passes; ~100 prompts is described as usable, while paper-scale examples
    used ~1000 sequences of 128 tokens.
- `2605.12460v1.pdf` — **"Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of
  Thoughts, Inputs and Outputs"**. The relevant implementation details are:
  - Multi-stream models split roles into separate streams and predict tokens across multiple output
    streams in each forward pass.
  - Data construction has three stages: stream-like generation, causal verification, and quality
    filtering.
  - Wait-k generation lets assistant streams begin after observing only a prefix of the user stream.
  - Stream-table generation asks a capable model to produce all streams row-by-row to avoid
    non-causal leakage.
  - True model-level multi-stream training requires stream-aware position encoding, stream identity
    embeddings, a cross-stream causal mask, and preferably interleaved packing.
  - Inference is synchronous multi-stream decoding: each step emits one token per stream, with empty
    slots masked.

Important interpretation for this repo: the existing autoresearch harness can support **loop-level
multi-stream tournaments immediately**, but the paper's **true model-level multi-stream architecture**
is a larger training/inference change that should be planned separately behind a surrogate/data-format
prototype.

## 0.2 Paper-grounded interpretation for this repo

The two sources should change the Phase 5 implementation, but not by widening the active loop
uncontrollably.

From the J-space paper/repo:

- The Jacobian lens is not just a standard logit lens. It uses an averaged linearized effect from
  intermediate residual-stream activations into the final-layer basis, then decodes with the model's
  own unembedding. This is why J1 should fit/load a lens rather than only reuse the older B3
  logit-lens probes.
- The useful signal is **verbalizable latent content**: what the model is disposed to say or reason
  with, not merely the next token it is about to emit.
- Early layers can be noisy and late layers can collapse toward the imminent output/motor regime.
  J1 should therefore record layer-wise trajectories rather than a single top-layer number.
- The published framing treats J-space as sparse and capacity-limited. For this project, that means
  the artifact should prefer top-k ranks/margins and bucketed routing over dense, high-dimensional
  feature dumps.
- The method is limited by tokenization: single-token concepts are easiest. Multi-token agronomy
  concepts should be deferred until answer-letter and explicit concept-token routing work.
- The counterfactual-reflection result is directly relevant to autoresearch: training on what the
  model would say if interrupted to reflect can shape later behavior. That makes rationale/reflection
  targets a legitimate Phase 5 candidate stream, but only after the gate and rubric guardrails remain
  clean.

From the multi-stream paper:

- The paper's central claim is not "run several shell jobs." It is that LLMs can be instruction-tuned
  to process separate token streams for roles such as system, user, thinking, tool, reflection, and
  response.
- True model-level multi-stream requires at least stream identity, stream-aware positions, a
  cross-stream causal mask, and synchronous decoding. That is outside the current frozen
  `finetune.py` loop surface.
- The immediately useful adaptation is a **loop-level tournament**: several candidate streams run
  against the same frozen baseline and gate, with at most one accepted winner.
- A second, later adaptation is a **single-stream surrogate** for multi-stream data: stream-table
  examples collapsed into ordinary chat/SFT records so we can test whether separated reasoning,
  audit, monitor, or response roles help before touching architecture.
- The paper's data pipeline gives the right guardrails for any future surrogate data: stream-like
  generation, causal verification, and quality filtering. Do not generate unverified "parallel
  thoughts" and feed them straight into the loop.

## 1. Decision: prefer J-space over SAE for the active loop

### Recommendation

Use **J-space / logit-lens / residual-stream projection diagnostics** as the next active-loop guidance mechanism. **Do not put SAE training in the automated loop.** SAE should be documented as a deferred interpretability escalation path, not deleted forever.

### Reasoning

The autoresearch loop is constrained by fixed, fast iteration: approximately 5–20 minutes per candidate at Qwen3-0.6B + LoRA scale. A useful diagnostic must produce signals that can be generated cheaply, attached to items/categories, and converted into immediate training levers.

J-space fits this requirement:

- **No extra model training.** It reuses hidden states and projects them through existing output/readout structure.
- **Per-item actionability.** It can classify failures as absent, weak/late, present-but-unsurfaced, format-sensitive, position-unstable, or robust.
- **Direct lever mapping.** Each class maps naturally to an autoresearch action: add data, add rationale targets, change answer surface, adjust replay, or apply permutation augmentation.
- **Compatible with existing diagnostics.** The project already has logit-lens/first-correct-layer/category-failure artifacts from B3, so this is an incremental extension rather than a new interpretability project.
- **Good fit for small models.** At 0.6B, the primary bottleneck is finding cheap, actionable steering signals; J-space gives those signals without consuming the loop budget.

SAE does not fit the active loop right now:

- **High setup cost.** SAE requires collecting activation corpora, training dictionaries, validating sparsity/faithfulness, and interpreting features.
- **Interpretation bottleneck.** Learned features must be labelled and mapped back to agronomy/reasoning interventions before the loop can use them.
- **Compute mismatch.** Training or maintaining SAEs is hours/days-scale work, not 5–20-minute loop work.
- **Uncertain immediate payoff.** Even a good SAE does not automatically tell `finetune.py` which lever to try next.

Decision wording to carry forward:

> J-space is the Phase 5 active-loop diagnostic because it is cheap, item-level, and directly actionable. SAE is deferred until the loop plateaus, J-space signals become ambiguous, or a compatible Qwen3-0.6B SAE exists and can be used without slowing the loop. SAE is not a first-pass autoresearch lever.

## 2. J-space feature plan

### J1 — Build a J-space signal artifact

Add a diagnostic script, proposed path:

- `autoresearch/jspace_annotator.py`

Use the local `jacobian-lens/` repo as the implementation source. Prefer importing it by adding
`jacobian-lens/` to `sys.path` rather than vendoring its code into this repo.

Inputs:

- model path, usually `outputs/merged-model-v3-clean` or current loop champion;
- lens path, if already fitted, for example `results/jspace/jacobian_lens.pt`;
- fitting prompts, if no lens path exists; start with held-in/dev-visible sources only and never use
  sealed test content;
- eval/dev items and/or MCQ dev records;
- optional category labels and position-spread artifacts from prior diagnostics;
- a small config listing target answer tokens/concepts, layers, prompt mode, and category mapping.

Outputs:

- `results/jspace_item_signals.jsonl`
- `results/jspace_category_summary.json`
- optional visual/debug artifacts under `results/jspace/slices/`, never required by the loop.

Each item row should include:

- `item_id` / question id;
- source split and category/type;
- current correctness or score if available;
- top-layer answer prediction;
- earliest layer where the gold answer/concept appears in the top-k projection;
- signal strength / margin;
- classification bucket:
  - `robust` — correct signal appears and remains stable;
  - `absent` — correct answer/concept never appears strongly;
  - `late_weak` — signal appears only late or with weak margin;
  - `present_unsurfaced` — signal appears mid-layer but is overridden by final prediction;
  - `format_sensitive` — signal differs sharply across prompt/answer surfaces;
  - `position_unstable` — overlaps existing permutation instability bucket;
- recommended intervention:
  - `add_category_data`;
  - `add_rationale_targets`;
  - `adjust_answer_surface`;
  - `targeted_permutation`;
  - `increase_replay`;
  - `do_not_target`.

Implementation details:

- Load model/tokenizer with the same env convention as diagnostics: `conda run -n qwen3-diagnostics`.
- Wrap the HF model with `jlens.from_hf(...)`.
- If a lens file exists, load it with `jlens.JacobianLens.load(...)`; otherwise fit with
  `jlens.fit(...)` on a bounded prompt set and save to `results/jspace/jacobian_lens.pt`.
- For the first implementation, use **answer-letter token ranks** for MCQ rows before trying to read
  latent agronomy concepts. This gives a deterministic, easy-to-validate bridge from the current
  option-set LL objective to J-space routing.
- Compute, per item, ranks/margins for the gold answer token and the model's final predicted answer
  across selected layers. The existing `jlens.vis._ranks_of` helper can be copied as an implementation
  pattern if direct import is inconvenient.
- For free-form items, begin with a small explicit concept-token list per item/type only after MCQ
  answer-letter routing works. Do not make open-ended semantic concept extraction part of J1.

Acceptance:

- Script runs on a small subset first and then the full dev set.
- Output is deterministic for fixed model/config.
- At least 95% of dev items receive a bucket.
- Category summary highlights Natural Fertilizers, Sustainable Water Management, and Soil Science separately, because prior B3 diagnostics identified them as priority categories.
- No sealed test split is read.
- If fitting is required, the fit command records prompt count, source layers, max sequence length,
  and lens file hash.

#### Chainlink scaffolding prompt for #55 / J1

```text
Implement the first J-space/Jacobian-lens artifact for qwen3-finetune.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md, sections 0.1, 1, and J1.
- jacobian-lens/README.md and jacobian-lens/jlens/{hf.py,lens.py,fitting.py,vis.py}.
- autoresearch/score_mcq_ll.py for MCQ normalization conventions.
- autoresearch/freeze_manifest.py for sealed-test/frozen-file discipline.

Task:
Create autoresearch/jspace_annotator.py. It should load a fitted lens from
--lens-path or fit one to a bounded prompt set when --fit-if-missing is passed. Use the local
jacobian-lens repo via sys.path; do not vendor it. For v1, focus on MCQ answer-letter token ranks:
for each scorable MCQ dev row, compute per-layer rank/margin of the gold answer letter and final
predicted letter, then bucket the item as robust / absent / late_weak / present_unsurfaced /
format_sensitive / position_unstable where enough evidence is available. Emit:
results/jspace_item_signals.jsonl and results/jspace_category_summary.json.

Constraints:
- Never read eval_freeform/test.jsonl or any sealed test path.
- Use conda env qwen3-diagnostics for real model/lens work.
- Deterministic output for fixed model/config/lens.
- If fitting a lens, record prompt count, source layers, max_seq_len, and sha256 of the lens file.
- Keep this diagnostic as proposer guidance only; do not alter objective weights or gate logic.

Acceptance:
- Small-subset smoke works.
- Full dev run buckets >=95% of scorable MCQ dev rows.
- Manifest check still passes.
- Closing comment reports artifact paths, bucket counts, lens hash, and confirms sealed tests untouched.
```

### J2 — Add J-space config and freeze policy

Add a config file after J1 stabilizes, proposed path:

- `autoresearch/jspace_config.json`

Config should pin:

- prompt mode;
- layer range;
- top-k threshold;
- bucket thresholds;
- category mapping;
- allowed interventions.

Freeze policy:

- Before J-space affects gate decisions or proposer priors, add the script/config to `autoresearch/freeze_manifest.py` and regenerate `frozen_manifest.json`.
- If J-space is used only to propose candidate configs, it can be treated as a frozen diagnostic input, not a climbable objective.

Acceptance:

- Manifest check passes.
- Intentional edit to the J-space config trips the checksum gate once frozen.

#### Chainlink scaffolding prompt for #56 / J2

```text
Freeze the J-space diagnostic surface and document SAE deferral.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections 0.2, 1, J1, and J2.
- The artifacts produced by #55.
- autoresearch/freeze_manifest.py and autoresearch/frozen_manifest.json.
- autoresearch/program.md and autoresearch/levers.md for where SAE/J-space policy should be
  summarized.

Task:
Add autoresearch/jspace_config.json after #55 stabilizes the artifact schema. Pin prompt mode,
layers, top-k/rank thresholds, bucket thresholds, category mapping, allowed interventions, and the
sealed-test exclusion policy. Add the J-space script/config to the frozen manifest. Document that
SAE is deferred, not deleted: SAE becomes relevant only after the loop plateaus, J-space signals are
ambiguous, or a compatible Qwen3-0.6B SAE is available with low setup cost.

Constraints:
- Do not let J-space become a primary objective or gate threshold.
- Do not freeze volatile result artifacts; freeze code/config only.
- Preserve existing manifest behavior and current frozen file checks.

Acceptance:
- Manifest check passes.
- A deliberate one-line config tamper causes manifest mismatch, then passes again after restore.
- Closing comment records frozen paths, config policy, and SAE deferral wording.
```

### J3 — Update `program.md` and `levers.md`

After J1/J2, promote `J-space-informed data selection` from `PENDING-UNLOCK` to an active proposer prior.

Important: do not immediately expose J-space as a broad free-form `finetune.py` flag. Start as a proposer-side selection prior and add narrow flags only when a specific intervention is proven useful.

Recommended mapping:

- `absent` → category/source data expansion or replay of related factual examples;
- `late_weak` → rationale/step-by-step targets in the same category;
- `present_unsurfaced` → CoT/rationale or answer-surface experiments;
- `format_sensitive` → SFT target format / chat template / answer surface experiments;
- `position_unstable` → targeted permutation augmentation;
- `robust` → avoid spending loop budget unless needed for replay/forgetting balance.

Acceptance:

- `program.md` tells the proposer how to cite J-space evidence in hypotheses.
- `levers.md` documents J-space as a routing prior, not an uncontrolled search dimension.
- Frozen manifest is regenerated if either file is frozen at that point.

#### Chainlink scaffolding prompt for #57 / J3

```text
Update the autoresearch proposer instructions with J-space routing priors.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections 0.2, J3, and section 4.
- autoresearch/program.md and autoresearch/levers.md.
- results/jspace_item_signals.jsonl and results/jspace_category_summary.json from #55, if present.

Task:
Promote J-space-informed data selection from PENDING-UNLOCK to an active proposer prior only after
#55/#56 are done. Add concise rules telling the proposer how to use bucket evidence:
absent -> category/source data expansion; late_weak -> rationale targets; present_unsurfaced ->
rationale or answer-surface experiments; format_sensitive -> target format/chat-template
experiments; position_unstable -> targeted permutation; robust -> avoid targeting except replay.

Constraints:
- J-space is evidence for choosing a lever, not a score to optimize.
- Every J-space-guided hypothesis must cite artifact path, category/bucket, and expected metric
  movement.
- Do not add broad unbounded knobs to finetune.py in this issue.

Acceptance:
- program.md contains J-space citation rules.
- levers.md contains the bucket-to-intervention mapping.
- Existing single-stream smoke path remains valid.
```

### J4 — Optional narrow training hooks

Only after J-space buckets are validated, add narrow data-selection hooks to the training pipeline.

Possible flags:

- `--jspace-signals PATH`
- `--jspace-bucket absent|late_weak|present_unsurfaced|format_sensitive|position_unstable`
- `--jspace-category CATEGORY`
- `--jspace-max-items N`
- `--jspace-action include|exclude|cap|rationale_overlay`

Constraints:

- No upsampling by default unless explicitly justified; prefer deterministic subset/cap behavior first.
- Preserve letter balance and contamination exclusions.
- Keep all selected item ids in the run artifact for audit.

Acceptance:

- Dry-run prints exactly which buckets/categories/actions would be used.
- Selection is deterministic under fixed seed.
- Existing baseline config is unchanged when no J-space flags are passed.

#### Chainlink scaffolding prompt for #58 / J4

```text
Add optional narrow J-space data-selection hooks.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections J1-J4.
- autoresearch/finetune.py and any config/data-loading code it calls.
- The frozen manifest policy from #56.

Task:
Add minimal opt-in data-selection flags for J-space-guided experiments:
--jspace-signals, --jspace-bucket, --jspace-category, --jspace-max-items, and --jspace-action.
Start with deterministic include/exclude/cap behavior. Add rationale_overlay only if the existing
data path can support it without changing objective/gate semantics.

Constraints:
- Baseline behavior must be unchanged when no J-space flags are passed.
- Preserve letter balance and contamination exclusions.
- Emit selected item ids and bucket/category counts in the run artifact.
- No sealed test reads.

Acceptance:
- Dry-run shows exact selected buckets/categories/item ids.
- Fixed seed gives deterministic selection.
- Manifest check passes, and frozen files are updated if needed.
```

## 3. Parallel multi-stream feature plan

### 3.0 Interpret the paper at two levels

The paper describes **model-level multi-stream LLMs**: separate role streams, cross-stream causal
attention, stream-aware positions, stream identity embeddings, interleaved packing, and synchronous
decoding. That is not the same thing as running several ordinary finetune candidates in parallel.

This repo should implement the idea in two stages:

1. **Loop-level multi-stream tournaments (now):** multiple candidate hypotheses/configs are proposed
   and evaluated in parallel or sequentially against the same frozen gate. This is immediately
   compatible with the current harness and is the right first implementation.
2. **Model/data-level multi-stream surrogate (later):** before changing Qwen's architecture, create
   stream-table training data and collapsed/surrogate targets that test whether separated streams
   (answer, reasoning, audit, monitor) improve quality under the existing single-stream model.
   Only if this wins should the project consider true architecture changes.

Do **not** jump directly to changing attention masks/RoPE/stream embeddings inside the active
autoresearch loop. That would violate the current `finetune.py`-only search surface and turn Phase 5
into a model-architecture project.

### M1 — Define multi-stream semantics, journal schema, and fixtures

Parallel multi-stream means a single research cycle proposes multiple candidate streams, evaluates each against the same frozen baseline, and accepts at most one winner.

Start with tournament semantics:

- one baseline champion;
- N candidate streams, default 2–4;
- each stream has one hypothesis and one smallest viable lever edit/config;
- each stream runs under the same fixed budget;
- each stream is gated independently;
- if multiple streams pass, accept the highest primary score;
- if none pass, reject all;
- all streams are journaled, including losers.

Do not allow streams to mutate frozen surfaces, objective weights, scorer code, sealed tests, or gate thresholds.

#### Chainlink scaffolding prompt for #59 / M1

```text
Define the multi-stream journal schema and dry-run fixtures.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections 0.1 and 3.
- 2605.12460v1.pdf section 3.1-3.4 only for concepts; do NOT implement model-level stream
  attention yet.
- autoresearch/run_loop.py and the current autoresearch/journal.jsonl row.
- autoresearch/gate.py self-test fixtures.

Task:
Add a documented batch journal schema for loop-level multi-stream tournaments. Backwards
compatibility is required: existing single-stream journal rows must remain valid. Create fixtures
for two-stream dry-run batches: both pass/best wins, one passes, neither passes, pause blocks
auto-accept, and sealed-test hit rejects.

Constraints:
- This is loop-level multi-stream, not true multi-stream architecture.
- At most one stream may be accepted per batch.
- All streams, including losers/failures, must be journaled.
- Do not modify frozen objective/gate semantics.

Acceptance:
- Schema is documented in the plan or code comments.
- Fixture JSON can be consumed by the later run_loop batch mode without ambiguity.
- Manifest and gate self-test still pass.
```

### M1 implementation detail — Extend journal schema

Add batch-level journal records while keeping backwards compatibility with the current single-row format.

Recommended fields:

- `batch_id`;
- `baseline_run_name`;
- `streams`: list of stream records;
- per stream:
  - `stream_id`;
  - `hypothesis`;
  - `lever_tier`;
  - `config_path` or patch summary;
  - `commands`;
  - `metrics`;
  - `guardrails`;
  - `gate_decision`;
  - `rank_in_batch`;
  - `final_outcome`: `accepted`, `rejected_by_gate`, `paused`, `defeated_in_tournament`, `failed_run`;
- `accepted_stream_id` or null;
- sealed-test audit for the entire batch.

Acceptance:

- Existing journal reader logic does not break on old rows.
- One dry-run multi-stream batch writes a valid JSONL row with two streams and no sealed-test hits.

### M2 — Add run-loop batch mode

Extend `autoresearch/run_loop.py` with explicit batch flags.

Possible flags:

- `--batch-id ID`
- `--stream-configs path1.yaml,path2.yaml,...`
- `--max-streams N`
- `--execution-mode sequential|parallel`
- `--accept-policy best_passing|first_passing`
- `--journal-mode batch`

Start with `sequential` as the default because one 4070 Ti may not fit multiple simultaneous Qwen3-0.6B finetunes/evals reliably. Add true parallel execution only after sequential tournament behavior is correct.

Acceptance:

- Two-stream dry run completes and journals both streams.
- If both pass, the better stream wins.
- If one passes and one fails, the passing stream wins.
- If neither passes, no candidate is kept.
- Guardrail pause blocks the batch from auto-accepting.

#### Chainlink scaffolding prompt for #60 / M2

```text
Implement sequential tournament batch mode in autoresearch/run_loop.py.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections M1-M3.
- The journal schema/fixtures from #59.
- autoresearch/run_loop.py, autoresearch/gate.py, and autoresearch/gate_config.json.

Task:
Add batch mode flags: --batch-id, --stream-configs, --max-streams, --execution-mode
sequential|parallel, --accept-policy best_passing|first_passing, and --journal-mode batch.
For this issue, implement sequential execution only and make parallel mode error clearly as
not-yet-implemented unless #62 has landed. Each stream must run manifest check, finetune/eval/gate
with isolated artifact paths, then the batch chooses at most one accepted stream according to the
accept policy.

Constraints:
- Existing single-stream behavior must be unchanged when no batch flags are passed.
- Per-stream run_name/artifact paths must not collide.
- Guardrail pause means no auto-accept for the batch.
- Sealed-test audit applies to the whole batch.

Acceptance:
- Two-stream dry-run batch journals both streams.
- Fixtures cover both-pass, one-pass, none-pass, pause, and sealed-test reject.
- Gate self-test and manifest check still pass.
```

### M3 — Add proposal diversification rules

Update `program.md` so streams are meaningfully different, not N copies of the same tweak.

Recommended first stream families:

1. Tier 2 position/order augmentation stream;
2. Tier 1 conservative hyperparameter stream, e.g. lower LR / fewer epochs / dropout;
3. J-space-guided category-data stream;
4. J-space-guided rationale/answer-surface stream.

Rules:

- No two streams may change the same primary lever unless the batch is explicitly a micro-sweep.
- Every stream must state a falsifiable hypothesis.
- Every stream must cite `levers.md`, J-space artifact, or prior journal evidence.
- Failed streams must be interpreted in the next `program.md` update instead of silently repeated.

Acceptance:

- `program.md` contains stream-diversity rules.
- First real batch includes at least one non-hyperparameter stream.

#### Chainlink scaffolding prompt for #61 / M3

```text
Add proposer stream-diversity rules to autoresearch/program.md.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections 0.2, M1-M3, and section 4.
- autoresearch/program.md, autoresearch/levers.md, and the latest journal rows.

Task:
Update program.md so a multi-stream batch is not N copies of the same lever. Define the first
allowed stream families: position/order augmentation, conservative hyperparameter, J-space-guided
category-data, and J-space-guided rationale/answer-surface. Require each stream to state one
falsifiable hypothesis, one primary lever, expected primary movement, guardrail risk, and evidence
source.

Constraints:
- No two streams may change the same primary lever unless the batch is explicitly labelled as a
  micro-sweep.
- Failed streams must be summarized in the next program/journal update instead of silently repeated.
- Do not alter gate/objective thresholds.

Acceptance:
- program.md contains batch proposal rules and stream-family examples.
- levers.md cross-references the multi-stream rules if needed.
- First smoke batch can cite these rules directly.
```

### M4 — True parallel execution, optional

Only add true concurrent execution after sequential batch mode works.

Implementation choices:

- local process pool if enough GPUs/VRAM exist;
- queue-based execution on one GPU;
- external scheduler later if needed.

Constraints:

- Per-stream compute budget remains fixed.
- Artifacts are isolated by `run_name`.
- A failed stream cannot corrupt another stream's outputs.
- Frozen manifest check runs before every stream.

Acceptance:

- Parallel mode produces the same gate decision as sequential mode on deterministic dry-run fixtures.
- Artifact directories do not collide.

#### Chainlink scaffolding prompt for #62 / M4

```text
Add optional true parallel execution after sequential tournaments are proven.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections 3.0 and M4.
- autoresearch/run_loop.py batch mode from #60.
- The fixture outcomes from #59/#60.

Task:
Implement --execution-mode parallel only after sequential mode is passing. Use isolated artifact
directories per stream and preserve the exact same tournament semantics as sequential mode. If local
GPU/VRAM cannot support concurrent real finetunes, implement parallelism only for dry-run/fixture
mode and keep real execution queued.

Constraints:
- Same fixed compute budget per stream.
- A failed stream cannot corrupt another stream's artifacts.
- Manifest check runs before every stream.
- Parallel and sequential deterministic fixtures must choose the same winner.

Acceptance:
- Dry-run parallel fixture matches sequential decisions.
- Artifact paths do not collide.
- If real parallel is not feasible locally, document the reason and keep real mode disabled.
```

## 4. Chainlink issue breakdown

Existing epic:

- `#54` — `Phase 5: J-space-guided and multi-stream autoresearch extensions`

Tracker hygiene:

- `#54` is intentionally blocked by `#64` so the ready list exposes executable leaf work rather than
  the parent epic.

Existing subissues:

| Issue | Short name | Purpose | Blocks / waits for |
|---|---|---|---|
| `#55` | J1 | Build J-space item/category signal artifact | Starts immediately |
| `#56` | J2 | Freeze J-space config and document SAE deferral | Blocked by `#55` |
| `#57` | J3 | Update `program.md` / `levers.md` with J-space priors | Blocked by `#55`, `#56` |
| `#58` | J4 | Optional J-space data-selection hooks | Blocked by `#55`, `#56` |
| `#59` | M1 | Define multi-stream journal schema and dry-run fixtures | Starts immediately |
| `#60` | M2 | Add sequential tournament batch mode to `run_loop.py` | Blocked by `#59` |
| `#61` | M3 | Add proposer stream-diversity rules | Blocked by `#59` |
| `#62` | M4 | Optional true parallel execution | Blocked by `#60` |
| `#63` | Smoke | Dry-run J-space-informed two-stream batch | Blocked by `#57`, `#60`, `#61` |
| `#64` | Real run | First non-dry-run tournament | Blocked by `#63` |

### Epic comment prompt for #54

```text
Phase 5 scope is now source-grounded.

Use JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md as the source of truth. The local sources are:
- jacobian-lens/ companion repo for the Transformer Circuits J-space paper.
- 2605.12460v1.pdf, "Multi-Stream LLMs: Unblocking Language Models with Parallel Streams of
  Thoughts, Inputs and Outputs".

Implementation policy:
- J-space is an active-loop diagnostic/routing prior, not a primary objective.
- SAE is deferred unless the loop plateaus, J-space is ambiguous, or a compatible low-cost
  Qwen3-0.6B SAE exists.
- Multi-stream starts as loop-level candidate tournaments with at most one accepted winner.
- True model-level multi-stream architecture is explicitly out of scope until a collapsed
  single-stream surrogate proves useful.
- Sealed test data remains forbidden for diagnostics, routing, and stream selection.
```

### Smoke prompt for #63

```text
Run the first dry-run J-space-informed two-stream batch.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md sections J3, M1-M3, and section 5.
- #55/#56/#57 outputs for J-space artifacts/config/program rules.
- #59/#60/#61 outputs for batch schema, run_loop batch mode, and stream-diversity rules.

Task:
Create a two-stream dry-run batch. Recommended streams:
1. position/order augmentation stream;
2. J-space-guided rationale or category-data stream, depending on #55 bucket counts.
Run through autoresearch/run_loop.py in dry-run batch mode, gate both streams, journal both, and
accept at most one winner according to best_passing semantics.

Constraints:
- No sealed test reads.
- No objective/gate threshold changes.
- Both streams must use isolated artifact paths.
- If J-space artifacts are absent or low-quality, stop and report the blocker instead of inventing
  synthetic J-space evidence.

Acceptance:
- Batch journal row includes both streams and one accepted_stream_id or null.
- Sealed-test audit passes with empty hits.
- Manifest check and gate self-test pass after the smoke.
- Closing comment reports stream hypotheses, decisions, artifact paths, and selected winner.
```

### Real-run prompt for #64

```text
Run the first real non-dry-run Phase 5 tournament.

Read first:
- JSPACE_MULTI_STREAM_AUTORESEARCH_PLAN.md section 5.
- The dry-run result from #63.
- autoresearch/program.md and latest autoresearch/journal.jsonl rows.

Task:
Run one non-dry-run sequential tournament under the fixed compute budget. Use the same frozen gate,
same guardrails, and same sealed-test audit as the single-stream loop. Start with two streams unless
there is explicit capacity evidence for more.

Constraints:
- Do not touch eval_freeform/test.jsonl.
- Do not change frozen objective/gate/scorer files inside the experiment.
- If GPU/CUDA is unavailable, do not silently downgrade a real run to CPU unless the expected wall
  clock is acceptable and documented.
- Accept at most one candidate. If guardrail pause occurs, stop for human review.

Acceptance:
- Journal contains full batch record.
- Gate result is present for every stream.
- Winner/revert action is explicit.
- Closing comment reports primary deltas, guardrail status, compute time, accepted stream if any,
  and sealed-test audit status.
```

## 5. Verification checklist

- Old infrastructure plans are archived; this file is the active next-phase plan.
- Frozen manifest still passes before any J-space/multi-stream work begins.
- J-space artifact generation does not read sealed test files.
- SAE deferral rationale is recorded in this plan and later in `program.md`/`levers.md` once J-space is implemented.
- Single-stream run_loop behavior remains unchanged when batch flags are absent.
- Multi-stream dry run accepts at most one stream and journals all streams.
- First real multi-stream run uses the same frozen gate and same sealed-test guard as the single-stream loop.

## 6. Non-goals

- Do not train an SAE in the active loop.
- Do not use J-space as a new primary objective.
- Do not change objective weights to make J-space or ref-LL look good.
- Do not read sealed test data for J-space annotation or stream selection.
- Do not add uncontrolled free-form data generation inside `run_loop.py`; data authoring remains explicit and auditable.
