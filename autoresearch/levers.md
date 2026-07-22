# Lever inventory and classification (Phase 2.7, chainlink #21)

Single source of truth feeding **both** `autoresearch/finetune.py`'s parameter surface
(SEARCH DIMENSION rows only) and `program.md`'s priors (every row's status +
direction). Built from `PHASE_2.6_2.7_4_PLAN.md` Part B, the B3 settling diagnostics
(chainlink #20, `../benchmark-eval-workspace/results/B3_diagnostic_summary.md`), and a
direct read of the current `train.py` / `prepare_dataset_v3.py` parameter surface plus
the actual hyperparameters recorded in v3-clean's `trainer_state.json` /
`adapter_config.json`.

> Multi-stream note (Phase 5, #61): when proposing a tournament batch, each stream moves
> exactly one lever row from this file, and no two streams may move the same row unless
> the batch is a labelled micro-sweep. Full stream-diversity rules live in `program.md`.

## Status legend

- **SETTLED CONSTANT** — diagnosis resolved it; not a search knob, but `finetune.py` must
  keep doing this.
- **FORBIDDEN** — a guardrail owns it; the loop must never move this direction.
- **SEARCH DIMENSION** — a live knob with a directional prior. Every row here maps 1:1
  to a `finetune.py` / `prepare_dataset_v3.py` parameter.
- **PENDING-UNLOCK** — can't be classified until a named prerequisite lands (usually
  Phase 2.6's free-form eval, or a data pipeline that doesn't exist yet).

---

## Tier 1 — Hyperparameters

| Lever | Status | Direction / prior | Evidence | Knob |
|---|---|---|---|---|
| Learning rate (peak) | **SETTLED CONSTANT** | Pin `lr=5e-5`; this is no longer a search dimension. v3-clean/v4clean's `2e-4` default is **FALSIFIED / DEAD** relative to 5e-5 at this dataset size. | Chainlink #86: n=4 identical runs confirmed 5e-5, primary delta +1.478 +/- 0.071 SD (an order of magnitude tighter than any other lever's run noise, on an effect ~20x that noise floor). | `--lr` |
| LR schedule | **SETTLED CONSTANT** | Pin `lr_scheduler=cosine`. Re-tested against v5clean (chainlink #91): linear delta -0.070 (-0.35 SE, not significant, guards clean -- forgetting actually improved -10%); constant delta -1.743 (-8.72 SE, a clear regression, fails forgetting 30% drop, bias reintroduced 0.299, rubric drop 5.95pts). Both reject; cosine remains the pin. Different mechanism than epochs/rank (not a capacity axis) but same outcome shape: undiminished optimization pressure (no LR decay) destabilizes the model on this small dataset. | Chainlink #91, commit ed318e9 | `--lr-scheduler {cosine,linear,constant}` |
| Warmup ratio | **SETTLED CONSTANT** | Pin `warmup_ratio=0.05`. Re-tested against v5clean (chainlink #91): no warmup (`0.00`) produced delta `-0.030` (`-0.15 SE`) with forgetting, bias, and rubric clean, but exceeded the 28-minute training guard at 28.17 minutes; doubled warmup (`0.10`) produced delta `-0.103` (`-0.52 SE`) with compute 26.93 minutes, rubric drop 4.41 points, and all other guards clean. Both reject; neither improves the settled 5% setting. | Chainlink #91, commit a6abf7b | `--warmup-ratio` |
| Epochs | **SETTLED CONSTANT** | Pin `epochs=4`. Re-tested against v5clean's lr=5e-5 base (chainlink #91): epochs=3 delta +0.150 (0.75 SE, not significant, otherwise clean); epochs=5 delta -1.116 (-5.58 SE, a clear regression that also fails compute/forgetting/bias guards). Both directions reject; 4 remains the pin under both the old (lr=2e-4) and new (lr=5e-5) regimes. | Chainlink #91, commit 0d45fcf | `--epochs` |
| Early-stop patience | SEARCH DIMENSION | v3-clean had no eval-based early stop. Now wired through eval-loss with epoch-aligned evaluation when enabled. | prior `train.py` used `eval_strategy="no"` | `--early-stop-patience` |
| Batch size | SEARCH DIMENSION | v3-clean used per-device batch=4. Untested. Low priority — unlikely to be the dominant lever given diagnostic 1's negative-headroom finding (see prose below). | v3-clean training args | `--batch-size` |
| Gradient accumulation | SEARCH DIMENSION | v3-clean used grad-accum=4 (effective 16 with batch=4). Untested. | v3-clean training args | `--grad-accum` |
| LoRA rank r | **SETTLED CONSTANT** | Pin `r=32` (alpha=64, ratio 2:1). Re-tested against v5clean (chainlink #91), alpha co-scaled to isolate pure capacity: r=16/alpha=32 delta +0.184 (0.92 SE, not significant, clean); r=64/alpha=128 delta -1.627 (-8.14 SE, a clear regression, fails forgetting 40% drop and bias reintroduced 0.299). Both reject; 32 remains the pin. Same shape as the epochs result: decreasing capacity is neutral, increasing it is a decisive regression. | Chainlink #91, commit 9acb52f | `--lora-r` |
| LoRA alpha | **SETTLED CONSTANT** | Pin `lora_alpha=64` at `r=32` (alpha/r=2). Geometric dose-response re-test against v5clean (chainlink #91, fixed r=32): alpha=16 delta `+0.161` (`+0.80 SE`) and alpha=32 delta `+0.193` (`+0.96 SE`) were clean but non-significant; alpha=128 delta `-1.602` (`-8.01 SE`) failed forgetting and reintroduced bias; alpha=256 delta `-1.577` (`-7.89 SE`) reintroduced bias. All reject; the lower settings do not clear the bar and higher settings reproduce the small-dataset instability. | Chainlink #91, commit 6200db6 | `--lora-alpha` |
| LoRA dropout | **SETTLED CONSTANT** | Pin `lora_dropout=0.025`; this is no longer a search dimension. `lora_dropout=0.01` is **FALSIFIED / DEAD**: the higher-precision rubric re-measurement reversed its earlier provisional accept. | Chainlink #84: repeated d025 runs confirmed `0.025` as the robust setting; chainlink #82: the higher-precision rubric result falsified `0.01`. | `--lora-dropout` |
| LoRA target modules | SEARCH DIMENSION | v3-clean uses the full 7-module list (attn q/k/v/o + mlp gate/up/down). Untested narrower sets (e.g. attn-only). | `adapter_config.json` | `--target-modules` |
| LoRA vs full finetune | SETTLED CONSTANT | Use LoRA. Phase 4's stated compute budget (~5–20 min/run at 0.6B+LoRA → 30–100 experiments/night) assumes LoRA; full finetuning changes the compute-budget tier entirely and isn't compatible with the nightly-loop design. | n/a | not exposed |
| Weight decay | SEARCH DIMENSION | Fixed 0.01 for v3-clean. Untested. Low priority. | v3-clean training args / prior hardcoded `weight_decay=0.01` | `--weight-decay` |
| Seq-len | **SETTLED CONSTANT** | Keep `max_seq_len=2048`. The 1024-cap test regressed primary by `-0.789` (`-3.945 SE`), increased forgetting by a 10% relative drop (41.67% → 37.50%), and reintroduced answer-letter bias (`0.1128` vs 0.1097 allowed), despite improving rubric from 60.8% to 61.71%. | `results/autoresearch_gate_p7-seqlen1024.json`, `results/autoresearch_guardrails_p7-seqlen1024.json`, `results/periodic_guards_p7-seqlen1024.json`; commit `bdc6b32` | `--max-seq-len=2048` |
| Packing | **SETTLED CONSTANT** | Keep `packing=false`. The correctness/performance check regressed primary by `-0.089` (`-0.445 SE`), missed the compute budget at 28.09 vs 28.00 minutes, lowered rubric from 60.8% to 57.79%, and increased forgetting by a 10% relative drop (41.67% → 37.50% accuracy). | `results/autoresearch_gate_p7-packing.json`, `results/autoresearch_guardrails_p7-packing.json`, `results/periodic_guards_p7-packing.json`; commit `ff506a1` | `--packing=false` |

---

## Tier 2 — Data / format

| Lever | Status | Direction / prior | Evidence | Knob |
|---|---|---|---|---|
| Gold-letter balance | **SETTLED CONSTANT** | Keep exactly balanced (25/25/25/25). Do not reintroduce imbalance. | Pre-v3 diagnostic: biased_letter error bucket 55.2%→0%, marginal entropy 1.009→1.683/2.0 bits after balancing (`summary_v3clean.md`). v3-clean's actual letter distribution is exact 25/25/25/25 (commit `85019a9`). | `_SHIFT_WEIGHTS` in `prepare_dataset_v3.py` already enforces this; keep as-is |
| Re-introducing letter bias | **FORBIDDEN** | Any lever move that skews the letter marginal away from ~25% each must be rejected by the bias guardrail (`diagnostics/selection_bias.py`, C4). | Same as above | guardrail, not a knob |
| Answer-position shuffle / per-question consistency augmentation | **SEARCH DIMENSION — highest-confidence target** | Increase multi-permutation coverage, but **targeted, not blanket**: B3 diagnostic 3 sized the augmentation-worthwhile subset to 321 items (20.0% of the benchmark — the `near_miss`+`toss_up` buckets), not the full 94.7% flip-any-fraction the old headline number implied. The dominant failure mode (74.7% `mostly_wrong`) won't be fixed by shuffling alone. Current caveat: the diagnostic seed set is keyed by benchmark IDs, while current QMD records use `_source_question_id` training IDs, so zero-overlap inputs warn rather than silently no-op. | `../benchmark-eval-workspace/results/position_spread_breakdown.json` — exact `augmentation_seed_set` (321 question IDs) ready to consume | `--target-permutation-ids` |
| General replay ratio (farming-conversational : MCQ mix) | SEARCH DIMENSION — **knob now exists** (chainlink #26) | **Non-monotone — do not assume more MCQ (or more conversational) is better.** v3-clean's mix (4409 conversation pairs : 311×~5 MCQ-with-shifts pairs, before dedup) was just "whatever's in the two source files," not a designed choice. CROP's own replay-ratio result was non-monotone (Table 2: +CROP 0.676→+CQIA+CROP 0.709 overall but Difficult −0.075) — expect the same risk here. | `PHASE_2.6_2.7_4_PLAN.md` CROP citation; `85019a9` commit for the prior emergent ratio; verified against real data at ratio=0.5 and 0.97 (both branches, letter balance held) | `--conversation-ratio F` in `prepare_dataset_v3.py` (subsample-only — never upsamples/duplicates; MCQ-side subsampling stratified by answer letter to preserve the balance invariant). `None` (default) = old uncontrolled behavior, unchanged. |
| Chat template / prompt variants | PENDING-UNLOCK | Can't be scored without the free-form eval axis (2.6) as a second signal — MCQ accuracy alone can't distinguish "better prompt" from noise at this granularity. | — | — |
| SFT target format (`"B"` vs `"Answer: B"` vs sentence) | SEARCH DIMENSION | v3-clean hardcodes bare-letter output (`<think>...\n</think>\n{letter}`). Untested alternatives. Diagnostic 1's finding (peak knowledge probe 37.1% vs 75.2% generation, confident-wrong-dominated errors) suggests the model already commits confidently to a format — worth testing whether a more verbose target format changes what surfaces, but this is speculative, not diagnostic-backed. | `_cyclic_shift_pair` output format | `--sft-format {bare-letter,answer-prefix,sentence}` |
| MCQ-vs-cloze items | PENDING-UNLOCK | No cloze format exists in any data-generation script yet; would need new item authoring before it's even testable. | — | — |
| Distractor quality | **OUT OF SCOPE for `finetune.py`** | This is a data-*authoring*-time property owned by `scripts/generate_mcq_from_qmd.py` + the judge pipeline, not a training-time knob. Not part of the autoresearch loop's search surface. | — | — |
| Curriculum order | PENDING-UNLOCK | No difficulty signal exists on any training item (`difficulty` field in benchmark data is uniformly `"unknown"`). Can't order by difficulty until one is computed. | checked `qwen3-0_6b-v3-clean_results.json` per-question `difficulty` field | — |
| Synth-item (QMD MCQ) count | SEARCH DIMENSION | v3-clean used all 311 available QMD-generated questions. The live harness can vary downward by deterministic subsampling of the existing pool; increasing above the current ceiling still requires an offline `generate_mcq_from_qmd.py` re-run. | `85019a9` — 311 raw → ~1555 with shift weighting | `--synth-item-count` |

---

## Tier 3 — Reasoning (PENDING-UNLOCK except the J-space routing prior, promoted Phase 5 #57)

| Lever | Status | Direction / prior (once unlocked) | Evidence |
|---|---|---|---|
| Rationale / CoT targets | PENDING-UNLOCK | **Caveat from B3, read before unlocking:** diagnostic 1's negative real-gen headroom (peak knowledge 37.1% vs 75.2% generation, `headroom=-0.381` on the full 1603Q benchmark) means this is *not* a guaranteed free win — whatever "unsurfaced" signal diagnostic 2 finds under a suppressed-reasoning prompt does not clearly persist once the model actually reasons. If/when unlocked, **aim first at Natural Fertilizers (65.3% acc), Sustainable Water Management (67.9%), Soil Science (68.9%)** — diagnostic 2's lowest-accuracy, majority-"unsurfaced" categories. Sustainable Water Mgmt is a distinct sub-case (signal arrives at layer 21/27 vs 1–3/27 elsewhere — late/weak, may need different treatment). | B3 diagnostics 1 & 2, full results in `../benchmark-eval-workspace/results/B3_diagnostic_summary.md` |
| Reflection-data proportion | PENDING-UNLOCK | Unmeasurable without a free-form quality signal. | plan doc |
| J-space-informed data selection | **ACTIVE ROUTING PRIOR** (promoted Phase 5, chainlink #57) | Use bucket evidence from `results/jspace_item_signals.jsonl` / `results/jspace_category_summary.json` to route which lever a proposal targets: `absent` → category/source data expansion; `late_weak` → rationale targets; `present_unsurfaced` → rationale or answer-surface experiments; `format_sensitive` → format/chat-template experiments; `position_unstable` → targeted permutation; `robust` → do not target except replay. Evidence for *choosing* a lever, never a score to optimize; regenerate artifacts whenever the champion changes (the lens is model-specific). Narrow selection knobs (`--jspace-*`) land with #58. | v3-clean dev baseline (113 scorable MCQ rows): present_unsurfaced 45, position_unstable 34, format_sensitive 12, late_weak 12, absent 10, robust 0; diagnostic surface frozen per #56 (`autoresearch/jspace_config.json`) |
| Free-form score as weighted objective term | PENDING-UNLOCK | This lever *is* Phase 2.6 itself — it doesn't unlock, it gets built. | plan doc |

---

## CROP-derived levers (verified from CROP PDF)

| Lever | Status | Direction / prior | Evidence |
|---|---|---|---|
| Auxiliary-task mixing (summary/NER/extraction/open-book) | PENDING-UNLOCK | No aux-task data pipeline exists in this repo yet — needs authoring before it's a knob, and scoring it needs the free-form axis anyway (MCQ accuracy can't detect whether aux-task mixing helped). | CROP PDF |
| Multi-turn / step-back SFT proportion | PENDING-UNLOCK | Same — no multi-turn step-back data exists; `_load_conversations` flattens ShareGPT multi-turn to single instruction/output pairs today, discarding turn structure. | CROP PDF |
| Open-book/RAG-grounded vs closed-book targets | PENDING-UNLOCK | No RAG-grounded data pipeline exists. | CROP PDF |
| General-instruction replay ratio | PENDING-UNLOCK (folds into the Tier-2 replay-ratio lever once a general-instruction source exists) | **Non-monotone — CROP Table 2 (Qwen1.5-7B): +CROP 0.676→+CQIA+CROP 0.709 overall but Difficult −0.075.** Don't assume more replay = better once this is buildable. | CROP PDF, verified |

---

## Prose priors per tier (for `program.md`)

**Tier 1 (hyperparameters):** all SEARCH DIMENSION except LoRA-vs-full (SETTLED: use
LoRA). None of these have diagnostic evidence pointing a specific direction yet — they're
all "untested at v3-clean's exact values, worth a sweep," not "known to move accuracy."
Treat as the loop's baseline/cheap-to-vary knobs, not where the interesting signal is
expected. Packing needs an implementation change before it's searchable at all.

**Tier 2 (data/format):** this is where B3's diagnostics concentrate the strongest,
most specific priors. Two settled facts anchor it — keep gold-letter balance exact
(SETTLED CONSTANT / FORBIDDEN to undo), and the position-invariance augmentation lever
is now sized and seeded concretely (321 items, not "shuffle everything"). The
general-replay-ratio lever now has a real knob (`--conversation-ratio`, chainlink #26) —
CROP's non-monotone result still means don't default to "more is better" when it's
first searched.

**Tier 3 (reasoning) and CROP-derived:** PENDING-UNLOCK behind Phase 2.6's
free-form eval axis, except J-space-informed data selection, which is now an ACTIVE
ROUTING PRIOR (Phase 5, #57): it steers which lever a proposal targets using the frozen
J-space diagnostic surface, without itself being a measurable objective. One caveat worth
flagging early: diagnostic 1's negative
real-gen headroom is a soft warning against over-investing in CoT/rationale levers as
an assumed win. When 2.6 lands and these unlock, diagnostic 2's category ranking
(Natural Fertilizers → Sustainable Water Mgmt → Soil Science) gives a concrete starting
point rather than searching blind across all 9 categories at once.

---

## Open engineering gaps this table surfaces (for C1/C2, not this issue)

- ~~`prepare_dataset_v3.py` needs a conversation:MCQ ratio knob~~ — done, chainlink #26
  (`--conversation-ratio`).
- ~~`prepare_dataset_v3.py` needs a targeted-permutation flag consuming the 321-item
  augmentation seed set~~ — done as `--target-permutation-ids`; current QMD source IDs do
  not overlap the benchmark seed IDs, so the implementation warns on zero-overlap input.
- ~~`prepare_dataset_v3.py` needs an SFT target-format knob~~ — done as `--sft-format`.
- ~~`prepare_dataset_v3.py` needs a synthetic-item-count knob~~ — done as
  `--synth-item-count`, implemented honestly as a deterministic cap/subsample of the
  existing QMD pool; increasing above the current pool still requires offline generation.
- ~~`train.py` needs flags for LR schedule type, warmup ratio, LoRA dropout, LoRA target
  module set, weight decay, and sequence packing~~ — done.
- ~~`train.py` has no eval loop during training (`eval_strategy="no"`)~~ — early-stopping
  now enables epoch-aligned eval and `EarlyStoppingCallback`.
