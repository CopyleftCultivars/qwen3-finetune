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
| Early-stop patience | **SETTLED CONSTANT** | Keep early stopping disabled. Patience=1 regressed primary by `-0.159` (`-0.793 SE`), exceeded compute at 28.32 vs 28.00 minutes, and increased forgetting by a 10% relative drop (41.67% → 37.50%); rubric improved to 64.15% and bias stayed clean. | `results/autoresearch_gate_p7-earlystop1.json`, `results/autoresearch_guardrails_p7-earlystop1.json`, `results/periodic_guards_p7-earlystop1.json`; commit `609d2b5` | no `--early-stop-patience` |
| Batch size | **SETTLED CONSTANT** | Keep `batch_size=4`. The batch_size=8 candidate failed during finetuning after 8.61 minutes, before scoring or gate evaluation; no passing evidence justifies changing the pinned value. | `autoresearch/journal.jsonl` batch `p7-batch8-sweep-1-gpu`; failed-run commit `6024117` | `--batch-size=4` |
| Gradient accumulation | **SETTLED CONSTANT** | Keep `grad_accum=4` (effective batch 16 with batch=4). The grad_accum=2 test regressed primary by `-0.985` (`-4.925 SE`), reintroduced bias (`0.1659` vs 0.1097 allowed), and lowered rubric to 59.83%, despite improving forgetting. | `results/autoresearch_gate_p7-gradacc2.json`, `results/autoresearch_guardrails_p7-gradacc2.json`, `results/periodic_guards_p7-gradacc2.json`; commit `8ade632` | `--grad-accum=4` |
| LoRA rank r | **SETTLED CONSTANT** | Pin `r=32` (alpha=64, ratio 2:1). Re-tested against v5clean (chainlink #91), alpha co-scaled to isolate pure capacity: r=16/alpha=32 delta +0.184 (0.92 SE, not significant, clean); r=64/alpha=128 delta -1.627 (-8.14 SE, a clear regression, fails forgetting 40% drop and bias reintroduced 0.299). Both reject; 32 remains the pin. Same shape as the epochs result: decreasing capacity is neutral, increasing it is a decisive regression. | Chainlink #91, commit 9acb52f | `--lora-r` |
| LoRA alpha | **SETTLED CONSTANT** | Pin `lora_alpha=64` at `r=32` (alpha/r=2). Geometric dose-response re-test against v5clean (chainlink #91, fixed r=32): alpha=16 delta `+0.161` (`+0.80 SE`) and alpha=32 delta `+0.193` (`+0.96 SE`) were clean but non-significant; alpha=128 delta `-1.602` (`-8.01 SE`) failed forgetting and reintroduced bias; alpha=256 delta `-1.577` (`-7.89 SE`) reintroduced bias. All reject; the lower settings do not clear the bar and higher settings reproduce the small-dataset instability. | Chainlink #91, commit 6200db6 | `--lora-alpha` |
| LoRA dropout | **SETTLED CONSTANT** | Pin `lora_dropout=0.025`; this is no longer a search dimension. `lora_dropout=0.01` is **FALSIFIED / DEAD**: the higher-precision rubric re-measurement reversed its earlier provisional accept. | Chainlink #84: repeated d025 runs confirmed `0.025` as the robust setting; chainlink #82: the higher-precision rubric result falsified `0.01`. | `--lora-dropout` |
| LoRA target modules | **SETTLED CONSTANT** | Keep the full 7-module list (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`). The attention-only test regressed primary by `-0.620` (`-3.101 SE`), reintroduced bias (`0.1394` vs 0.1097 allowed), and lowered rubric from 60.8% to 57.59%, despite improving forgetting. | `results/autoresearch_gate_p7-target-attn.json`, `results/autoresearch_guardrails_p7-target-attn.json`, `results/periodic_guards_p7-target-attn.json`; commit `b2c395f` | full `--target-modules` list |
| LoRA vs full finetune | SETTLED CONSTANT | Use LoRA. Phase 4's stated compute budget (~5–20 min/run at 0.6B+LoRA → 30–100 experiments/night) assumes LoRA; full finetuning changes the compute-budget tier entirely and isn't compatible with the nightly-loop design. | n/a | not exposed |
| Weight decay | **SETTLED CONSTANT** | Keep `weight_decay=0.01`. The zero-decay test regressed primary by `-0.156` (`-0.780 SE`) and lowered rubric from 60.8% to 58.64%; forgetting, bias, anchors, and compute remained clean. | `results/autoresearch_gate_p7-wd0.json`, `results/autoresearch_guardrails_p7-wd0.json`, `results/periodic_guards_p7-wd0.json`; commit `ceb779f` | `--weight-decay=0.01` |
| Seq-len | **SETTLED CONSTANT** | Keep `max_seq_len=2048`. The 1024-cap test regressed primary by `-0.789` (`-3.945 SE`), increased forgetting by a 10% relative drop (41.67% → 37.50%), and reintroduced answer-letter bias (`0.1128` vs 0.1097 allowed), despite improving rubric from 60.8% to 61.71%. | `results/autoresearch_gate_p7-seqlen1024.json`, `results/autoresearch_guardrails_p7-seqlen1024.json`, `results/periodic_guards_p7-seqlen1024.json`; commit `bdc6b32` | `--max-seq-len=2048` |
| Packing | **SETTLED CONSTANT** | Keep `packing=false`. The correctness/performance check regressed primary by `-0.089` (`-0.445 SE`), missed the compute budget at 28.09 vs 28.00 minutes, lowered rubric from 60.8% to 57.79%, and increased forgetting by a 10% relative drop (41.67% → 37.50% accuracy). | `results/autoresearch_gate_p7-packing.json`, `results/autoresearch_guardrails_p7-packing.json`, `results/periodic_guards_p7-packing.json`; commit `ff506a1` | `--packing=false` |

---

## Tier 2 — Data / format

| Lever | Status | Direction / prior | Evidence | Knob |
|---|---|---|---|---|
| Gold-letter balance | **SETTLED CONSTANT** | Keep exactly balanced (25/25/25/25). Do not reintroduce imbalance. | Pre-v3 diagnostic: biased_letter error bucket 55.2%→0%, marginal entropy 1.009→1.683/2.0 bits after balancing (`summary_v3clean.md`). v3-clean's actual letter distribution is exact 25/25/25/25 (commit `85019a9`). | `_SHIFT_WEIGHTS` in `prepare_dataset_v3.py` already enforces this; keep as-is |
| Re-introducing letter bias | **FORBIDDEN** | Any lever move that skews the letter marginal away from ~25% each must be rejected by the bias guardrail (`diagnostics/selection_bias.py`, C4). | Same as above | guardrail, not a knob |
| Answer-position shuffle / per-question consistency augmentation | **SEARCH DIMENSION — highest-confidence target** | Increase multi-permutation coverage, but **targeted, not blanket**: B3 diagnostic 3 sized the augmentation-worthwhile subset to 321 items (20.0% of the benchmark — the `near_miss`+`toss_up` buckets), not the full 94.7% flip-any-fraction the old headline number implied. The dominant failure mode (74.7% `mostly_wrong`) won't be fixed by shuffling alone. Current caveat: the diagnostic seed set is keyed by benchmark IDs, while current QMD records use `_source_question_id` training IDs, so zero-overlap inputs warn rather than silently no-op. First training-overlap dose (17 IDs, multiplier 2) was clean but rejected: primary delta `-0.097` (`-0.484 SE`), rubric improved to 63.89%. | B3 `position_spread_breakdown.json`; tested report `results/jspace_position_unstable_target_ids.json`; results `results/autoresearch_gate_p7-perm-targeted.json`, commit `0d0f95b` | `--target-permutation-ids` |
| General replay ratio (farming-conversational : MCQ mix) | SEARCH DIMENSION — **knob now exists** (chainlink #26) | **Non-monotone — do not assume more MCQ (or more conversational) is better.** v3-clean's mix (4409 conversation pairs : 311×~5 MCQ-with-shifts pairs, before dedup) was just "whatever's in the two source files," not a designed choice. CROP's own replay-ratio result was non-monotone (Table 2: +CROP 0.676→+CQIA+CROP 0.709 overall but Difficult −0.075) — expect the same risk here. Ratio 0.5 was a near miss: primary `+0.372` (`+1.858 SE`, threshold `+0.4`), rubric 57.39%, and forgetting improved to 66.67% accuracy. Ratio 0.4 was also rejected: primary `+0.327` (`+1.637 SE`), rubric 55.67% (5.13-point drop, just beyond the 5-point limit), while forgetting improved to 62.5%; bias, anchors, compute, and sealed-test checks were clean. The dimension remains open for dose refinement. | `PHASE_2.6_2.7_4_PLAN.md`; tested `results/autoresearch_gate_p7-ratio05.json` and `results/autoresearch_gate_p7-ratio04.json`; ratio-0.5 result commit `5c4fdb4`, ratio-0.4 result commit pending | `--conversation-ratio F` in `prepare_dataset_v3.py` (subsample-only — never upsamples/duplicates; MCQ-side subsampling stratified by answer letter to preserve the balance invariant). `None` (default) = old uncontrolled behavior, unchanged. |
| Chat template / prompt variants | **SEARCH REJECTED — tokenizer-v1** | The periodic free-form rubric axis unlocked this lever and exposed a severe regression: official tokenizer rendering versus the historical hard-coded rendering produced primary delta `-1.213` (`-6.065 SE`), bias `0.3164`, forgetting drop 10%, rubric `60.80% → 49.81%`, and compute 28.18 vs 28.00 minutes. Keep the hardcoded-v1 training interface; deployment system-prompt A/B remains a separate rejected diagnostic. | `autoresearch/configs/phase10-tier2/stream-p10-chat-template-tokenizer.yaml`; `results/autoresearch_gate_p10-chat-template-tokenizer.json`; `results/autoresearch_guardrails_p10-chat-template-tokenizer.json` | `train.py --chat-template-variant {hardcoded-v1,tokenizer-v1}` |
| SFT target format (`"B"` vs `"Answer: B"` vs sentence) | **SETTLED CONSTANT** | Keep bare-letter output (`<think>...\n</think>\n{letter}`). Answer-prefix is **FALSIFIED / DEAD** in the current base: primary delta `-2.006` (`-10.028 SE`), forgetting drop 30%, bias `0.3872`, and rubric 60.69% vs 60.8%. Sentence format is also **FALSIFIED / DEAD**: primary delta `-1.783` (`-8.916 SE`), forgetting drop 50%, bias `0.4314`, compute 28.07 minutes, and rubric 60.25%. The surface alternatives do not beat the pinned bare-letter target. | `_cyclic_shift_pair` output format; `results/autoresearch_gate_p7-surface-prefix.json`, `results/autoresearch_gate_p7-surface-sentence.json`; answer-prefix commit `1e2c682`, sentence result commit pending | `--sft-format {bare-letter,answer-prefix,sentence}` |
| MCQ-vs-cloze items | **SEARCH REJECTED — 400-item isolated dose** | The isolated p8-cloze400 candidate changed only the audited cloze stream and was rejected: primary delta `-0.316` (`-1.580 SE`), compute 28.35 vs 28.00 minutes, bias `0.1121` vs `0.1097` allowed, and rubric `60.80% → 59.37%`; forgetting improved 41.67% → 45.83%. The earlier combined p9 rejection is now properly separated from this causal dose result. Richer cloze authoring or a smaller dose would be a new hypothesis, not an automatic continuation. | `autoresearch/cloze_pipeline.md`; `results/autoresearch_gate_p8-cloze400.json`; `results/autoresearch_guardrails_p8-cloze400.json` | `--cloze-source PATH`, `--cloze-count N` |
| Embedded system/persona text | **SEARCH REJECTED — isolated filter-only candidate** | Composite QMD records previously retained an old persona prefix as user content: 16/311 raw QMD records, 65 expanded variants survived into the p8 final dataset. The isolated filter-only run improved primary `+0.059` (`+0.293 SE`) but missed the `+0.4` threshold and reintroduced bias (`0.1173` vs `0.1097` allowed); rubric was essentially unchanged (`60.80% → 60.68%`) and forgetting improved. No champion promotion. | `prepare_dataset_v3.py`; `tests/test_cloze_pipeline.py`; `results/autoresearch_gate_p10-filter-only.json`; `results/autoresearch_guardrails_p10-filter-only.json` | `--filter-system-prompts` |
| Reviewed adversarial refusals | **SEARCH REJECTED — isolated refusal-only candidate** | Incidental prompt-injection-looking chat rows are not reliable robustness training. The isolated ten-row candidate improved primary `+0.050` (`+0.248 SE`) but missed the `+0.4` threshold; rubric fell 1.67 points, forgetting improved, and bias/anchors/compute stayed clean. No champion promotion. | `data/sources/adversarial_refusals.jsonl`; `autoresearch/adversarial_refusals.md`; `results/autoresearch_gate_p10-refusal-only.json`; `results/autoresearch_guardrails_p10-refusal-only.json` | `--adversarial-refusals PATH`, `--adversarial-refusal-count N` |
| Deployment system prompt | **A/B REJECTED for current domain MCQ interface** | The clean deployment prompt reduced baseline validation argmax accuracy `73.28% → 55.17%` and candidate `49.14% → 39.66%`; option-set gold log-likelihood also fell for both. Keep it out of the default path for this interface. This does not rule out a separately designed system-conditioned free-form experiment. | `autoresearch/system_prompts/natural_farming.txt`; `autoresearch/compare_system_prompt.py`; `results/system_prompt_compare_p9-clean-cloze400-refusal.json` | post-training A/B diagnostic |
| Distractor quality | **OUT OF SCOPE for `finetune.py`** | This is a data-*authoring*-time property owned by `scripts/generate_mcq_from_qmd.py` + the judge pipeline, not a training-time knob. Not part of the autoresearch loop's search surface. | — | — |
| Curriculum order | **SEARCH REJECTED — three-repeat confirmation** | No trusted difficulty field exists, so `autoresearch/curriculum.py` adds an explicit `length-structure-v1` heuristic using only training instruction/target text. The original easy-to-hard run was a +0.3994 near miss; fixed-data repeats at training seeds 101/102/103 produced +0.199, +0.338, and +0.340, all below the +0.4 threshold. Hard guards stayed clean and forgetting improved, but the primary effect did not clear the gate consistently. This closes the current heuristic/dose; any larger or richer curriculum pool is a new hypothesis. | `autoresearch/curriculum.py`; `tests/test_curriculum.py`; `results/autoresearch_gate_p10-curriculum-confirm-r1.json`; `results/autoresearch_gate_p10-curriculum-confirm-r2.json`; `results/autoresearch_gate_p10-curriculum-confirm-r3.json` | `--curriculum-order {none,easy-to-hard,hard-to-easy}` + sequential train sampler |
| Synth-item (QMD MCQ) count | SEARCH DIMENSION — **half-pool dose explored** | v3-clean used all 311 available QMD-generated questions. A deterministic half-pool (`155` raw items, then the normal cyclic-shift expansion) was clean but rejected: primary delta `+0.184` (`+0.922 SE`), rubric drop 0.28 points, forgetting improved to 62.5% accuracy, and bias/anchors/compute/sealed checks passed. The live harness can vary downward by deterministic subsampling; increasing above the current ceiling still requires an offline `generate_mcq_from_qmd.py` re-run and richer item-authoring pipeline. | `85019a9` — 311 raw → ~1555 with shift weighting; `results/autoresearch_gate_p7-synth155.json`, commit `dbf4876` (config) and result commit pending | `--synth-item-count` |

---

## Tier 3 — Reasoning (rationale-target path unlocked; replay strategy rejected)

| Lever | Status | Direction / prior (once unlocked) | Evidence |
|---|---|---|---|
| Rationale / CoT targets | **SEARCH REJECTED — two category/dose runs; replay strategy closed** | P11 combined natural_fertilizers + soil_science_and_microbiology rationale replay produced primary delta `-0.0732` and bias `0.1645`; p12 natural_fertilizers-only replay produced delta `-0.0597` and bias `0.1469`. Both missed the `+0.4` gate; p12 kept rubric within guard (`60.80% → 60.53%`), forgetting unchanged, and anchors clean, but still failed bias. A direct postmortem found all 311 clean rationale outputs exactly match `qmd_mcq_all_phase3.jsonl` by `_source_question_id`: the source is contamination-audited and opt-in, but it is not novel reasoning supervision. Close this replay strategy; a future rationale lever needs genuinely new authoring or target construction. | `results/autoresearch_gate_p11-rationale-overlay.json`; `results/autoresearch_gate_p12-rationale-nf-only.json`; `results/autoresearch_guardrails_p12-rationale-nf-only.json`; `results/periodic_guards_p12-rationale-nf-only.json`; `data/sources/qmd_mcq_rationale_clean.jsonl`; `autoresearch/configs/phase12-tier3/p12-rationale-nf-only.yaml` |
| Reflection-data proportion | PENDING-UNLOCK | Unmeasurable without a free-form quality signal. | plan doc |
| J-space-informed data selection | **ACTIVE ROUTING PRIOR** (promoted Phase 5, chainlink #57) | Use bucket evidence from `results/jspace_item_signals.jsonl` / `results/jspace_category_summary.json` to route which lever a proposal targets: `absent` → category/source data expansion; `late_weak` → rationale targets; `present_unsurfaced` → rationale or answer-surface experiments; `format_sensitive` → format/chat-template experiments; `position_unstable` → targeted permutation; `robust` → do not target except replay. Evidence for *choosing* a lever, never a score to optimize; regenerate artifacts whenever the champion changes (the lens is model-specific). Narrow selection knobs (`--jspace-*`) land with #58. | v5clean J-space refit: late_weak 63, position_unstable 20, format_sensitive 13, present_unsurfaced 11, robust 6, absent 0; diagnostic surface frozen per #56 (`autoresearch/jspace_config.json`) |
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

**Tier 3 (reasoning) and CROP-derived:** the free-form axis and rationale path are
unlocked, but the current QMD rationale-replay strategy is rejected after p11/p12.
J-space-informed data selection remains an ACTIVE ROUTING PRIOR (Phase 5, #57): it
steers which lever a proposal targets using the frozen v5clean J-space diagnostic
surface, without itself being a measurable objective. The v5clean category ranking
(Natural Fertilizers → Soil Science → Sustainable Water Mgmt) now guides P13 toward
genuinely new supervision rather than another replay overlay.

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
