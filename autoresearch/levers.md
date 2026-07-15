# Lever inventory and classification (Phase 2.7, chainlink #21)

Single source of truth feeding **both** `autoresearch/finetune.py`'s parameter surface
(SEARCH DIMENSION rows only) and `program.md`'s priors (every row's status +
direction). Built from `PHASE_2.6_2.7_4_PLAN.md` Part B, the B3 settling diagnostics
(chainlink #20, `../benchmark-eval-workspace/results/B3_diagnostic_summary.md`), and a
direct read of the current `train.py` / `prepare_dataset_v3.py` parameter surface plus
the actual hyperparameters recorded in v3-clean's `trainer_state.json` /
`adapter_config.json`.

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
| Learning rate (peak) | SEARCH DIMENSION | Untested below default. v3-clean used `train.py`'s default (2e-4 peak, confirmed via `trainer_state.json` max observed LR 1.9999e-4) — the "5e-5" figure in `prepare_dataset_v3.py`'s docstring was aspirational and **was never actually applied**. Try lower LR next; no evidence yet either way. | `--lr` (float), range 2e-5–2e-4, currently exists in `train.py` |
| LR schedule | SEARCH DIMENSION | Cosine w/ 5% warmup used for v3-clean, untested vs linear/constant. No diagnostic evidence yet. | hardcoded `lr_scheduler_type="cosine"` / `warmup_ratio=0.05` in `build_training_args` — needs to become a flag |
| Epochs / early-stop | SEARCH DIMENSION | v3-clean ran fixed 6 epochs, no eval-based early stop (`eval_strategy="no"`). Untested whether fewer epochs avoids overfitting to the small (4400-pair) train set. | `--epochs` exists; early-stopping needs new implementation (currently no eval loop during training at all) |
| Batch size / grad-accum | SEARCH DIMENSION | v3-clean used batch=4, grad-accum=4 (effective 16). Untested. Low priority — unlikely to be the dominant lever given diagnostic 1's negative-headroom finding (see prose below). | `--batch-size`, `--grad-accum` exist in `train.py` |
| LoRA rank r / alpha | SEARCH DIMENSION | v3-clean used r=32, alpha=64 (2x ratio). Untested at other ranks. | `--lora-r`, `--lora-alpha` exist |
| LoRA dropout | SEARCH DIMENSION | Fixed at 0 for v3-clean. Untested — could help given the small dataset (4400 pairs) is a mild overfitting risk. | hardcoded `lora_dropout=0` in `load_model_with_lora` — needs a flag |
| LoRA target modules | SEARCH DIMENSION | v3-clean uses the full 7-module list (attn q/k/v/o + mlp gate/up/down). Untested narrower sets (e.g. attn-only). | hardcoded list in `load_model_with_lora` — needs a flag (e.g. `--lora-target-modules attn\|mlp\|all`) |
| LoRA vs full finetune | SETTLED CONSTANT | Use LoRA. Phase 4's stated compute budget (~5–20 min/run at 0.6B+LoRA → 30–100 experiments/night) assumes LoRA; full finetuning changes the compute-budget tier entirely and isn't compatible with the nightly-loop design. | n/a — not exposed as a knob |
| Weight decay | SEARCH DIMENSION | Fixed 0.01 for v3-clean. Untested. Low priority. | hardcoded `weight_decay=0.01` — needs a flag |
| Seq-len / packing | SEARCH DIMENSION (packing needs implementation first) | `max_seq_len` exists as a flag (2048, unused headroom since MCQ pairs are short). Packing is **not implemented at all** in `build_dataset` — this is a build-before-search item, not just a knob. | `--max-seq-len` exists; packing needs new code in `train.py`'s `build_dataset` |

---

## Tier 2 — Data / format

| Lever | Status | Direction / prior | Evidence | Knob |
|---|---|---|---|---|
| Gold-letter balance | **SETTLED CONSTANT** | Keep exactly balanced (25/25/25/25). Do not reintroduce imbalance. | Pre-v3 diagnostic: biased_letter error bucket 55.2%→0%, marginal entropy 1.009→1.683/2.0 bits after balancing (`summary_v3clean.md`). v3-clean's actual letter distribution is exact 25/25/25/25 (commit `85019a9`). | `_SHIFT_WEIGHTS` in `prepare_dataset_v3.py` already enforces this; keep as-is |
| Re-introducing letter bias | **FORBIDDEN** | Any lever move that skews the letter marginal away from ~25% each must be rejected by the bias guardrail (`diagnostics/selection_bias.py`, C4). | Same as above | guardrail, not a knob |
| Answer-position shuffle / per-question consistency augmentation | **SEARCH DIMENSION — highest-confidence target** | Increase multi-permutation coverage, but **targeted, not blanket**: B3 diagnostic 3 sized the augmentation-worthwhile subset to 321 items (20.0% of the benchmark — the `near_miss`+`toss_up` buckets), not the full 94.7% flip-any-fraction the old headline number implied. The dominant failure mode (74.7% `mostly_wrong`) won't be fixed by shuffling alone. | `../benchmark-eval-workspace/results/position_spread_breakdown.json` — exact `augmentation_seed_set` (321 question IDs) ready to consume | New flag on `prepare_dataset_v3.py`, e.g. `--extra-permute-ids PATH` reading the seed-set question IDs and generating additional cyclic-shift variants (beyond the uniform `_SHIFT_WEIGHTS` already applied) targeted at just those items |
| General replay ratio (farming-conversational : MCQ mix) | SEARCH DIMENSION — **knob now exists** (chainlink #26) | **Non-monotone — do not assume more MCQ (or more conversational) is better.** v3-clean's mix (4409 conversation pairs : 311×~5 MCQ-with-shifts pairs, before dedup) was just "whatever's in the two source files," not a designed choice. CROP's own replay-ratio result was non-monotone (Table 2: +CROP 0.676→+CQIA+CROP 0.709 overall but Difficult −0.075) — expect the same risk here. | `PHASE_2.6_2.7_4_PLAN.md` CROP citation; `85019a9` commit for the prior emergent ratio; verified against real data at ratio=0.5 and 0.97 (both branches, letter balance held) | `--conversation-ratio F` in `prepare_dataset_v3.py` (subsample-only — never upsamples/duplicates; MCQ-side subsampling stratified by answer letter to preserve the balance invariant). `None` (default) = old uncontrolled behavior, unchanged. |
| Chat template / prompt variants | PENDING-UNLOCK | Can't be scored without the free-form eval axis (2.6) as a second signal — MCQ accuracy alone can't distinguish "better prompt" from noise at this granularity. | — | — |
| SFT target format (`"B"` vs `"Answer: B"` vs sentence) | SEARCH DIMENSION | v3-clean hardcodes bare-letter output (`<think>...\n</think>\n{letter}`). Untested alternatives. Diagnostic 1's finding (peak knowledge probe 37.1% vs 75.2% generation, confident-wrong-dominated errors) suggests the model already commits confidently to a format — worth testing whether a more verbose target format changes what surfaces, but this is speculative, not diagnostic-backed. | `_cyclic_shift_pair`'s hardcoded `new_output = f"<think>\n{rationale}\n</think>\n{new_letter}"` | Needs a new flag on `prepare_dataset_v3.py` output formatting |
| MCQ-vs-cloze items | PENDING-UNLOCK | No cloze format exists in any data-generation script yet; would need new item authoring before it's even testable. | — | — |
| Distractor quality | **OUT OF SCOPE for `finetune.py`** | This is a data-*authoring*-time property owned by `scripts/generate_mcq_from_qmd.py` + the judge pipeline, not a training-time knob. Not part of the autoresearch loop's search surface. | — | — |
| Curriculum order | PENDING-UNLOCK | No difficulty signal exists on any training item (`difficulty` field in benchmark data is uniformly `"unknown"`). Can't order by difficulty until one is computed. | checked `qwen3-0_6b-v3-clean_results.json` per-question `difficulty` field | — |
| Synth-item (QMD MCQ) count | SEARCH DIMENSION | v3-clean used all 311 available QMD-generated questions. Untested whether more synthetic items help or just adds more of the same distribution (diagnostic 1 argues data quantity has *some* ceiling, but doesn't rule out more coverage helping). | `85019a9` — 311 raw → ~1555 with shift weighting | `data/sources/qmd_mcq_all_phase3.jsonl` size is currently the ceiling; would need a `generate_mcq_from_qmd.py` re-run at scale, not a `prepare_dataset_v3.py` flag |

---

## Tier 3 — Reasoning (ALL PENDING-UNLOCK — free-form axis required, per plan)

| Lever | Status | Direction / prior (once unlocked) | Evidence |
|---|---|---|---|
| Rationale / CoT targets | PENDING-UNLOCK | **Caveat from B3, read before unlocking:** diagnostic 1's negative real-gen headroom (peak knowledge 37.1% vs 75.2% generation, `headroom=-0.381` on the full 1603Q benchmark) means this is *not* a guaranteed free win — whatever "unsurfaced" signal diagnostic 2 finds under a suppressed-reasoning prompt does not clearly persist once the model actually reasons. If/when unlocked, **aim first at Natural Fertilizers (65.3% acc), Sustainable Water Management (67.9%), Soil Science (68.9%)** — diagnostic 2's lowest-accuracy, majority-"unsurfaced" categories. Sustainable Water Mgmt is a distinct sub-case (signal arrives at layer 21/27 vs 1–3/27 elsewhere — late/weak, may need different treatment). | B3 diagnostics 1 & 2, full results in `../benchmark-eval-workspace/results/B3_diagnostic_summary.md` |
| Reflection-data proportion | PENDING-UNLOCK | Unmeasurable without a free-form quality signal. | plan doc |
| J-space-informed data selection | PENDING-UNLOCK | Unmeasurable without a free-form quality signal. | plan doc |
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

**Tier 3 (reasoning) and CROP-derived:** uniformly PENDING-UNLOCK behind Phase 2.6's
free-form eval axis, with one exception worth flagging early: diagnostic 1's negative
real-gen headroom is a soft warning against over-investing in CoT/rationale levers as
an assumed win. When 2.6 lands and these unlock, diagnostic 2's category ranking
(Natural Fertilizers → Sustainable Water Mgmt → Soil Science) gives a concrete starting
point rather than searching blind across all 9 categories at once.

---

## Open engineering gaps this table surfaces (for C1/C2, not this issue)

- ~~`prepare_dataset_v3.py` needs a conversation:MCQ ratio knob~~ — done, chainlink #26
  (`--conversation-ratio`).
- `prepare_dataset_v3.py` needs a targeted-permutation flag consuming the 321-item
  augmentation seed set, distinct from the existing uniform `_SHIFT_WEIGHTS`.
- `train.py` needs flags for: LR schedule type, warmup ratio, LoRA dropout, LoRA target
  module set, weight decay, and sequence packing (packing needs new code, not just a flag).
- `train.py` has no eval loop during training (`eval_strategy="no"`) — early-stopping as
  a lever needs that wired up first.
