# Phase 2.6 → 2.7 → 4: Free-form Eval, Lever Definition, Frozen Autoresearch Harness

## Context

MCQ contamination remediation (Phases 0–5) is done; v3-clean beats v2-clean (73.0% vs 42.55%
generation accuracy). The next objective is the autoresearch loop the original plan called
"Phase 4," adapted from Karpathy's design: a frozen `prepare.py`, an agent-edited `train.py`,
a human-iterated `program.md`, a fixed compute budget, and a continuous `val_bpb`-style metric
chosen to stay honest under the moves the agent may make.

A search loop is exactly three things — an **objective**, an **action space**, and an
**iteration budget** — and **search amplifies whatever is broken in the objective, overnight,
unsupervised.** (Concretely: on the pre-Phase-0 benchmark, "always answer B" scored 71.07% vs
the model's 66.48% — the argmax of that objective was literally letter bias.) Building the loop
therefore has three prerequisites, done in order:

1. **Phase 2.6 — free-form eval** (the *second objective axis*). MCQ accuracy is (a) noisy
   (binomial; dev-split SE ≈ 1.8%, ±3.5% band — a naive loop accepts pure noise and builds on
   it) and (b) structurally blind to reasoning. The CROP paper (verified from the PDF) makes
   this concrete: CROP's *training* set is richly free-form and multi-task (210K single-turn +
   1,871 multi-turn dialogues — closed/open-book Q&A, summary, NER, event extraction, multi-turn
   step-back reasoning), but its *benchmark* is deliberately MCQ-only, chosen "to avoid disputes
   arising from subjective questions." The free-form rubric eval is exactly the axis CROP
   declined to build — it makes reasoning levers legible and is the Goodhart guard.
2. **Phase 2.7 — lever definition** (the *action space*, and the piece not yet done). The
   diagnostics exist but nobody has converted them into a concrete, classified lever inventory.
   Diagnosis mostly works by **removing** dimensions (turning them into settled constants) and
   supplying **direction**, not by adding knobs — and that conversion has to be an explicit
   artifact, because the search-dimension subset *is* `finetune.py`'s parameter surface and the
   classification *is* `program.md`'s priors. Must precede the harness build.
3. **Phase 4 — the frozen harness** (objective wiring + gate + guardrails + `program.md`).

User decisions so far (AskUserQuestion):
- 2.6 item authoring: **generate + human-verify** via the existing quality pipeline.
- Grader: **two-tier** — Haiku 4.5 frequent, Sonnet audits verdicts / catches drift.

### v3-clean diagnostic facts that drive lever classification (from `summary_v3clean.md`)
- Generation accuracy **73.0%**; forced-empty-`<think>` logprob accuracy only 32.6% (the probe
  shape is blind to the real number — do not read headroom off it).
- Error buckets under the probe: **biased_letter 55.2%→0%**, confident_wrong 0%→43.0%,
  uniform 13.0%→24.3%. Marginal entropy **1.009→1.683** of 2.0 bits.
- Linear probe: peak knowledge acc 23.0%, knowledge-prediction gap **−0.006** (closed).
- Logit lens wrong-split: **present-but-barely-not-surfaced** (first-correct layer 23/27).
- **Mean per-question spread 0.982→0.947, essentially flat** — the one residual weakness.

## Reusable infrastructure (do NOT reinvent)
- `scripts/judge_training_data.py` — OpenRouter→Claude LLM-judge, 5-dim rubric scoring,
  keep/review/reject, resume, workers. Rubric-grader scaffold.
- `benchmark-workspace/quality/`: `llm_judge.py`, `self_refine.py`, `rubrics/`,
  `judge_calibration.py` (anchor calibration), `filter_contamination_dedup.py` (n-gram dedup).
- `benchmark-eval-workspace/diagnostics/*` (calibration, selection_bias, linear_probes,
  logit_lens, forgetting), `train.py`, `merge_adapter.py`, `prepare_dataset_v3.py`.

---

## Part A — Phase 2.6: Free-form / multi-hop evaluation  (new dir `eval_freeform/`)

- **A1. Items** (~150–250; 5 types × ~30–50: multi-hop factual, diagnostic reasoning, free-form
  explanation, constraint satisfaction, quantitative). Schema `{id, type, question,
  rubric:[{point,weight}]×3–5, reference_answer, source_doc, held_out:true}`. Generate +
  human-verify via `benchmark-workspace/quality/` + `self_refine.py` from held-out source docs.
  → `eval_freeform/items.jsonl`.
- **A2. Contamination wall.** Exclude any source doc that produced an item from finetuning (or
  strip+flag the specific Q/A); 13-gram overlap report vs training corpus via
  `filter_contamination_dedup.py`; quarantine high-overlap. → `contamination_report.json`.
- **A3. Two scorers.** `score_rubric.py` (grader returns per-point true/false + justification;
  score = weighted fraction; **two-tier** Haiku-grade + Sonnet-audit-anchors; temp 0; grader
  model+version pinned; 5–10 anchors via `judge_calibration.py`). `score_refll.py` (mean
  teacher-forced LL of the reference answer — deterministic, grader-free, high-frequency proxy;
  LL-up-while-rubric-down = surface-learning alarm).
- **A4. Splits.** Stratified `dev.jsonl` (~60%, loop-visible) / `test.jsonl` (~40%, sealed).

---

## Part B — Phase 2.7: Lever Definition & Action-Space Specification  (THE MISSING STEP)

Goal: produce `autoresearch/levers.md` — the single source of truth that feeds both
`finetune.py`'s parameter surface (search dimensions only) and `program.md`'s priors
(classification + direction). This is analysis + authoring, not just a list.

### B1. Enumerate the full candidate inventory
Tier 1 (hyperparams): LR + schedule, warmup, epochs/early-stop, batch/grad-accum, LoRA
rank/α/dropout/target-modules, LoRA-vs-full, weight decay, seq-len/packing.
Tier 2 (data/format): farming:conversational:replay mix, answer-position shuffle rate,
gold-letter balance, chat template/prompt variants, SFT target format (`"B"` vs `"Answer: B"`
vs sentence), MCQ-vs-cloze items, distractor quality, curriculum order, synth-item count.
Tier 3 (reasoning): rationale/CoT targets, reflection-data proportion, J-space-informed data
selection, free-form score as a weighted objective term.
CROP-derived (verified from PDF): **auxiliary-task mixing** (summary/NER/extraction/open-book,
not just Q&A); **multi-turn / step-back SFT proportion**; **open-book/RAG-grounded vs closed-book
targets**; **general-instruction replay ratio — empirically NON-monotone** (CROP Table 2,
Qwen1.5-7B: +CROP 0.676 overall → +CQIA+CROP 0.709 overall but **Difficult −0.075**).

### B2. Classify each lever (the core deliverable)
For every lever assign one status + a directional prior + the diagnostic evidence + a concrete
knob (param name, type, range) in `finetune.py`/`prepare_dataset_v3.py`:
- **SETTLED CONSTANT** — diagnosis resolved it; remove from search. e.g. **gold-letter balance**
  (biased_letter 55.2%→0%, entropy 1.009→1.683) → keep balanced, not a knob.
- **FORBIDDEN** — a guardrail owns it. e.g. re-introducing letter bias (bias guard).
- **SEARCH DIMENSION + prior** — live knob with a direction. e.g. **answer-position shuffle /
  per-question consistency augmentation** — the #1 open target (spread flat at 0.947); prior =
  increase multi-permutation coverage. Replay ratio = search but **non-monotone** prior (don't
  assume more is better).
- **PENDING-UNLOCK** — cannot be classified until a prerequisite lands. e.g. all Tier-3
  reasoning/CoT levers are unmeasurable until Phase 2.6's free-form axis exists; classify them
  once dev free-form scores are available.

### B3. Settling diagnostics still needed (fill the essay's `<FILL>` priors)
Some statuses can't be set from existing results — schedule these small runs first:
- **Knowledge-probe headroom under the real generation prompt shape** (existing probe used
  forced-empty-`<think>`, blind to the 73%). Decides whether data-quantity levers have any
  ceiling vs format/calibration — currently unsettled. Extend `diagnostics/linear_probes.py`.
- **Per-category failure localization** (which of the 9 categories fail, and whether the correct
  intermediate concept is absent vs unsurfaced) → aims aux-task / CoT levers at specific topics.
- **Per-question position-spread breakdown** (which items flip under permutation) → sizes the
  position-invariance lever and seeds its augmentation set.

### B4. Output
`autoresearch/levers.md`: a table (lever · status · direction · evidence · knob · range) plus a
short prose prior per Tier. Search-dimension rows define exactly what `finetune.py` exposes;
constant/forbidden/pending rows become `program.md`'s "do not touch / do not rediscover" priors.

---

## Part C — Phase 4: frozen autoresearch harness + `program.md`  (new dir `autoresearch/`)

- **C1. Frozen/editable split (a security property here, stricter than Karpathy's).** FROZEN
  (checksum-guarded manifest; a pre-run check aborts on any change): all scorers, dev/test
  splits + sealed test sets (MCQ + free-form), baselines (random, majority, base 0.6B), grader
  prompt/anchors/pinned version, guardrail thresholds, accept/revert logic, compute budget.
  EDITABLE: `finetune.py` only, parameterized over exactly the B4 search dimensions.
- **C2. Continuous objective.** Primary = `w1·mean gold-token LL (MCQ dev) + w2·mean ref-answer
  LL (free-form dev)` (weights frozen). Accuracy + rubric % are reported secondary only. New
  `score_mcq_ll.py`.
- **C3. Gate + budget.** Accept iff primary Δ > ~2·SE (or replicates on 2 seeds) AND no guardrail
  fired; else revert. Fixed compute budget/run (~5–20 min at 0.6B+LoRA → 30–100 experiments/night).
- **C4. Guardrails.** Forgetting (`diagnostics/forgetting.py`) · bias (`selection_bias.py`) ·
  generation (free-form rubric drop) · grader-drift (anchors move → pause) · sealed test touched
  once at the end.
- **C5. Loop + `program.md`.** `run_loop.py`: proposer reads `program.md` + journal + `levers.md`
  → one `finetune.py` edit → budgeted run → frozen eval → gate → append `journal.jsonl`.
  `program.md` = diagnosis-as-manual: editable surface, forbidden moves, continuous-metric rule,
  guardrails, and the B4 lever classification + priors verbatim. Driven by a Claude Code agent.

---

## Chainlink hygiene (first, quick)
Close #6 (Phase 4 rebuild done) and epic #1 (all subphases done). Open a new epic
"Phase 2.6 / 2.7 / 4 autoresearch" with subissues mirroring Parts A–C.

## Files
- New: `eval_freeform/{items,dev,test}.jsonl`, `eval_freeform/{score_rubric,score_refll}.py`,
  `eval_freeform/contamination_report.json`; `autoresearch/{levers.md, finetune.py,
  score_mcq_ll.py, run_loop.py, program.md, journal.jsonl, frozen_manifest.json, configs/}`;
  `scripts/split_benchmark.py`.
- Modified: `prepare_dataset_v3.py` (shuffle/mix/aux-task/replay knobs from B4), `train.py`
  (LR flag if missing), `diagnostics/linear_probes.py` (real-prompt headroom, B3),
  `MCQ_DIAGNOSTIC_PLAN.md` (Phase 2.6/2.7/4 sections).

## Verification
1. **2.6 grader:** Haiku verdicts within tolerance of hand grades on anchors; Sonnet audit agrees
   (drift guard sane). Contamination report: 0 high-overlap items survive; no 2.6 source_doc in
   training corpus.
2. **2.7 levers:** every enumerated lever has a status + evidence + knob in `levers.md`; each
   search-dimension row maps 1:1 to a `finetune.py` param; no lever left unclassified except
   explicit PENDING-UNLOCK ones; the B3 settling diagnostics have run and their numbers are
   filled into the priors.
3. **Continuous metric sanity (REVISED 2026-07-16 — see `results/C2_metric_sanity.md` and
   `AUTORESEARCH_INFRASTRUCTURE_REMAINING_PLAN.md`):** The MCQ option-set gold-token-LL term must
   rank v3-clean > v2-clean (same direction as accuracy) with visibly lower run-to-run variance
   than accuracy. **Free-form ref-LL is NOT a valid cross-approach comparator** and must not be
   used to rank v3-clean vs v2-clean: v2/v3 are different prompt/training surfaces, and teacher-
   forced reference-answer LL rewards base-like prose continuation over v3's chat-template +
   `<think>` conditioning (v2-clean scores −4.370 vs v3-clean −4.993 on dev, backwards from the
   73.0% vs 42.55% generation-accuracy ground truth). Treat ref-LL as an **intra-lineage proxy
   only** — valid for comparing v3-lineage loop variants that share prompt/template conventions,
   never as an a-priori cross-model quality metric. Cross-approach quality sanity is established by
   generation accuracy and/or the rubric grader, not ref-LL.
4. **Gate/security dry run:** known-good change accepts; noise/no-op reverts; editing any frozen
   file aborts via checksum. One full loop iteration writes a journal row (change, diff, metrics,
   decision, rationale) with reproducible artifacts. Journal/grep confirms sealed test untouched.

## Deferred
SAE training (Stage D) — only if the loop plateaus or spurious features are suspected; re-check
for a released Qwen3-0.6B SAE first. #10 tool-calling, #9 bilingual QA.
