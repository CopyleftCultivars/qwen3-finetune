# MCQ Diagnostic Observability Plan — Qwen3-0.6B Fine-Tune

## ⚠️ CRITICAL FINDING (2026-07-09): Benchmark contamination in v2 and v3 training data

**All eval and diagnostic numbers produced for v2 and v3 so far measure memorization of the
eval set, not generalization.** Do not cite v2/v3 benchmark accuracy, logprob accuracy, or any
Stage A/B diagnostic result as evidence of model capability until this is fixed and both
versions are rebuilt from clean sources.

### What was found

`prepare_dataset.py` (the v2 dataset-prep script) has a documented usage example that passes
`benchmark-eval-workspace/benchmark_combined.jsonl` — **the eval benchmark itself** — directly
as its `--mcq` argument. `_load_mcq()` (Source 2) converts every benchmark question into 5
training pairs: 1 open-ended Q&A pair + 4 cyclic option-position-shift MCQ variants (same
question stem, same option text, same correct answer — only the A/B/C/D position and the
`<think>` rationale differ). This was verified empirically, not assumed:

| File | Total records | Unique MCQ question stems | Overlap with `benchmark_combined.jsonl` (1603 q) |
|---|---:|---:|---:|
| `data/train.jsonl` (v2 training data) | — | 1603 | **1603 / 1603 (100%)** |
| `data/val.jsonl` (v2 "held-out" split) | 1052 | 556 | 556 / 1603 (34.7%) |
| `data/sources/mcq_with_rationale.jsonl` | 9271 | 1603 | **1603 / 1603 (100%)** |
| `data/sources/mcq_with_rationale.repaired.jsonl` | 9456 | 1603 | **1603 / 1603 (100%)** |
| `data/train_v3.jsonl` (v3 training data) | — | — | **1601 / 1603 (99.9%)** |
| `data/val_v3.jsonl` (v3 "held-out" split) | 971 | 510 | 510 / 1603 (31.8%) |

The train/val split offers **no real protection**: because every benchmark question has 4
cyclic-shift siblings, a question whose shift=0 variant lands in `val.jsonl` almost always has
its shift=1/2/3 siblings in `train.jsonl`. The model can see 3 of 4 phrasings of a "held-out"
validation question during training.

### Propagation chain

```
benchmark_combined.jsonl (THE EVAL SET)
  └─▶ prepare_dataset.py --mcq benchmark_combined.jsonl   (v2 build)
        └─▶ data/train.jsonl, data/val.jsonl               (v2 training data — 100% contaminated)
              └─▶ scripts/add_rationale.py                 (adds genuine rationale, passthrough of contamination)
                    └─▶ data/sources/mcq_with_rationale.jsonl
                          └─▶ scripts/repair_mcq_rationales.py
                                └─▶ data/sources/mcq_with_rationale.repaired.jsonl
                                      └─▶ prepare_dataset_v3.py --mcq-rationale (default)   (v3 build)
                                            └─▶ data/train_v3.jsonl, data/val_v3.jsonl      (v3 training data — 99.9% contaminated)
```

Neither `add_rationale.py`, `repair_mcq_rationales.py`, nor `prepare_dataset_v3.py` reference
`benchmark_combined.jsonl` directly — they inherited the contamination from `data/train.jsonl`,
which was the actual point of entry.

### This also invalidates prior diagnostics, not just training

Every diagnostic script (`calibration.py`, `logit_lens.py`, `linear_probes.py`,
`selection_bias.py`, `forgetting.py`) takes `--dataset benchmark_combined.jsonl`. Since both v2
and v3 were trained directly on that file, **the Stage A/B findings below (median first-correct
layer, "present-not-surfaced" diagnosis, knowledge-probe accuracy, logprob vs. generation gap)
were all measured on memorized questions.** They may still be internally consistent (the
relative v2→v3 delta reflects a real change in how well the model reproduces memorized answers
in the right format), but they are not evidence of the diagnosed failure modes on genuinely
unseen agricultural questions. `forgetting.py`'s "100 held-out questions, not in train.jsonl"
(Stage B, below) is the one existing measurement that used a genuinely non-benchmark set — it
should be treated as more trustworthy than the others, though it wasn't designed to measure MCQ
accuracy specifically.

### Second contamination vector: QMD/QTMD corpus overlap (confirmed, not yet triggered)

`scripts/generate_mcq_from_qmd.py` walks
`[LOCAL_PATH]` directly.
`benchmark-workspace/generate_mcqs_rag.py` (which produced the original benchmark questions)
retrieves from "the QMD corpus" via the `qmd` CLI — the same corpus family. This path was
correctly **not used** for v3 (`data/sources/qmd_mcq_generated.jsonl` does not exist on disk),
so it hasn't contributed contamination yet — but any future run of
`generate_mcq_from_qmd.py` risks generating near-duplicate or literally-duplicate questions to
`benchmark_combined.jsonl`, for the same reason (2) did: same source documents.

### Decision

Redo v2 and v3 from sources that never reference `benchmark_combined.jsonl`, directly or
transitively. See **Redo Plan** below. See chainlink issues for tracked implementation steps.

### Root cause — how this happened

Not a single bad decision; a scoping gap across two separate, individually-reasonable steps.
Stage B correctly diagnosed "present-not-surfaced" from `data/train.jsonl`'s MCQ pairs having
thin, label-restating think blocks (`<think>The correct answer is A. ...</think>`). The fix —
`scripts/add_rationale.py` replacing those with genuine 1-3 sentence rationales — is sound
engineering for that specific problem. But that fix was scoped to *rationale quality only*; it
operated on `data/train.jsonl` as a given, without re-examining where its underlying MCQ
*questions* came from. That provenance question belonged one step earlier, at
`prepare_dataset.py`'s `--mcq benchmark_combined.jsonl` invocation — which predates
`add_rationale.py` in the pipeline and wasn't back in view by the time the rationale-quality
work was being scoped. Lesson for future data-pipeline work: before improving any derived
artifact, trace its lineage back to the original source files, not just its immediate input.

---

## Redo Plan

**Phase 0 — Quarantine (done in this pass).** Mark contaminated scripts and data files with
loud warnings so they can't be silently reused as training inputs. Scripts get docstring
warnings; `prepare_dataset.py` additionally gets a runtime guard that refuses to run against a
`--mcq` path matching known benchmark filenames.

**Phase 1 — ✅ DONE (2026-07-10). Rebuild a clean baseline (v2-clean): conversations only.**
Ran `prepare_dataset.py` with **only** `data/sources/gemma4/hf_dataset/train.jsonl`
(expanded 2865 → 3998 conversations first, see below) — no MCQ data of any kind. 3,669
pairs (3,302 train / 367 val), verified 0% overlap with `benchmark_combined.jsonl`. Trained
26:33 (default hyperparameters), merged to `outputs/merged-model-v2-clean` (1137MB).

**Phase 2 — ✅ DONE (2026-07-10). Evaluate v2-clean.** Full Stage A/B suite run against the
full 1603-question benchmark. Full writeup with reasoning: `summary.md`. Headline numbers:

| Metric | Value |
|---|---|
| Generation accuracy | 42.55% (682/1603) |
| Logprob accuracy | 31.8% (509/1603) |
| Letter bias | 77.2% of logprob top-choices were 'A' (biased_letter errors: 55.2%) |
| Logit lens (wrong split) | median first-correct layer 28/27 (never), 26/50 never-correct |
| Linear probes (finetuned peak) | 25.8% at layer 24 — *below* base model's 30.7% at layer 17 |

First trustworthy numbers this project has produced. Key finding: v2-clean's dominant
failure mode is **lack of MCQ-format exposure entirely** (severe letter bias, near-chance
linear decodability), not "present-not-surfaced" — the diagnosis that drove the original
(contaminated) v2→v3 rationale-quality fix doesn't describe what v2-clean actually needs.
Phase 3/4 should prioritize simply *having* balanced MCQ training data.

**Phase 3 — Build a genuinely clean MCQ training source.** As of 2026-07-10, `qmd` is
working again (a Node ABI mismatch from an `npm i` update was fixed via
`npm rebuild better-sqlite3` in the `@tobilu/qmd` package directory — no revert needed).
Its indexed collections are organized by the *same 9 topics* as the benchmark categories:

| Collection | Files |
|---|---:|
| Permaculture | 53 |
| Organic_Farming | 110 |
| Natural_Fertilizers | 143 |
| Regenerative_Agriculture | 59 |
| Agroecology | 69 |
| Soil_Science_and_Microbiology | 115 |
| Sustainable_Water_Management | 65 |
| Integrated_Farming_Systems | 56 |
| Indigenous_and_Traditional_Knowledge | 73 |

Confirmed directly, not just inferred: `qmd collection show Permaculture` resolves to
`[LOCAL_PATH]` — these collections
live *inside* `benchmark-workspace`, the project that built the benchmark. This is not "same
corpus family," it's almost certainly the exact source documents the benchmark questions were
written from. The contamination problem was never "QMD-derived content is forbidden" — it was
"the literal eval file was fed into training." Generating fresh MCQs from these source
documents is fine; duplicating benchmark *questions* is not. Given the source-document overlap
is now confirmed high, the dedup step (below) is essential, not a nice-to-have.

**Reuse the benchmark's own generation rigor, don't reinvent a simpler pipeline.**
`benchmark-workspace/generate_mcqs_rag.py` + `quality/` already implement a real quality
system: per-topic QMD retrieval → LLM generation → 5-dimension LLM judge scoring against
per-category rubrics (`quality/rubrics/`) → composite-score keep/review/reject → regeneration
loop for failing topics (up to 3 rounds, `REGENERATION_PIPELINE.md`) → generate→critique→correct
self-refinement for borderline items (`quality/self_refine.py`, MCQG-SRefine method, Yao et al.
NAACL 2025) → final packaging. Training data deserves the same quality bar as eval data, not a
cheaper ad-hoc generator. `generate_mcqs_rag.py` already has fuzzy dedup infrastructure
(`_load_existing_stems`/`_is_duplicate`, `difflib.SequenceMatcher` fuzzy match) built to avoid
regenerating duplicates within a category's own regen loop — the plan is to reuse that same
mechanism, pointed at `benchmark_combined.jsonl`'s question stems as an *additional* forbidden
set (on top of the category's own accepted/rejected stems), so nothing generated for training
can fuzzy-match an eval question.

Concrete plan:
  - **3a. Fresh QMD-generated MCQs.** Run the existing rigor pipeline (`generate_mcqs_rag.py`
    + `quality/*`) pointed at each `qmd://<Category>/` collection, generating new questions
    distinct from what's already in the benchmark — with the dedup set extended to include
    `benchmark_combined.jsonl`'s question stems, not just the category's own existing items.
  - **3b. Conversation-derived MCQs (new source, not in the original 6-phase plan).** Also
    generate MCQs *from* `data/sources/gemma4/hf_dataset/train.jsonl` (the trusted,
    already-vetted conversation content used for v2-clean) — same judge/refine rigor. This
    source has zero benchmark-corpus overlap risk by construction (it isn't derived from the
    QMD corpus the benchmark was built from), and reinforces facts the model is already
    learning conversationally, in multiple-choice form — directly targeting v2-clean's actual
    failure mode (MCQ-format unfamiliarity), not just adding more raw facts.
  - Mechanical dedup against `benchmark_combined.jsonl` (exact question-text match, minimum
    bar) applies to *both* 3a and 3b before anything is used for training.

**Design principle carried over from v2/v3 and reconfirmed:** the deployed model needs to
answer bare MCQ questions correctly (benchmark eval shape — no system prompt) *and* follow the
Natural Farming Fertilizer Assistant persona during normal conversational use (system-prompt-
wrapped). This is already how the pipeline works — `_load_mcq()`/MCQ training pairs are bare
instruction/output with no system-prompt wrapper, matching eval format, while conversation
pairs retain their system-prompt handshake turn where present. Keep it that way; don't
accidentally wrap MCQ training pairs in the persona system prompt, which would mismatch the
eval format and reintroduce a version of the format-mismatch problem Stage A/B already flagged.

**Phase 4 — Rebuild v3-clean.** Gemma4 conversations (3a) + the clean MCQ sources from Phase 3
(3a fresh-generated + 3b conversation-derived). This *does* include a QMD-sourced stream now
(unlike the original Phase 4 draft, which deferred all QMD use to Phase 6) — the original
"defer QMD" reasoning was really "defer benchmark-corpus risk," and that's now handled by the
dedup step instead of a blanket QMD ban. Phase 6 below is superseded/folded into this phase.

**Phase 5 — ✅ DONE (2026-07-13). Evaluate v3-clean, compare fairly to v2-clean.** Same
Stage A/B suite (calibration, logit lens, linear probes — full 1603-question benchmark;
`selection_bias.py` skipped for both checkpoints, matching v2-clean's actual eval scope,
see chainlink #13 for a dedicated follow-up). Full write-up: `summary_v3clean.md`.
Headline: generation accuracy 42.55% (v2-clean) → **73.0%** (v3-clean), +30.5pp,
confirming the Phase 2 diagnosis that lack of MCQ-format exposure was v2-clean's dominant
failure mode. Letter bias reshaped rather than eliminated under the forced-empty-think
probe (A-win 77.2%→54.6%; `biased_letter` errors 55.2%→0% but `confident_wrong` 0%→43.0%).
Linear-probe knowledge-prediction gap essentially closed (−0.131→−0.006), and logit lens
shows a present-but-barely-not-surfaced pattern on the wrong split (median first-correct
layer 23/27) rather than v2-clean's knowledge-absent verdict. This is the first
trustworthy v2→v3-style comparison this project has produced.

**Phase 6 — superseded (2026-07-10), folded into Phase 3/4.** Originally: "add QMD-generated
MCQs only after a solid non-QMD baseline exists." Revised understanding: QMD-sourced content
was never the actual risk (it's the same class of natural-farming material the benchmark
itself represents) — the risk was training on the literal benchmark file, which the dedup
step in Phase 3 handles directly. No reason to gate QMD-derived generation behind a separate
checkpoint anymore; it's part of Phase 3/4 now.

**Future, out of scope until the current plan (Phases 1-5) is complete: tool-calling.** The
model doesn't currently have tool-calling capability, and evaluating/training for it would
need its own separate benchmark (distinct from the MCQ benchmark this plan is about). Not
started — revisit once v3-clean's limits are actually known.

The existing `scripts/TODO_mcq_rationale_audit.md` (auditing refusal-style rationales in
`mcq_with_rationale.repaired.jsonl`) is superseded by this plan — that file is being rebuilt
from a clean source in Phase 3, so auditing its current (contaminated) contents is no longer
useful. Its content is preserved in that file for reference.

---

## Workspace Layout

| Purpose | Path |
|---|---|
| MCQ question generation (original benchmark) | `[LOCAL_PATH]` |
| Eval harness + results | `[LOCAL_PATH]` |
| QMD/QTMD source corpus | `[LOCAL_PATH]` |
| v2 fine-tune workspace | `[LOCAL_PATH]` |
| v2 merged model (contaminated) | `[LOCAL_PATH]` |
| v3 merged model (contaminated) | `[LOCAL_PATH]` |
| v2 training data (contaminated) | `[LOCAL_PATH]` |
| v3 repaired MCQ/rationale source (contaminated) | `[LOCAL_PATH]` |

**Dataset:** `benchmark-eval-workspace/benchmark_combined.jsonl` — 1603 questions, 9 categories.
Schema: `{question_id, question, A, B, C, D, answer, topic, category, difficulty}`

**Conda env for all diagnostic scripts:** `marker`

---

## Three Possible Failure Modes (and how to tell them apart)

| Mode | Symptom | Cheap test |
|---|---|---|
| **Selection/token bias** | Same letter wins regardless of content | Cyclic permutation test |
| **Format/symbol-binding** | Knows content but can't route to letter | cloze_acc >> symbol_acc |
| **Knowledge absent** | High entropy (uniform) on every option | Entropy distribution |
| **Present-not-surfaced** | Correct answer decodable from hidden states | Linear probes (Stage B) |
| **Catastrophic forgetting** | General domain accuracy drops vs base | Base vs finetune diff (Stage B) |

---

## Historical Record — v1→v2 and v2→v3 (contaminated; kept for process reference only)

The findings below were real measurements at the time, but per the Critical Finding above, they
were measured on data the models were trained on. Keep this section as a record of what
diagnostic infrastructure exists and how it's invoked — re-run all of it against v2-clean and
v3-clean once available (Phases 2 and 5).

### Stage A — Triage (v2 model, 2026-07-01)

| Metric | v1 (old) | v2 (retrained) |
|---|---|---|
| Logprob accuracy | 53.4% | 78.3% |
| PriDe debiased accuracy | ~55% | 79.8% |
| confident_wrong errors | 25.1% | 8.7% |
| uniform errors | 21.5% | 13.0% |
| D letter marginal | 1.6% | 16.9% |
| Marginal entropy | low | 1.928/2.0 bits |
| Mean permutation spread | 0.681 | 0.451 |

```bash
conda run -n marker python -m diagnostics.calibration \
  --model <merged-model-path> \
  --dataset benchmark_combined.jsonl

conda run -n marker python -m diagnostics.selection_bias \
  --model <merged-model-path> \
  --dataset benchmark_combined.jsonl
```

### Stage B — Localize (v2 model, 2026-07-01)

Requires TransformerLens: `conda run -n marker pip install transformer-lens scikit-learn matplotlib pandas tqdm`.
Qwen3-0.6B via TransformerLens: 28 layers, hidden 1024, 16 query / 8 KV heads.

**Logit lens** (50 correct + 50 wrong questions): correct-split median first-correct layer 2/27;
wrong-split median first-correct layer 8/27, never-correct 0/50 — diagnosed as
present-not-surfaced.

```bash
conda run -n marker python -m diagnostics.logit_lens \
  --model <merged-model-path> \
  --dataset benchmark_combined.jsonl --n-correct 50 --n-wrong 50
```

**Linear probes** (300 questions at the time — see note below): base peak knowledge acc 0.163
(layer 23); fine-tuned peak knowledge acc 0.337 (layer 26), gap ≈ 0; generation acc 56.3% vs
logprob acc 78.3%.

```bash
conda run -n marker python -m diagnostics.linear_probes \
  --base unsloth/Qwen3-0.6B \
  --finetuned <merged-model-path> \
  --dataset benchmark_combined.jsonl --max-questions 300
```

> Note (2026-07-09): the 300-question limit above was a deliberate speed shortcut, not a
> dataset-size constraint — the full 1603-question benchmark already existed at the time. A
> full-1603 rerun on both v2 and v3 (2026-07-09, in `benchmark-eval-workspace`) found peak
> knowledge accuracy actually *higher* for v2 (52.3%) than v3 (46.1%), with the
> knowledge-prediction gap shrinking from +0.162 (v2) to +0.042 (v3) — i.e. v3 didn't encode
> more, it surfaced what it had more reliably. This is a real, reproducible measurement of the
> *training* effect, but see the Critical Finding above for why it isn't evidence of
// generalization.

**Forgetting** (100 held-out questions **not in `train.jsonl`** — the one genuinely
uncontaminated check that exists so far): base accuracy 20.0%; fine-tuned accuracy 74.0%; no
forgetting detected.

```bash
conda run -n marker python -m diagnostics.forgetting \
  --base unsloth/Qwen3-0.6B \
  --finetuned <merged-model-path> \
  --train-data data/train.jsonl \
  --dataset benchmark_combined.jsonl
```

### MCQ Rationale Repair (2026-07-08) — superseded, see Phase 3 above

`scripts/repair_mcq_rationales.py` repaired 188 missing MCQs, replaced 337 refusal-style
rationales, and cleaned 405 embedded option-label issues in
`data/sources/mcq_with_rationale.jsonl` → `.repaired.jsonl` (9459 records, 0 remaining format
errors at the time). This repair was sound *engineering*, but it repaired a contaminated source
— redo against the clean source built in Phase 3.

### v3 Dataset Assembly (2026-07-08) — superseded, see Phase 4 above

```bash
python prepare_dataset_v3.py \
  --conversations data/sources/gemma4/hf_dataset/train.jsonl \
  --output-dir data --split 0.9 --seed 42
```
(defaulted `--mcq-rationale` to the contaminated repaired source; `--qmd-mcq` intentionally
omitted). Produced `data/train_v3.jsonl` (8737 rows) / `data/val_v3.jsonl` (971 rows), balanced
MCQ letter distribution (~25% each). The balancing and filtering logic is reusable — only the
MCQ *source* needs to change for the redo.

---

## Diagnosis → Fix Table

| Stage A signal | Failure mode | Training fix | Phase-4 sweep knob |
|---|---|---|---|
| Letter marginal skewed > 5 pts from uniform | Selection bias | Balance gold-letter distribution; shuffle options at train time | data aug ratio; symbol set |
| cloze_acc >> symbol_acc (> 5 pts gap) | Format/symbol-binding | Train with eval format; explicit "Answer: X" targets | prompt template |
| Both clean, entropy high (uniform) | Knowledge absent | More domain Q&A; curriculum; more epochs | dataset size/mix; epochs |
| Stage B: knowledge_probe >> gen_acc (> 10 pts) | Present-not-surfaced | Format-matched training; lower LR | format-match weight |
| Stage B: accuracy drop > 5 pts on held-out | Catastrophic forgetting | Lower LoRA rank; lower LR; replay general data | LoRA rank; LR; replay fraction |

---

## Stage C — Causal Confirmation (optional, not yet run)

`diagnostics/activation_patch.py` — activation patching between clean (correct) and corrupt
(incorrect) runs for matched question pairs. Patches `resid_post` layer-by-layer; metric is
logit difference (gold − top_wrong). Output: `results/patching_heatmap.png`.

---

## Chainlink Issues

Tracked in `qwen3-finetune/.chainlink` (reinitialized 2026-07-09 with `--no-hooks` — see
project memory for why). Run `chainlink list -s open` for current status; issue numbers below
were current as of this plan's writing and may drift.
