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

**Phase 1 — Rebuild a clean baseline (v2-clean): conversations only.** Run
`prepare_dataset.py` (or a v3-style equivalent) with **only** the `finetune-gemma4` conversation
source — no MCQ data of any kind. Train and merge. This is the first trustworthy checkpoint:
its benchmark accuracy is a fair (if likely low) measurement, since nothing it was trained on
touches `benchmark_combined.jsonl`.

**Phase 2 — Evaluate v2-clean.** Run the full Stage A/B diagnostic suite against v2-clean on
`benchmark_combined.jsonl`. These numbers are now trustworthy — first real baseline for this
project.

**Phase 3 — Build a genuinely clean MCQ training source.** Two options, not mutually exclusive:
  - (a) Identify which `qtmd_workspace/markdown_corpus` collections/documents
    `benchmark-workspace/generate_mcqs_rag.py` actually drew from to build
    `benchmark_combined.jsonl`, and exclude those collections from any MCQ generation source.
  - (b) Generate MCQs from any source (QMD-derived or otherwise), then mechanically dedupe by
    question-text (and near-duplicate/embedding similarity, not just exact match) against
    `benchmark_combined.jsonl` before using them for training. Treat this as mandatory going
    forward, not optional — the exact-match check alone would have caught this contamination
    on day one.

**Phase 4 — Rebuild v3-clean.** Gemma4 conversations + the clean MCQ source from Phase 3. Still
**no QMD stream** — matches the original v3 intent, deferred until a solid non-QMD finetune
exists.

**Phase 5 — Evaluate v3-clean, compare fairly to v2-clean.** Same Stage A/B suite. This is the
first trustworthy v2→v3-style comparison.

**Phase 6 (future, gated on Phase 5 looking solid) — Add QMD-generated MCQs.** Only after
Phase 4/5 produces a checkpoint judged solid. Any QMD-generated MCQ must pass the dedup check
from Phase 3(b) before inclusion.

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
