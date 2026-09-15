# From-scratch finetuning plan

## Decision point

`outputs/merged-model-p6-lr5e5-r2` remains the frozen v5clean champion. A
one-epoch continuation on the strict English-only base regressed the English
anchor from option-set gold LL `-0.7717` to `-1.0134` and Spanish transfer dev
from `-1.0265` to `-1.2324`. The Spanish overlays add composition effects, but
continued training from the merged champion is already unsafe at the full
`5e-5` recipe.

The lower-pressure mixed Spanish diagnostic at `1e-5` improved the result to
Spanish `-1.1045` and English `-0.8395`, but still failed both bilingual gates.
It is a useful direction, not a promotion candidate.

## What is already evaluated

The repository does not have a large set of forgotten ready-to-train recipes.
The following work has already been evaluated against v5clean and rejected or
settled:

- Tier 1 recipe dimensions: epochs, LoRA rank/alpha, schedule, warmup, packing,
  sequence length, target modules, early stopping, gradient accumulation, batch
  size, and weight decay;
- current-source Tier 2 data/format dimensions: permutation, replay ratio,
  answer surface, cloze, curriculum, refusal/filtering, and QMD item count;
- Tier 3 P11/P12 rationale replay and P13 novel step-back rationale authoring;
- P14/P15/P16 targeted permutation, replay-ratio, and synthetic-item-count
  follow-ups;
- Spanish QMD four-epoch baseline plus MCQ-only, free-form-only, mixed-dose,
  English-only continuation, and lower-LR debug controls.

The decision history is in `autoresearch/levers.md` and
`results/autoresearch_report_summary.md`. Those rejected streams should not be
silently reintroduced into a new composite dataset.

## Unevaluated work that must be built before a broad from-scratch candidate

These are new data pipelines, not existing files that can simply be added:

1. Preserved multi-turn and step-back SFT. The current loader flattens ShareGPT
   conversations into independent instruction/output pairs. Build a provenance-
   tracked preserved-turn representation and a separately tagged step-back
   subset.
2. CROP-style auxiliary tasks from natural-farming sources: summarization,
   NER/entity extraction, structured extraction, and domain open-book QA. Do
   not ingest generic CROP directly. Every record needs source-document IDs,
   task labels, language labels, and contamination checks.
3. Open-book/RAG-grounded training. Use the existing natural-farming corpus and
   RAG artifacts only after adding document IDs/citation targets. Evaluate this
   stream separately from closed-book MCQ and free-form performance.
4. Reflection training. Author a controlled answer-audit or counterfactual-
   reflection target set, with a Spanish-specific or English-specific rubric
   as appropriate. The rejected P13 rationale source is not a substitute for
   this stream.
5. General-instruction replay. This remains blocked until a clean, provenance-
   tracked general-instruction source is selected; generic CROP is not an
   acceptable shortcut.
6. Spanish QMD data. Keep the translated MCQ/free-form stream as a separately
   tagged ablation. Its contamination audits pass, but its bilingual gate is
   not yet passed, so it must not be allowed to hide regressions in an all-in
   composite score.

Tool-calling (#10), true model-level multi-stream architecture (#67), and SAE
interpretability remain explicitly deferred. They are not data streams that
can responsibly be folded into this first from-scratch recipe.

## Build and evaluation order

1. Build and audit the pending streams on CPU: schema, source provenance,
   document/source-ID split disjointness, benchmark contamination, language,
   task balance, token lengths, and exact train.py rendering.
2. Assemble a composite manifest from the original `unsloth/Qwen3-0.6B` base
   recipe. Preserve per-stream tags and checksums so each stream can be removed
   or ablated without rebuilding the others.
3. Run a from-scratch reproduction using the clean English v5clean data mix;
   this calibrates the base-model lineage and catches any pipeline drift.
4. Train the composite stream set serially on the one GPU. Score every stage on
   English v5clean, Spanish transfer dev, and free-form dev. Keep all sealed
   tests untouched until a candidate clears the development gates.
5. Promote only a candidate that improves Spanish while producing no English
   regression and no free-form, forgetting, bias, compute, or contamination
   failure. Otherwise keep v5clean and diagnose the failing stream.

The next implementation task is therefore the pending-stream data pipeline and
composite manifest—not another blind full training run.
