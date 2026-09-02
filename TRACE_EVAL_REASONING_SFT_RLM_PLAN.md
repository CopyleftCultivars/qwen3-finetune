# Plan: Trace-conditioned eval + reasoning SFT + minimal on-device RLM slice

Date: 2026-09-01

## Purpose

The deployed role of the model is free-form question/answer, not multiple-choice.
The MCQ benchmark is a CROP-style proxy only
(https://proceedings.neurips.cc/paper_files/paper/2024/file/5e5783c673cf05cfd4b3ebf46e96abfc-Paper-Datasets_and_Benchmarks_Track.pdf).

The current primary objective scores `P(gold letter | forced-empty-<think>)`.
That objective removes reasoning from the scored sequence.
J-space needs a reasoning trace to read verbalizable content, so it had no signal to work with.
This plan restores a reasoning surface without breaking the CROP scored-artifact (a single letter).

## Assumptions and constraints

1. Base model stays `unsloth/Qwen3-0.6B` (LoRA). Part 1 and Part 2 make no forward-graph change.
2. Do not edit files listed in `autoresearch/frozen_manifest.json`. Add new modules instead.
3. Never read `eval_freeform/test.jsonl` during any loop step. Keep the sealed-test wall.
4. Pass 2 metric = option-set-normalized gold-letter log-likelihood (CROP-continuous). Confirmed choice.
5. Trace-conditioned log-likelihood is a NEW objective. It is not comparable to the old empty-think
   log-likelihood. Build a new baseline.

---

## Part 1 - Training and evaluation (Cactus-safe)

### Plan A - 2-pass reason->extract eval

1. Add `autoresearch/score_mcq_trace_ll.py` (new file, not the frozen `score_mcq_ll.py`).
2. Pass 1 (reason): input = question + options, think-on. Decode greedy (temperature = 0),
   fixed seed, `max_trace_tokens = 256`. Output = reasoning trace.
3. Pass 2 (extract): input = question + options + Pass-1 trace, think-off.
   Reuse `score_symbol` and `get_letter_token_ids` from
   `../benchmark-eval-workspace/diagnostics`.
   Output = option-set-normalized gold-letter log-likelihood.
4. Leakage guards: strip the gold letter from the Pass-1 prompt.
   Refuse sealed paths by denylist, the same pattern as `jspace_annotator.py`.
5. Ablation guard: also score with an empty trace.
   Record `delta_trace = LL(with_trace) - LL(empty_trace)` per item.
6. Baseline artifacts: write `results/mcq_trace_ll_<champion>.json` and a per-item JSONL
   with the trace and both scores.
7. Objective policy: add `autoresearch/objective_weights_v3.json` for the trace-conditioned primary.
   Keep it as a PARALLEL track. Do not replace the current primary until step 8 passes.
8. Validation gate for the new track: accept the new objective only when
   (a) `delta_trace > 0` on a clear majority of items, and (b) the rubric guard stays clean.
   Document the result in `autoresearch/program.md`.

### Plan B - Synthetic reasoning-trace SFT

1. Add `data/make_synthetic_traces.py`. For each TRAIN row only, call the existing teacher backend
   (OpenRouter) to produce a reasoning trace that ends at the gold letter.
2. Contamination wall: hash every train row and reject any overlap with `data/val_v3.jsonl`,
   `eval_freeform/dev.jsonl`, and `eval_freeform/test.jsonl`.
   Reuse the contamination logic from `prepare_dataset_v3.py`.
3. Causal verification and quality filter: keep a row only if the teacher trace, when scored
   think-off, selects the gold letter. Drop all other rows. Report the drop rate.
4. Add `prepare_dataset_v6.py` (or a `_v6` suffix path). Emit chat records with the reasoning in the
   think channel and the letter in the answer channel, kept separable. This matches the Plan A eval.
5. Add `autoresearch/configs/<phase>-trace-sft.yaml`. Train with the existing
   `autoresearch/finetune.py`. Keep LoRA and the same base model.
6. Evaluate the candidate with Plan A. Gate against the new trace-LL baseline.
   Keep the rubric Goodhart guard.

---

## Part 2 - Qwen3 RLM model proof (reference runtime)

Design: depth-1, no graph or engine change. Prove the two-pass behavior directly on Qwen3 against
the frozen v5clean champion before doing any Flutter, device, or Cactus integration.

1. Document the current Qwen3 reference-runtime interfaces for model load, generation, two-pass
   prompting, KV-cache reuse (if available), and embeddings.
2. Implement the RLM controller shape in the reference runtime. Pass 1 generates a reasoning trace
   (think-on); Pass 2 generates the final answer (think-off), conditioned on that trace.
3. Test optional external RAG memory ("Metis-lite"): retrieve top-k agronomy notes and inject them
   into Pass 1, with a toggle that restores the exact no-memory baseline.
4. Measure quality, latency, and peak memory for single-pass and two-pass Qwen3 paths. Define and
   test a resource fallback, but defer Pixel-specific optimization.
5. Gate the RLM candidate against frozen v5clean using trace-conditioned MCQ, standard primary,
   forgetting, and rubric guards. Continue to integration work only after a material, guard-clean win.

Note on the RLM speed claim: RLM lowers cost only when the input context is long.
The Q/A prompts here are short, so the 2-pass path adds latency.
Justify Part 2 by answer quality, not speed. Budget for about 2x decode time.

---

## Part 3 - Qwen3 Metis-style model proof, then integration gate

1. Test an external/persistent-memory surrogate directly against Qwen3 in the reference runtime.
   Keep the mechanism reproducible and independently disableable.
2. Compare it with frozen v5clean under the same quality and resource gates used in Part 2.
3. Do not modify Cactus or begin native Metis kernels, true multi-stream decoding, Flutter, or device
   work unless the model-level candidate materially and cleanly beats v5clean.
4. Only after such a win, open a separate integration-feasibility phase covering Cactus architecture,
   custom kernels, model conversion, supported backends, device budgets, and maintenance cost.

## Difficulty summary

| Item | Engine change | Effort | Deploy risk |
| --- | --- | --- | --- |
| RLM model proof (Qwen3 reference runtime) | None | LOW | LOW |
| Multi-stream surrogate (reasoning SFT) | None (data) | LOW | LOW |
| External RAG / persistent-memory surrogate | None (reference runtime) | LOW-MEDIUM | LOW |
| Cactus/device integration after a model win | Integration work | MEDIUM-HIGH | MEDIUM |
| True model-level multi-stream | Deferred engine fork | HIGH | HIGH |
| Metis native memory | Deferred engine fork + custom kernels + base swap | VERY HIGH | VERY HIGH |
