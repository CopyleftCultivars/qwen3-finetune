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

## Part 2 - Minimal RLM slice (Flutter application)

Design: depth-1, no Python, no graph change. A Dart controller wraps the Cactus Flutter plugin and
runs the same 2-pass shape as the eval. Training and deployment then match.

1. Recon (required first): confirm the latest Cactus Flutter plugin API. Record model load,
   `generate(prompt)`, streaming, KV-cache reuse, and any embedding API.
   Write `docs/cactus_flutter_api.md`.
2. Add `RlmController` (Dart). Pass 1: generate reasoning trace (think-on).
   Pass 2: generate the final user answer (think-off), conditioned on the trace.
3. Optional on-device memory (RAG, "Metis-lite"): a small local embedding store of agronomy notes.
   Retrieve top-k and inject into the Pass-1 context. Use the Cactus embedding API if present;
   otherwise use a small separate embedder.
4. Budget harness: measure latency (ms) and peak memory (MB) on a Pixel 7. Set a hard latency budget.
   Add a single-pass fallback path when the 2-pass path exceeds the budget.
5. UI: hide the reasoning trace by default. Show the final answer only.
   Add a debug toggle to show the trace.
6. Parity test: run a golden item set through both the device 2-pass path and the server Plan A eval.
   Confirm the outputs match.

Note on the RLM speed claim: RLM lowers cost only when the input context is long.
The Q/A prompts here are short, so the 2-pass path adds latency.
Justify Part 2 by answer quality, not speed. Budget for about 2x decode time.

---

## Part 3 - Cactus fork spike (Metis + true multi-stream) - gated, recon-first

1. Recon the latest Cactus engine: architecture support matrix, custom-op / kernel extension points,
   GGUF format and converter, backends (CPU / GPU / NPU), and Qwen3.5-hybrid support.
2. Write a capability report `docs/cactus_fork_feasibility.md`. Answer two questions:
   (a) Can Cactus hold a persistent memory-state buffer plus a Gated Delta Network update kernel
   (Metis)? (b) Can Cactus run a cross-stream mask plus multi-token-per-step decode
   (true multi-stream)?
3. Go / no-go gate: start a fork only if the effort and fork-maintenance cost are acceptable AND the
   Part 1 / Part 2 results justify the added capability. Otherwise keep the external surrogate
   (Plan B) and the RLM slice (Part 2).

## Difficulty summary (Cactus)

| Item | Engine change | Effort | Deploy risk |
| --- | --- | --- | --- |
| RLM minimal slice (Flutter host controller) | None | LOW | LOW |
| Multi-stream surrogate (reasoning SFT) | None (data) | LOW | LOW |
| External RAG memory ("Metis-lite") | None (app) | LOW-MEDIUM | LOW |
| True model-level multi-stream | Engine fork | HIGH | HIGH |
| Metis native memory | Engine fork + custom kernels + base swap | VERY HIGH | VERY HIGH |
