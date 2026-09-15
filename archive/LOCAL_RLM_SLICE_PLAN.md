# Plan: Local minimal RLM slice (Python-first, benchmark-gated)

Date: 2026-09-01
Companion plans: `TRACE_EVAL_REASONING_SFT_RLM_PLAN.md`, `TRACE_EVAL_REASONING_SFT_RLM_CHAINLINK_EPIC.md`.

## 0. Answers to the open questions

1. RLM does NOT require retraining or finetuning the Qwen3 model. Correct.
   RLM is an inference harness. It wraps the existing model with a controller, an execution
   step, and optional sub-calls. The champion `outputs/merged-model-p6-lr5e5-r2` (v5clean)
   stays unchanged. Because the weights and the graph do not change, the later Cactus
   conversion is unaffected.
2. The RLM repository also ships an optional RL training environment. We do NOT use it here.
   The minimal slice is inference-only.
3. Tool-calling RAG can be part of the minimal slice, but it is an OPTIONAL arm.
   RAG helps only failures caused by missing knowledge (the J-space "absent" class).
   RAG does not help failures caused by a weak answer surface or weak reasoning.
   RAG also needs a retrieval corpus. If no agronomy corpus exists yet, defer the RAG arm.

## 1. Strategy

1. Build the RLM slice as a local Python layer on this machine first.
2. Measure it against the benchmark with the existing scorer.
3. Port it to the Flutter application only if a local arm beats the v5clean baseline and keeps
   the guardrails clean.

## 2. Scope and constraints

1. No finetune. No weight change. No frozen-file edits (`autoresearch/frozen_manifest.json`).
2. Never read `eval_freeform/test.jsonl`. Keep the sealed-test wall.
3. Reuse `score_with_trace` from `autoresearch/score_mcq_trace_ll.py` for Pass-2 scoring, so
   results are directly comparable to `results/mcq_trace_ll_v5_clean.json`.
4. Put new code under a new directory `rlm_slice/`. Do not modify the frozen scorers.
5. Decode is deterministic: temperature = 0 (greedy), fixed seed.

## 3. Components

### 3.1 Model server
1. Serve the merged champion `outputs/merged-model-p6-lr5e5-r2` through an OpenAI-compatible
   endpoint. Use vLLM (recommended by the RLM repository for local models) or a small local
   wrapper.
2. This lets the RLM library talk to the local model with `backend="openai"`.

### 3.2 Controller arms (compare these)
1. Arm A - bare 2-pass (baseline slice): Pass 1 generates a reasoning trace (think-on);
   Pass 2 scores the gold letter conditioned on that trace. This is the current Plan A path.
2. Arm B - RLM LocalREPL, depth 1: use the `rlm` package `LocalREPL` with `max_depth = 1`
   (no deep recursion). Give the model helper functions and optional `custom_tools`.
   This is the closest faithful "minimal RLM" arm.
3. Arm C - self-refine: draft answer, then critique, then revise (2 to 3 passes, no tools).

### 3.3 RAG tool (optional arm, gated on a corpus)
1. Step 1: identify or build a small agronomy corpus (source to be confirmed).
2. Step 2: build a local embedding index over that corpus.
3. Step 3: expose `retrieve(query, k)` as an RLM `custom_tool`. It returns the top-k snippets.
4. Step 4: the controller injects the snippets into the Pass-1 context.
5. Measure the RAG arm separately from the no-RAG arms.

Note on tool calling: a 0.6B model is weak at reliable tool-call formatting. Prefer a
fixed-pipeline retrieve step first (controller calls `retrieve`, not the model). Only test
model-driven tool calls after the fixed pipeline shows value.

### 3.4 Evaluation harness
1. Run every arm over the 113 scorable MCQ rows in `data/val_v3.jsonl`.
2. Score with `score_with_trace` (option-set-normalized gold-letter log-likelihood) so numbers
   compare directly to the v5clean trace-LL baseline.
3. Also run the free-form dev set (`eval_freeform/dev.jsonl`) with the rubric, because the deploy
   target is free-form question/answer, not MCQ.
4. Record per-item results, latency (ms), and total decode calls per item.

### 3.5 Decision gate (port-to-Flutter rule)
Port an arm to Flutter only when ALL of these are true:
1. The arm beats v5clean by a meaningful margin on the primary or the trace track.
2. The free-form rubric guard stays clean (drop < 5.0 pp).
3. The estimated on-device latency is acceptable. Estimate = decode calls per item times the
   per-token decode time on a Pixel 7.

## 4. Procedure

1. Stand up the local model server (3.1).
2. Implement `rlm_slice/controller.py` with Arm A, Arm B, and Arm C (3.2).
3. Implement `rlm_slice/eval_slice.py` that runs an arm over the MCQ set and the free-form dev
   set and reuses `score_with_trace` (3.4).
4. Run Arm A, Arm B, and Arm C. Compare against `results/mcq_trace_ll_v5_clean.json` and the
   `codexv5` rubric baseline.
5. If a knowledge corpus exists, add the RAG tool (3.3) and run the RAG arm.
6. Apply the decision gate (3.5). Record the verdict in `autoresearch/journal.jsonl` and
   `autoresearch/levers.md`.
7. If an arm passes, start the Flutter port (separate plan / Epic M3). If no arm passes, stop and
   record the negative result.

## 5. Chainlink epic adjustment (M3 re-scoped local-first)

1. I11 (was "Cactus Flutter API recon") becomes "local model server + `rlm_slice` controller".
2. I12 becomes "local eval harness + Arm A/B/C comparison".
3. I13 becomes "optional RAG arm (gated on a corpus)".
4. I14 becomes "decision gate + journal record".
5. I15 (Flutter port + parity test) runs only if step 4 passes. Cactus Flutter recon moves here.

## 6. Risks

1. Extra passes add latency. RLM lowers cost only for long context. These prompts are short, so
   expect about 2x to 3x the decode time. Justify by quality, not speed.
2. RAG needs a corpus. Without one, the RAG arm is not testable.
3. A 0.6B model may not reliably follow model-driven tool calls. Prefer a fixed retrieve step.
4. The gain may be small. The champion already benefits from a trace (`delta_trace = +0.167`),
   so Arm A may already capture most of the available gain.

## 7. Chainlink decomposition (created 2026-09-02)

M3 (milestone #3) re-scoped local-first in place, per section 5. Parents #116-#120 keep their
I11-I15 identity with new titles/acceptance; 13 executable subissues (label `rlm-slice`) sit
under them.

| Issue | Item | Parent | Blocked by |
| --- | --- | --- | --- |
| #116 | I11 local model server + controller arms | #105 | - |
| #124 | L1 serve champion via OpenAI-compatible endpoint | #116 | - |
| #125 | L2 `rlm_slice/` scaffold, client, determinism + sealed-test guards | #116 | #124 |
| #126 | L3 Arm A bare 2-pass | #116 | #125 |
| #127 | L4 Arm B RLM LocalREPL depth 1 | #116 | #125 |
| #128 | L5 Arm C self-refine | #116 | #125 |
| #117 | I12 local eval harness + A/B/C comparison | #105 | #116 |
| #129 | L6 `eval_slice.py` MCQ track (113 rows, `score_with_trace`) | #117 | #126 |
| #130 | L7 free-form dev rubric track | #117 | #129 |
| #131 | L8 Arm A/B/C comparison report | #117 | #129, #130 |
| #118 | I13 optional RAG arm | #105 | #117 |
| #132 | L9 corpus decision (build or defer) | #118 | - |
| #133 | L10 embedding index + fixed-pipeline `retrieve(query, k)` | #118 | #132 |
| #134 | L11 run and score the RAG arm separately | #118 | #133, #131 |
| #119 | I14 decision gate + journal record | #105 | #117, #118 |
| #120 | I15 Flutter port + parity test (gated) | #105 | #119 |
| #135 | L12 Cactus Flutter API recon | #120 | #119 |
| #136 | L13 port winning arm + parity test | #120 | #135 |

Entry points now ready: #124 (stand up the server) and #132 (corpus decision, which can run in
parallel and may close #118 as "defer").

Note: milestone #3 still carries its old name "M3: Minimal RLM slice (Flutter)". The scope is
now local-first; Flutter work is confined to #120/#135/#136 behind the #119 gate.
