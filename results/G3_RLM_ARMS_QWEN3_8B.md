# G3 (#150): rlm_slice arms A/B/R/C against Qwen3-8B

Date: 2026-09-11. Model: `unsloth/qwen3-8b-unsloth-bnb-4bit`, in-process
(4-bit NF4 — full bf16 needs ~16GB, this card has 12GB). Dataset: the same
113 scorable MCQ rows of `data/val_v3.jsonl` used for the 0.6B champion's
`rlm_slice` runs, so numbers sit directly beside the prior local-champion
work referenced in `rlm_slice/README.md`. Greedy throughout (temperature 0,
seed 42). Scoring is the frozen `score_with_trace` objective, unmodified.

## Infrastructure note: sharing one model copy across Pass 1 and Arm B

`rlm_slice` was built around one in-process model per run — fine for arms
A/C/R, but Arm B's RLM client talks to `rlm_slice.server` over HTTP, normally
a *separate OS process* with its own model copy. At the 0.6B champion's size
two copies were cheap; at Qwen3-8B 4-bit (~7.5GB each) two copies do not fit
on a 12GB card. Fixed with `rlm_slice/run_g3_qwen3_8b.py` (new, orchestration
only — no changes to `model_host.py`, `server.py`, `controller.py`,
`eval_slice.py`, or `rlm_arm.py`): it starts `rlm_slice.server`'s Flask app in
a background thread of the *same* process instead of a separate one. The
server's routes call `model_host.get_host()` with no arguments, which returns
the process-wide singleton — so the thread and the main process's Pass-2
scoring share the exact same loaded weights at zero extra VRAM cost.
Confirmed via a 2-item smoke test before the full run (no OOM, Arm B showed
real REPL activity, not degenerate zero-block behavior).

## Results

| Arm | Accuracy | Mean option-set gold LL | Mean delta_trace | delta_trace>0 | Mean decode calls/item | Mean latency/item |
| --- | --- | --- | --- | --- | --- | --- |
| A (bare 2-pass) | 0.3717 | -2.362 | +0.389 | 50.4% | 1.0 | 7.4s |
| C (self-refine) | 0.2124 | -2.325 | +0.425 | 39.8% | 3.0 | 16.3s |
| R (controller retrieval) | 0.5398 | -1.764 | +0.987 | 63.7% | 1.0 | 7.3s |
| **B (real RLM, LocalREPL)** | **0.6991** | **-1.265** | **+1.485** | **81.4%** | **10.08** | **58.2s** |

## Arm B adaptability metrics (accept criterion)

- **REPL-block rate**: 107/113 = **94.69%** of items emitted at least one
  ` ```repl ` block (mean 1.52 blocks/item).
- **Valid-tool-call / submitted-answer rate**: 112/113 = **99.12%** of items
  reached a submitted answer through the RLM loop.
- **Decode calls/item**: mean 10.08 (vs 1 for A/R, 3 for C).
- **Latency/item**: mean 58.2s (vs 7.4s for A, ~7x A's latency — close to the
  0.6B smoke test's "closer to 7x on decode calls" finding in
  `rlm_slice/README.md`, though the multiplier here is on latency rather than
  raw decode-call count since each Qwen3-8B decode call itself takes longer).
- Degradation notes (6 `no_repl_blocks`, 1 `no_submitted_answer`, 1
  `empty_rlm_response`) account for all 8 non-clean items; the rest fully
  engaged the scaffold.

## The headline finding: the capability gap this epic exists to measure

`rlm_slice/README.md` documents, for the 0.6B champion, that "the champion
cannot initiate tool calls" — four separate probes, zero ` ```repl ` blocks
emitted across every one, model-driven tool use "not reachable by prompting
alone."

At Qwen3-8B, the same scaffold, same prompt, same fence convention: **94.7%
of items spontaneously invoke the REPL**, and full RLM (Arm B) is the
strongest arm measured, beating bare 2-pass by +32.7pp accuracy and beating
even controller-owned retrieval (Arm R) by +15.9pp. This is exactly the
scale-dependent capability gap Epic F's scoping note anticipated ("the RLM
paper's gains are at 8B+ backbones... even a prompted Qwen3-8B struggles...
without being fine-tuned to act recursively" — that hedge turns out to
undersell it: prompting alone, unfine-tuned, already gets there at 8B).

Two secondary findings worth carrying forward:

1. **Self-refine (Arm C) underperforms bare 2-pass** (0.212 vs 0.372) — more
   decode calls and latency for a *worse* answer. Iteration without a tool to
   ground it can talk the model out of a correct first pass.
2. **Retrieval alone (Arm R) captures over half of Arm B's gain** (+0.987
   mean delta_trace vs A, vs Arm B's +1.485) at a fraction of the cost (7.3s
   vs 58.2s/item, 1 decode call vs 10.08). If G4's gate weighs cost against
   benefit, R is the cheap partial win; B is the expensive full one.

## Accept criterion (#150)

Per-arm benchmark scores and RLM-adaptability metrics (REPL-block rate,
valid-tool-call rate, decode calls/item, latency) recorded above. Full
per-item detail: `results/slice_qwen3-8b_{A,C,R,B}.jsonl`. Summary:
`results/slice_qwen3-8b_summary.json`.
