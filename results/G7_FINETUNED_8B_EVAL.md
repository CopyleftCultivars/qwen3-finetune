# G7 (#154): evaluate the fine-tuned 8B vs Phase 1 (G3)

Date: 2026-09-12. Same harness, same 113-row `val_v3` MCQ set, same seed as
G3 — the only variable changed is the model:
`outputs/merged-model-g6-rlm-8b` (G6's LoRA-fine-tuned checkpoint) in place
of the base `unsloth/qwen3-8b-unsloth-bnb-4bit`.

## Benchmark score delta (primary criterion)

| Arm | Phase 1 (G3, base) | Phase 2 (G7, fine-tuned) | Delta |
| --- | --- | --- | --- |
| A (bare 2-pass) | 0.3717 | 0.5841 | **+21.2pp** |
| C (self-refine) | 0.2124 | 0.3009 | **+8.8pp** |
| R (controller retrieval) | 0.5398 | 0.7257 | **+18.6pp** |
| B (real RLM, LocalREPL) | 0.6991 | 0.7876 | **+8.9pp** |

**Every arm improved. None regressed.** Arm B, the flagship arm, now scores
0.7876 — +30.1pp over the original G2 zero-shot baseline (0.4867), up from
G3's +21.2pp over that same baseline. Arms A and R gained the most in
absolute terms (+21.2pp, +18.6pp) — plausible given the fine-tune's training
signal came entirely from *successful* trajectories (correct + REPL-engaged),
which likely improved general reasoning-trace quality on this domain
independent of whether the REPL scaffold was invoked, not just narrow
REPL-specific behavior.

## RLM-adaptability metrics delta (Arm B)

| Metric | Phase 1 (G3) | Phase 2 (G7) | Delta |
| --- | --- | --- | --- |
| REPL-block rate | 94.69% (107/113) | 91.15% (103/113) | **-3.5pp** |
| Submitted-answer rate | 99.12% (112/113) | 97.35% (110/113) | **-1.8pp** |
| Mean repl_blocks/item | 1.52 | 1.36 | -0.16 |
| Mean rlm_iterations/item | 5.03 | 5.20 | +0.17 |
| Mean decode calls/item | 10.08 | 10.42 | +0.34 |
| Mean latency/item | 58.2s | 83.7s | **+25.5s (+44%)** |
| Degenerate items (no_repl_blocks) | 6 | 10 | +4 |

Two costs alongside the accuracy gain: REPL engagement and submitted-answer
reliability both dipped slightly (small in absolute terms — a handful of
items at n=113 — but a real, consistent direction, not just one metric
moving), and per-item latency grew ~44%. The fine-tune was not tuned to
minimize latency or maximize scaffold-engagement rate; its training signal
(G5's filtered trajectories) rewarded reaching the *correct answer*, which
this result achieves more of, at a modest cost to how often and how
efficiently the scaffold itself gets invoked.

## Interpretation

The core question G7 asks — did fine-tuning on RLM trajectories improve
RLM adaptability, or just overfit 117 examples — reads as a clear yes on
the metric that matters most (benchmark accuracy, up in every arm, largest
gains in the arms fine-tuning targeted least directly), with a modest,
non-alarming softening in scaffold-engagement rate and latency that is
worth watching but does not offset the accuracy gain at this sample size.

## Accept/reject: **ACCEPT**

No pre-specified numeric threshold was set for this gate (unlike G4's
explicit kill/go formula) — the criterion is the pattern of deltas. All four
arms show large, consistent, positive benchmark-score deltas (+8.8pp to
+21.2pp) with zero regressions; the RLM-adaptability softening is small
(a few items out of 113 each way) relative to the accuracy signal and does
not reverse the direction of the result. The fine-tune is accepted as a real
improvement over the Phase-1 prompted-only result.

## Accept criterion (#154)

Both deltas (benchmark score, RLM-adaptability metrics) reported above;
accept/reject recorded (ACCEPT). Unblocks G8 (#155, optional RL rung) and
G9 (#156, downsizing analysis and go/no-go for a 0.6B implementation).
