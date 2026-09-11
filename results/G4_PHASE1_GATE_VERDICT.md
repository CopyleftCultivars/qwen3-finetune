# G4 (#151): Phase-1 gate verdict — GO

Date: 2026-09-11. Inputs: G2 (`results/G2_QWEN3_8B_BASELINE.md`), G3
(`results/G3_RLM_ARMS_QWEN3_8B.md`). Same model (`qwen3:8b` / the 4-bit
`unsloth/qwen3-8b-unsloth-bnb-4bit` weights), same 113-row scorable subset of
`data/val_v3.jsonl` in both.

## The two criteria (from #151)

- **Kill**: no arm reaches the G2 baseline AND 8B does not drive the REPL.
- **Go**: 8B drives the REPL AND at least one arm shows a non-negative trend.

## The numbers

G2 baseline (zero-shot, no Pass-1 trace, `score_mcq_ll.py` argmax over the
4 letter-token logprobs, same 113 rows): **0.4867**.

G3 arms (same underlying letter-logprob argmax metric, `score_with_trace`,
conditioned on each arm's Pass-1 trace):

| Arm | Accuracy | vs G2 baseline (0.4867) |
| --- | --- | --- |
| A (bare 2-pass) | 0.3717 | -11.5pp |
| C (self-refine) | 0.2124 | -27.4pp |
| R (controller retrieval) | 0.5398 | **+5.3pp** |
| B (real RLM, LocalREPL) | 0.6991 | **+21.2pp** |

REPL-driving: 107/113 items (94.69%) spontaneously emitted at least one
` ```repl ` block under Arm B, with 99.12% reaching a submitted answer
through the RLM loop — a direct reversal of the 0.6B champion's documented
0/N REPL-block rate under the same scaffold (`rlm_slice/README.md`).

## Verdict: GO

Both go-conditions hold, and by a wide margin rather than marginally:

1. **8B drives the REPL.** 94.69% block-emission rate, not a borderline
   result.
2. **At least one arm shows a non-negative trend.** Two of four exceed the
   G2 baseline outright (R: +5.3pp, B: +21.2pp) — stronger than the bar
   requires ("non-negative"), not just clearing it.

The kill condition is false on both conjuncts simultaneously (some arms do
reach baseline; 8B does drive the REPL), so kill is not a close call either
way.

Two qualifications carried into G5+, not blockers to the GO call itself:

- Arms A and C actively regress below the zero-shot baseline. A naive
  Pass-1 trace without a tool to ground it, or self-refine iteration without
  one, can talk the model out of a correct zero-shot answer. Whatever G5's
  trajectory-SFT data targets should be shaped around what Arm B (or R) is
  doing, not a generic "add a reasoning pass" recipe.
- Arm R (retrieval alone) captures roughly a quarter of Arm B's total gain
  over baseline (+5.3pp vs +21.2pp) at ~8x lower latency and 10x fewer
  decode calls. If G5-G7's fine-tuning investment is meant to buy
  Arm-B-level gains specifically, that delta is the number to beat; if
  cost matters as much as accuracy, R is worth keeping as a cheaper
  fallback design point.

## Accept criterion

Verdict recorded (GO) with the numbers behind it, above. Unblocks G5 (#152).
