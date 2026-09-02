# I12 (#117) MCQ track: A/B/C/R over all 113 scorable rows

Date: 2026-09-02. Model `outputs/merged-model-p6-lr5e5-r2` (v5clean), greedy,
seed 42, `data/val_v3.jsonl`, scored with the frozen
`score_mcq_trace_ll.score_with_trace`. Per-item rows in `results/slice_v1_*.jsonl`.

| Arm | Accuracy | Mean gold LL | delta_trace | delta+ | Decode calls | ms/item |
| --- | --- | --- | --- | --- | --- | --- |
| **A** bare 2-pass | **0.7522** | **-0.7898** | +0.1964 | 68.1% | 1.0 | 2680 |
| C self-refine | 0.5841 | -1.0505 | -0.0642 | 43.4% | 3.0 | 8332 |
| R retrieval | 0.7080 | -0.7819 | +0.2043 | 62.8% | 1.0 | 2736 |
| B RLM LocalREPL | 0.6549 | -0.9714 | +0.0148 | 47.8% | 7.0 | 11007 |

Paired against Arm A (n=113):

| Arm | Mean dLL | SE | t | Accuracy net | A-only right | Arm-only right |
| --- | --- | --- | --- | --- | --- | --- |
| C | -0.2606 | 0.1388 | -1.88 | **-19** | 27 | 8 |
| R | +0.0079 | 0.0692 | +0.11 | -5 | 9 | 4 |
| B | -0.1816 | 0.1111 | -1.64 | -11 | 18 | 7 |

## Verdict: no arm beats the bare 2-pass baseline

**Arm A wins outright** on accuracy, gold LL, decode cost and latency. Its
`delta_trace` of +0.1964 (68.1% positive) reproduces the sec.6.4 figure of
+0.167 that motivated this whole slice, and sec.6.4's warning turns out to be
the headline result: the champion already extracts the available gain from its
own trace, and none of the three elaborations add to it.

**Arm R (retrieval) is a clean null, not a plumbing failure.** dLL = +0.008
with t = +0.11 -- indistinguishable from zero. Accuracy is 4.4 pp lower, on 9
vs 4 discordant items (McNemar p ~ 0.27, also not significant). Retrieval
demonstrably fired: 0 items with no context, 4.31 snippets/item, 12 vsearch
fallbacks, and the model-authored queries were on-topic. The mechanism works
and the content does not help. The 12-item smoke that showed +0.16 was noise.

**Arm C (self-refine) actively harms**, and is the largest effect in the table:
-19 accuracy items, dLL -0.26, at 3x the decode cost. The critique pass damages
correct drafts more often than it repairs wrong ones (27 items lost vs 8 gained).

**Arm B never used the REPL: 0 code blocks across all 113 items.** The retune in
#127 did not change this at scale. Arm B is therefore not a test of the RLM
hypothesis at all -- it is a 7-decode-call, 11-second paraphrase of Arm A that
scores worse, and its number should not be read as "RLM does not help", only as
"this 0.6B cannot drive an RLM controller".

## Consequence for the corpus question (#132 follow-on)

The master copyleft collection should **not** be built on the strength of this
result. The corpus used here overlaps the tree the MCQs were generated from,
which biases retrieval *optimistically*; retrieval still produced no measurable
gain. A cleaner collection removes that favourable bias, so it is very unlikely
to do better on this benchmark. Build one for deployment or licensing reasons if
those matter independently -- not to rescue Arm R.

## What this does not settle

1. The free-form dev track (L7 / #130) has not run. The deploy target is
   free-form Q/A, not MCQ (sec.3.4.3), and retrieval may behave differently when
   the answer is generated rather than scored by option-set-normalized LL.
2. Arm B's result is about the champion's instruction-following, not about RLM.
   A larger orchestrator driving the same REPL is untested and out of scope for
   an on-device slice.
3. These are val_v3 numbers. The sealed test split was never touched.
