# I14 (#119) decision gate: NO-GO for the Flutter port

Date: 2026-09-03. Applies `LOCAL_RLM_SLICE_PLAN.md` sec.3.5 to the M3 evidence.
Decided on the MCQ track, which is the metric that selected the champion.
Free-form is corroboration, not a driver.

## Gate criteria (sec.3.5) -- all three assessed

**1. "Beats v5clean by a meaningful margin on the primary or the trace track."
FAIL.** All 113 scorable `data/val_v3.jsonl` rows, frozen trace objective:

| Arm | Accuracy | Mean gold LL | Decode calls | ms/item |
| --- | --- | --- | --- | --- |
| **A** bare 2-pass | **0.7522** | -0.7898 | 1.0 | 2680 |
| R retrieval | 0.7080 | -0.7819 | 1.0 | 2736 |
| B RLM LocalREPL | 0.6549 | -0.9714 | 7.0 | 11007 |
| C self-refine | 0.5841 | -1.0505 | 3.0 | 8332 |

No arm beats the bare 2-pass path. Paired vs A: R +0.008 dLL (t=+0.11, a null),
B -0.182 (t=-1.64), C -0.261 (t=-1.88). Arm A's `delta_trace` = +0.1964, 68.1%
positive, reproducing the +0.167 that motivated the slice.

**2. "Free-form rubric guard stays clean (drop < 5.0 pp)." FAIL, and it is the
same ordering.** 75 paired items, `codex/gpt-5.6-luna`: A 0.5569, R 0.4804
(t=-2.57), C 0.4079 (t=-4.66), B 0.0875 (t=-14.35). Every elaboration drops far
more than 5.0 pp. This criterion is recorded as failed but is **not** what the
decision rests on.

**3. "Estimated on-device latency is acceptable." FAIL for B and C.** Arm B
spends 7 decode calls and 11.0 s per item, Arm C 3 calls and 8.3 s, against Arm
A's single call at 2.7 s -- and sec.6.1 only budgeted 2-3x. Arm R is the one
elaboration that is latency-neutral (1 call, 2.7 s), because it replaces Pass 1
rather than adding a pass. Its problem is that it buys nothing.

**Verdict: NO-GO.** No arm passes criterion 1, so no arm ports. #120 (I15) and
its subissues #135/#136 stay closed. The champion `merged-model-p6-lr5e5-r2`
ships, if it ships, as the plain 2-pass path it already is.

## Why this is a real result and not a plumbing failure

* **Arm R** demonstrably retrieved on MCQ -- 0 no-context items, 4.31 snippets
  each, on-topic model-authored queries -- and still produced a null. On
  free-form, restricted to the 26 items where retrieval fired, it was *worse*
  (-0.1885, 5 better / 15 worse). The retrieval corpus also overlaps the tree
  the MCQs were generated from, which biases it optimistically; it still did not
  help.
* **Arm B** is the one caveat: it emitted **0 REPL blocks across all 113 items**.
  Its number measures this 0.6B's inability to drive an RLM controller, not the
  RLM hypothesis. Do not cite it as evidence against RLM generally.
* **Arm C** is an unambiguous harm: the critique pass broke 27 correct MCQ
  drafts while repairing 8.

## The wider conclusion

The inference-time lever is exhausted. Three different controllers wrapped
around the frozen champion all made it worse, and `delta_trace` says the model
already extracts what its own reasoning affords. Closing the remaining distance
to larger models has to come from the **training side** -- data and objective --
not from the harness around inference.

## Measurement gap worth fixing before targeting "8B performance at 0.6B"

`benchmark-workspace/EVALUATION_RESULTS.md` (Natural Fertilizer MCQ Benchmark
v1, 98 questions, 2026-02-25) records the ladder this project is chasing:

| Model | Accuracy |
| --- | --- |
| qwen3:8b (Ollama) | 90.8% |
| claude-opus-4-6 | 89.8% |
| claude-haiku-4-5 | 73.5% |
| cactus-pixel7 qwen3-0.6 (base, on-device) | 28.6% |

**No fine-tuned checkpoint has ever been scored on this set.** The champion's
numbers live on `data/val_v3.jsonl` (113 rows) under option-set-normalized
gold-letter log-likelihood; the table above is 98 different questions scored by
generative answer-parsing with an "ambiguous" bucket. The two are not
comparable, so Arm A's 0.7522 **cannot** be read against qwen3:8b's 90.8%.

That means the size-gap this program is trying to close is currently
unquantified. Scoring v5clean on the 98-question benchmark with the same
generative protocol is a small, self-contained job and would give the program a
real target line. Recommended as the next concrete step.
