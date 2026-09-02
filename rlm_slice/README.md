# rlm_slice - local minimal RLM slice

Implements `LOCAL_RLM_SLICE_PLAN.md` milestone M3 (chainlink #116, subissues
#124-#128). Inference only: no finetune, no weight change, and nothing here
modifies `autoresearch/frozen_manifest.json` or the frozen scorers.

## Environment

Everything runs in the `qwen3-train` conda env (Python 3.10.20, torch
2.10.0+cu128, CUDA available). Run from the repo root with `PYTHONPATH=.`:

```
conda run -n qwen3-train python -m rlm_slice.server --port 8000
```

## Layout

| File | Role |
| --- | --- |
| `guards.py` | Sealed-test denylist, seed pinning, greedy-decode assertion (sec.2.2/2.5) |
| `model_host.py` | Loads the champion once; Pass-1 generation + Pass-2 scoring + telemetry |
| `server.py` | OpenAI-compatible HTTP surface over that same host (sec.3.1) |
| `rlm_arm.py` | Wiring for the real `rlm` package used by Arm B (sec.3.2.2) |
| `controller.py` | Arms A, B, C and the shared Pass-2 scoring tail (sec.3.2) |

## Three decisions worth knowing

**1. A local wrapper, not vLLM.** sec.3.1 allowed either. `score_with_trace` in
`autoresearch/score_mcq_trace_ll.py` takes a live HF model object and reads
per-letter logits, which cannot be expressed over an OpenAI HTTP surface -- so
Pass-2 needs an in-process model regardless of what serves Pass 1. vLLM would
have been a *second* copy of the weights on a 12 GB card, and its torch pin
would put the env that produced the champion at risk. One in-process copy also
makes Pass 1 and Pass 2 provably the same weights, which is what sec.2.3's
comparability requirement actually needs. `server.py` puts the OpenAI surface
over that same singleton so Arm B can still use `backend="openai"`.

**2. The `rlm` package is vendored by path, not pip.** It is the Recursive
Language Models repo (Alex Zhang, MIT) at
`../persistent-memory-workspace/rlm`, distributed as **`rlms`** -- which is why
it does not resolve on PyPI as `rlm`. Its metadata pins `requires-python >=
3.11` while `qwen3-train` is 3.10.20, but the code imports and runs clean on
3.10, so `rlm_arm.ensure_rlm_importable()` adds it to `sys.path` instead. That
avoids both rebuilding the champion's env and editing a pin in a repo we do not
own. If Arm B ever behaves oddly, this version skew is the first thing to check.

**3. Arm B's telemetry is read back over HTTP.** Arm B is the only arm that
generates out-of-process. Its decode calls are counted by the server and pulled
via `/stats`, because otherwise Arm B would report zero decode calls and the
sec.3.5.3 on-device latency estimate would be wrong in the direction that makes
Arm B look free.

## Arms (sec.3.2)

All three end at the *same* `score_with_trace` call, which is what makes them
comparable to each other and to `results/mcq_trace_ll_v5_clean.json`.

- **A** - bare 2-pass. Delegates Pass 1 to the trace scorer's own
  `generate_trace`, so it *is* the existing baseline path, not a lookalike.
- **B** - `rlm` LocalREPL, `max_depth=1`, greedy, with `count_words` /
  `compare_numbers` as `custom_tools`. `retrieve` is deliberately absent: that
  is I13 (#118) and must be measured as its own arm (sec.3.3.5).
- **C** - self-refine: draft, critique, revise. Three passes, no tools.

Arms B and C degrade to the bare 2-pass trace when the 0.6B model fails to
produce the structure they ask for (sec.6.3). Every degradation is recorded in
the per-item `notes` field rather than hidden, because "Arm B silently became
Arm A on most items" is the most likely way this comparison could mislead.

## Measured so far (2 items, smoke only -- not a result)

| Arm | Decode calls / item | Latency / item |
| --- | --- | --- |
| A | 1 | ~3.3 s |
| B | 7 | ~8.9 s |
| C | 3 | ~9.8 s |

sec.6.1 predicted 2-3x the decode time; Arm B is closer to 7x on decode calls.
That is a cost the sec.3.5 gate has to be worth, and it is the number to carry
into the sec.3.5.3 Pixel-7 estimate. Real per-arm quality numbers come from I12
(#117), over all 113 scorable rows.

## Arm B finding: the champion cannot initiate tool calls (2026-09-02)

Investigated under #127 after the first Arm B run returned clean prose while
never opening the REPL. Four probes against the champion, greedy, via the local
server:

| Probe | Emits ```repl fence? |
| --- | --- |
| Stock `RLM_SYSTEM_PROMPT`, think-on | no (0 blocks over 7 iterations) |
| MCQ-shaped system prompt, `orchestrator=False`, think-on | no (0 blocks, 4/4 items) |
| MCQ-shaped system prompt, think-off | no |
| "Copy this exactly: ```repl print(context)```" | **yes**, exactly |

So the capability is present at the token level and the *instruction-following
to invoke it* is absent. v5clean is a 0.6B SFT'd on question-in / agronomy-prose
-and-MCQ-answer-out; anything shaped like a question triggers that reflex and
the system prompt loses. Retuning the prompt does not fix this -- three
framings, zero blocks -- so the `orchestrator=False` + `MCQ_SYSTEM_PROMPT`
retune is kept (it is strictly better framing) but it is not sufficient.

The useful half. Prefilling the assistant turn changes the picture:

```
prefill "```repl\n"                  -> model restates the question (useless)
prefill "```repl\nprint(retrieve("   -> '"permaculture guilds role in ecosystem
                                        design"))' -- a well-formed, on-topic
                                        retrieval query, correctly closed
```

The model cannot decide *to* call a tool, but it authors a good *argument* when
the call scaffolding is placed for it. That is precisely the split
LOCAL_RLM_SLICE_PLAN.md sec.3.3 predicted: "Prefer a fixed-pipeline retrieve
step first (controller calls `retrieve`, not the model). Only test model-driven
tool calls after the fixed pipeline shows value."

Consequence for I13 (#118): the QMD collections are still reachable, and the
model can still supply the query -- the controller just has to own the call
site. Model-driven tool calling with this champion is not reachable by
prompting alone.

## Arm R: controller-owned retrieval (L10 / #133)

`retrieval.py` implements `retrieve(query, k)` per CORPUS_DECISION.md: `qmd
search` with over-fetch (n = 4k) and docid dedupe, `vsearch` fallback when BM25
returns nothing, disk cache keyed by (mode, k, query). Snippet locators
(`@@ -12,2 @@ ...`) are stripped -- useful to a human reading QMD output, pure
noise in a 0.6B's context window.

`controller.arm_r` is the benchmark-flow integration. Two query modes:
`query_mode="model"` prefills the assistant turn with ` ```repl\nprint(retrieve( `
and parses the query the model writes; `query_mode="question"` retrieves on the
raw question text and exists to measure whether the model-authored query is
worth its extra decode call.

**Comparability**: snippets enter Pass 1 only. Pass 2 still goes through the
frozen `build_trace_prompt`, byte-identical to every other arm, so retrieval can
move the score only through the trace it produced. Injecting snippets into the
Pass-2 prompt would change the scored sequence and break sec.2.3.

### Mechanism check (12 items -- a smoke test, NOT a result)

| Arm | Accuracy | Mean gold LL | Delta vs A | Improved | Mean ms |
| --- | --- | --- | --- | --- | --- |
| A | 0.583 | -1.8064 | - | - | 3023 |
| R (model query) | 0.583 | -1.6422 | +0.1642 | 8/12 | 3682 |
| R (question query) | 0.500 | -1.6311 | +0.1753 | 7/12 | 3065 |

Retrieval cache hit: 0.13 ms. Model-authored queries came out clean and
on-topic -- "zone analysis in permaculture", "worm species for vermicomposting",
"soil food web explanation" -- with the prefill working on 5/6 of an earlier
sample and falling back to the question text otherwise.

Three things to carry into #134, none of them settled at n=12:

1. The gold-LL gain is real in sign but the **accuracy did not move** (7/12 both
   ways). A Pass-2 log-likelihood shift that never crosses a decision boundary
   does not become a better answer on a phone.
2. **The model-authored query does not beat the raw question** (+0.164 vs
   +0.175). The prefill mechanism works, but on this evidence it costs an extra
   decode call and ~600 ms to do slightly worse. `query_mode="question"` is the
   cheaper default until a full run says otherwise.
3. The delta is the same order as sec.6.4's existing `delta_trace` (+0.167), so
   the sec.3.5 gate question is whether retrieval adds anything *on top of* the
   trace the champion already writes -- not whether it beats a no-trace baseline.
