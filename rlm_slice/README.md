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
