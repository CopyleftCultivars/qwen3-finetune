# Chainlink epic decomposition: Trace-conditioned eval + reasoning SFT + minimal RLM slice

Date: 2026-09-01
Status: chainlink issues created (2026-09-01). Epic #105; milestones #1-#4; issues #106-#122
(I1-I17 map to #106-#122 in order). See "Chainlink issue map" below.

Companion plan: `TRACE_EVAL_REASONING_SFT_RLM_PLAN.md`.

## Epic

- **Epic E:** "Trace-conditioned eval + reasoning SFT + minimal on-device RLM slice."
- Goal: make reasoning measurable (Plan A), teach reasoning (Plan B), and deploy a minimal 2-pass
  path on device (Part 2). Keep the Qwen3 graph and the Cactus conversion intact.

## Milestone M1 - Plan A eval (blocks M2 accept)

- **I1** Add `score_mcq_trace_ll.py` (2-pass, pass 2 = option-set-norm gold-letter LL).
  Accept: runs on `data/val_v3.jsonl`; writes per-item JSONL.
- **I2** Determinism and leakage guards (temperature = 0, seed, gold stripped, sealed denylist).
  Accept: repeated runs match; sealed paths refused. Dep: I1.
- **I3** Ablation guard `delta_trace`. Accept: both scores logged per item. Dep: I1.
- **I4** New baseline plus `objective_weights_v3.json` plus `program.md` note.
  Accept: `results/mcq_trace_ll_<champion>.json` exists; policy states non-comparability.
  Dep: I2, I3.
- **I5** Gate integration as a PARALLEL track in the `gate.py` wrapper.
  Accept: gate reads the new track without touching frozen files. Dep: I4.

## Milestone M2 - Plan B SFT (depends on M1)

- **I6** `make_synthetic_traces.py` plus teacher config.
  Accept: traces generated for train rows only. Dep: I1.
- **I7** Contamination wall plus causal verification plus quality filter.
  Accept: zero eval overlap; wrong-letter traces dropped; drop-rate reported. Dep: I6.
- **I8** `prepare_dataset_v6.py` (separable think/answer).
  Accept: valid chat records; schema check passes. Dep: I7.
- **I9** `configs/<phase>-trace-sft.yaml` plus training run.
  Accept: adapter merged; artifact produced. Dep: I8.
- **I10** Evaluate plus gate vs trace-LL baseline.
  Accept: primary delta reported; `delta_trace > 0` majority; rubric clean. Dep: I5, I9.

## Milestone M3 - Qwen3 RLM model proof (parallel with M2)

- **I11** Qwen3/v5clean reference-runtime recon.
  Accept: generation / 2-pass / KV / embedding interfaces documented; no Cactus changes.
- **I12** Qwen3/v5clean 2-pass RLM reference test.
  Accept: reproducible outputs and quality metrics versus frozen v5clean. Dep: I11.
- **I13** Qwen3/v5clean external RAG-memory test.
  Accept: top-k retrieval injected; toggle to disable; quality delta reported. Dep: I12.
- **I14** Qwen3 RLM latency and memory reference benchmark.
  Accept: ms and MB reported; fallback triggers over budget. Dep: I12.
- **I15** RLM model-quality gate versus frozen v5clean.
  Accept: primary, trace benefit, rubric/forgetting guards, and pass/reject reported. Dep: I12, I13, I14.

## Milestone M4 - Qwen3 Metis-style model proof and integration gate

- **I16** Qwen3 Metis-style memory experiment versus v5clean.
  Accept: reproducible mechanism plus quality/resource deltas and pass/reject; no Cactus changes. Dep: I11.
- **I17** Go / no-go for Cactus integration after a model win.
  Accept: GO only if I15 or I16 materially and cleanly beats frozen v5clean; otherwise NO-GO.
  Dep: I16, I10, I15.

## Blocker chain (summary)

- I1 -> I2 -> (I3) -> I4 -> I5 -> I10.
- I1 -> I6 -> I7 -> I8 -> I9 -> I10.
- I11 -> I12 -> {I13, I14} -> I15.
- I11 -> I16.
- {I10, I15, I16} -> I17.

## Review notes

1. M1 and the Qwen3 reference-runtime recon I11 can start immediately and in parallel.
2. M2 and the model-first M3/M4 experiments run without Cactus modifications.
3. Flutter, Pixel, and Cactus work stays closed until a candidate beats frozen v5clean.

## Chainlink issue map

Epic issue #105, with I1-I18 created as its subissues #106-#123 (in order) and
grouped into milestones #1 (M1), #2 (M2), #3 (M3), #4 (M4). Blocks were set
from the model-first dependency chain above. I18 (#123) is the bounded Phase24
failure analysis and starts after I10 records its final evaluation result.
