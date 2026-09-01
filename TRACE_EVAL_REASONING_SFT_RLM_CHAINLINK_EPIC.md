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

## Milestone M3 - Minimal RLM slice, Flutter (parallel with M2 after I11)

- **I11** Cactus Flutter API recon -> `docs/cactus_flutter_api.md`.
  Accept: load / generate / KV / embedding APIs documented.
- **I12** `RlmController` 2-pass in Dart. Accept: same 2-pass shape as I1 on a sample. Dep: I11.
- **I13** Optional on-device RAG memory. Accept: top-k retrieval injected; toggle to disable. Dep: I12.
- **I14** Latency and memory budget harness on Pixel 7 plus single-pass fallback.
  Accept: ms and MB reported; fallback triggers over budget. Dep: I12.
- **I15** Device-to-server parity test plus hide-trace UI.
  Accept: outputs match on golden set; trace hidden by default. Dep: I12, I14.

## Milestone M4 - Cactus fork spike, gated (depends on M1-M3 results)

- **I16** Cactus capability report `docs/cactus_fork_feasibility.md` (Metis + true multi-stream).
  Accept: both questions answered with effort estimates.
- **I17** Go / no-go decision. Accept: written decision referencing M1-M3 results. Dep: I16, I10, I15.

## Blocker chain (summary)

- I1 -> I2 -> (I3) -> I4 -> I5 -> I10.
- I1 -> I6 -> I7 -> I8 -> I9 -> I10.
- I11 -> I12 -> {I13, I14} -> I15.
- {I10, I15} -> I16 -> I17.

## Review notes

1. M1 and the recon issue I11 can start immediately and in parallel.
2. M2 and M3 run in parallel after their first dependency clears.
3. M4 stays closed until M1-M3 produce results.

## Chainlink issue map

Epic issue #105, with I1-I17 created as its subissues #106-#122 (in order) and
grouped into milestones #1 (M1), #2 (M2), #3 (M3), #4 (M4). Blocks were set
from each issue's own `Dep:` line. `chainlink issue ready` confirms exactly
#106 (I1), #116 (I11), #121 (I16) are unblocked, matching review note 1.

**Discrepancy found and left unresolved**: the blocker-chain summary above
lists `I11 -> I12 -> {I13, I14} -> I15`, implying I15 depends on both I13 and
I14, but I15's own line only states `Dep: I12, I14`. The chainlink block was
set from the per-issue line (I15 blocked by I12 and I14 only, not I13). If
I15 was meant to also wait on I13, run `chainlink issue block 120 118`.
