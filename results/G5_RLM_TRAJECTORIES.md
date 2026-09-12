# G5 (#152): RLM trajectory collection from Qwen3-8B

Date: 2026-09-11. Model: `unsloth/qwen3-8b-unsloth-bnb-4bit`, in-process,
same infra as G3 (`rlm_slice/generate_g5_trajectories.py`, new script sharing
the model_host singleton with `rlm_slice.server`'s Arm-B REPL client, no
changes to any frozen `rlm_slice` module).

## Domain pool and contamination wall

Source: `data/train_v3.jsonl`'s MCQ subset (the same pool
`data/make_synthetic_traces.py` draws from for the analogous plain-teacher-
trace pipeline) — 1,107 MCQ-normalized records via `score_mcq_ll.normalize_records`.
Contamination check: `make_synthetic_traces.instruction_hash` /
`load_eval_hash_denylist` against `data/val_v3.jsonl` and
`eval_freeform/dev.jsonl` (never the sealed `eval_freeform/test.jsonl`, same
rule this repo enforces everywhere else). Denylist: 579 hashes.

**Result: 0 overlap.** All 1,107 records survive — train_v3/val_v3 were
already disjoint splits by construction, so this exclusion pass is a
confirmation, not a live filter. Sampled 200 items via `random.Random(42).shuffle`
over the full clean pool.

## Generation

Backend: the real `rlm` package's LocalREPL (`rlm_slice.rlm_arm.build_rlm`,
`max_depth=1`), greedy (temperature 0), same `_ARM_B_INSTRUCTION` prompt and
MCQ system prompt as G3's Arm B. Full per-item transcript captured via the
`rlm` package's `RLMLogger` (`result.metadata`) — every iteration's `prompt`,
`response` (including `<think>` reasoning and any ` ```repl ` block), REPL
`code`/`stdout`/`stderr`, and `final_answer` — not just the summary counts
`controller.arm_b` keeps. This is the actual SFT-training material for G6.

## Results

| | Count | Rate |
| --- | --- | --- |
| Items generated | 200/200 | 100% |
| Correct (top logprob letter == gold) | 136/200 | 68.0% |
| REPL-engaged (>=1 ` ```repl ` block) | 192/200 | 96.0% |
| Reached a submitted answer | 200/200 | 100% |
| **Kept after quality filter** | **131/200** | **65.5%** |

Accuracy (68.0%) is close to G3's Arm B accuracy on val_v3 (69.9%) — expected,
same model/backend/config, different item pool.

Quality filter (causal verification, same idea as
`data/filter_synthetic_traces.py`): keep a trajectory only if it is **both**
correct **and** REPL-engaged. A correct answer reached without touching the
scaffold isn't evidence the RLM behavior itself is trainable (it degraded to
bare reasoning and got lucky); a REPL-engaged but wrong trajectory isn't a
target worth imitating. 131 of 200 items clear both bars.

## Outputs

- `results/g5_trajectories_qwen3-8b.jsonl` — all 200 raw trajectories (8.8MB).
- `results/g5_trajectories_qwen3-8b_filtered.jsonl` — the 131 quality-filtered
  trajectories (5.7MB), the candidate SFT set for G6.
- `results/g5_generation_config.json` — model, backend, seed, pool
  provenance, denylist size, and the counts above, in one place.

## Accept criterion (#152)

Trajectory set built (200 generated, 131 filtered); zero eval overlap
reported (0/1,107, denylist size 579); generation config recorded
(`results/g5_generation_config.json`). Unblocks G6 (#153).
