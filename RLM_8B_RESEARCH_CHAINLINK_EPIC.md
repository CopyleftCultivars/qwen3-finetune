# Chainlink epic decomposition: RLM on Qwen3-8B research track

Date: 2026-09-03
Status: chainlink issues created (2026-09-03). Epic #147 (related to Epic F #137);
milestones #7 (M1), #8 (M2), #9 (M3), #10 (M4); G1-G9 map to #148-#156 in order.
See "Chainlink issue map" below.
Companion plan: `RLM_8B_RESEARCH_PLAN.md`.

## Epic

- **Epic G:** "RLM on Qwen3-8B: baseline, apply RLM, fine-tune for adaptability, then theorize
  downsizing to 0.6B."
- Goal: test whether the RLM approach helps Qwen3-8B on the benchmark, then whether fine-tuning 8B
  improves RLM adaptability and score, then whether the working method can be downsized.
- Scope: research probe. Not a deployment candidate. Only the Phase 3 downsized artifact must be
  Cactus-safe.

## Cross-epic note (required)

1. Epic F (`LADDER_AND_DISTILLATION_CHAINLINK_EPIC.md`) must inform the distillation procedure used
   here. If the Phase 1 gate (G4) passes, reuse Epic F's F5-F8 building blocks for G5, G6, and G9:
   - the contamination wall (exclude every eval item),
   - the quality filter,
   - the `autoresearch/finetune.py` config pattern.
2. In particular, G9 Path A (distill the 8B RLM behavior into the 0.6B champion) is a distillation
   step. It must follow the same contamination and gate rules as Epic F, not a new ad hoc process.
3. Run Epic F and Epic G in parallel. Do not merge their gates.

## Milestone M1 - Baseline Qwen3-8B (Phase 0)

- **G1** Serve Qwen3-8B (vLLM, OpenAI-compatible) and wire the Epic F harness to it.
  Accept: the harness scores one sample item end to end.
- **G2** Baseline prompted Qwen3-8B on the 210-item ladder and `val_v3`.
  Accept: scores with a 95% confidence interval recorded. Dep: G1.

## Milestone M2 - Apply RLM to Qwen3-8B (Phase 1)

- **G3** Run the `rlm_slice/` arms (A, B, R, C) against the 8B server; record RLM-adaptability
  metrics (REPL-block rate, valid-tool-call rate, decode calls, latency). Accept: per-arm scores
  plus adaptability metrics saved. Dep: G1.
- **G4** Phase-1 gate decision (kill or go). Accept: a written verdict.
  - Kill: no arm reaches the G2 baseline AND 8B does not drive the REPL. Stop the epic.
  - Go: 8B drives the REPL AND at least one arm shows a non-negative trend.
  Dep: G2, G3.

## Milestone M3 - Fine-tune Qwen3-8B for RLM adaptability (Phase 2, gated on G4 go)

- **G5** Trajectory SFT data: collect RLM trajectories from a teacher RLM on domain items; exclude
  all eval items. Reuse Epic F's contamination wall and quality filter (see cross-epic note).
  Accept: trajectory set built; zero eval overlap reported. Dep: G4.
- **G6** 8B LoRA SFT run on the trajectory set (Epic F finetune config pattern).
  Accept: adapter merged; artifact produced. Dep: G5.
- **G7** Evaluate the fine-tuned 8B: RLM-adaptability metrics plus benchmark score vs Phase 1.
  Accept: both deltas reported; accept/reject recorded. Dep: G6.
- **G8** (optional, heavier) RL rung via the RLM `training/` harness (verifiers + prime-rl) with a
  domain reward. Accept: run completes; benchmark and adaptability deltas reported. Dep: G7.

## Milestone M4 - Downsizing theory to 0.6B (Phase 3, gated on G7 accept)

- **G9** Written transfer analysis and a go/no-go for a 0.6B implementation. Identify the minimal
  sub-skill that produced the gain, and compare Path A (distill 8B RLM behavior into 0.6B, using
  Epic F's distillation procedure) versus Path B (extract one fixed Cactus-safe step).
  Accept: analysis plus a recorded decision. No 0.6B code before this passes. Dep: G7 (and G8 if run).

## Blocker chain (summary)

- G1 -> G2 ; G1 -> G3 ; {G2, G3} -> G4.
- G4 (go) -> G5 -> G6 -> G7.
- G7 -> G8 (optional).
- {G7 accept, or G8} -> G9.

## Review notes

1. M1 and M2 are cheap (inference only). Start them immediately.
2. M3 starts only on a G4 go. It reuses Epic F's distillation building blocks.
3. M4 is analysis only, gated on a real G7 accept.
4. Keep #121 (Metis) and #122 (Cactus fork) deferred.

## Chainlink issue map

Epic issue #147, with G1-G9 created as its subissues #148-#156 (in order), labelled `rlm-8b`,
and grouped into milestones #7 (M1, G1-G2), #8 (M2, G3-G4), #9 (M3, G5-G8) and #10 (M4, G9).
Blocks were set from the dependency chain above. #147 is linked as *related* to Epic F (#137).

| Issue | Item | Blocked by |
| --- | --- | --- |
| #148 | G1 serve Qwen3-8B (vLLM) + wire the Epic F harness | - |
| #149 | G2 baseline prompted 8B on the 210-item ladder + val_v3 | #148 |
| #150 | G3 rlm_slice arms A/B/R/C + adaptability metrics | #148 |
| #151 | G4 Phase-1 kill/go gate | #149, #150 |
| #152 | G5 trajectory-SFT data (Epic F contamination wall + filter) | #151 |
| #153 | G6 8B LoRA SFT run | #152 |
| #154 | G7 evaluate fine-tuned 8B, accept/reject | #153 |
| #155 | G8 (optional) RL rung via verifiers + prime-rl | #154 |
| #156 | G9 downsizing analysis + 0.6B go/no-go | #154 |

Entry point: **#148** is the only unblocked item. G8 (#155) is optional; G9 (#156) is blocked on
G7 (#154) directly, so skipping G8 does not strand M4.

### Ladder retarget (2026-09-03)

G2 originally said "98-question ladder". Per Epic F's amendment (#138) that dataset is
unrecoverable, so G2 as filed targets the 210-item
`natural_fertilizers_benchmark_v1_graded.jsonl` at `--temperature 0`, with qwen3-8b 0.895 in
`quality/irt/responses/` as the existing reference point.
