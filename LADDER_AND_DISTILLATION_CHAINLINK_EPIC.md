# Chainlink epic decomposition: 98-question ladder + distillation lever

Date: 2026-09-03
Status: chainlink issues created (2026-09-03). Epic #137; milestones #5 (M1) and #6 (M2);
F1-F9 map to #138-#146 in order. See "Chainlink issue map" below.
Companion plan: `LADDER_AND_DISTILLATION_PLAN.md`.

## Epic

- **Epic F:** "Quantify the size gap on the 98-question ladder, then distill toward it."
- Goal: place v5clean on the ladder (Part A), then run one contamination-guarded, Cactus-safe
  distillation lever aimed at the target line (Part B).

## Milestone M1 - Ladder measurement (Part A)

- **F1** Locate the 98-question dataset and the generative scorer; document the protocol.
  Accept: file path, prompt template, parser, and "ambiguous" rule recorded.
- **F2** Contamination check: 98 questions vs `train_v3`, `train_v6`, `val_v3`.
  Accept: overlap count recorded; contaminated items flagged. Dep: F1.
- **F3** Rebuild the ladder, one protocol, temperature = 0: base 0.6B (target ~28.6%), champion,
  and qwen3:8b if available. Accept: raw per-item outputs saved for each model. Dep: F1.
- **F4** Report `results/LADDER_98Q_RESULTS.md`: table, Wilson 95% CI (n = 98), ambiguous rate,
  contamination note, and the champion-to-8b target line. Dep: F2, F3.

## Milestone M2 - Distillation lever (Part B)

- **F5** Teacher-data generation with qwen3:8b over the QMD/`mcq_generation` pool; exclude the 98
  eval questions. Accept: teacher outputs saved; generation config recorded. Dep: F4.
- **F6** Contamination wall + quality filter + dataset build (`prepare_dataset_v7` or a `_distill`
  path). Accept: zero eval overlap; filter drop-rate reported; train/val built. Dep: F5.
- **F7** Config `phaseNN-distill` + training run (single lever = dataset; v5clean hyperparameters).
  Accept: adapter merged; artifact produced. Dep: F6.
- **F8** Evaluate + gate: frozen MCQ primary/trace track AND the 98-question ladder.
  Accept: both deltas reported; guardrails clean; accept/reject recorded. Dep: F4, F7.
- **F9** (gated on F8 pass) Cactus conversion + on-device ladder score.
  Accept: on-device number reported. Dep: F8.

## Blocker chain (summary)

- F1 -> F2 -> F4 ; F1 -> F3 -> F4.
- F4 -> F5 -> F6 -> F7 -> F8 -> F9.

## Review notes

1. M1 is small and self-contained. Start it immediately.
2. M2 starts only after F4 gives a real target line.
3. F9 (Cactus) runs only if F8 accepts a candidate.
4. Keep #121 (Metis) and #122 (Cactus fork) deferred. The M3 evidence says the lever is
   training-side, not architecture or inference-harness.

## Chainlink issue map

Epic issue #137, with F1-F9 created as its subissues #138-#146 (in order), labelled
`ladder-distill`, and grouped into milestones #5 (M1, F1-F4) and #6 (M2, F5-F9).
Blocks were set from the dependency chain above.

| Issue | Item | Blocked by |
| --- | --- | --- |
| #138 | F1 locate dataset + scorer, document protocol | - |
| #139 | F2 contamination check vs train_v3/train_v6/val_v3 | #138 |
| #140 | F3 rebuild ladder, one protocol | #138 |
| #141 | F4 report with Wilson CIs + target line | #139, #140 |
| #142 | F5 teacher-data generation (qwen3:8b) | #141 |
| #143 | F6 contamination wall + quality filter + dataset | #142 |
| #144 | F7 phaseNN-distill config + training run | #143 |
| #145 | F8 gate on frozen MCQ primary AND the ladder | #144, #141 |
| #146 | F9 Cactus conversion + on-device ladder score | #145 |

Entry point: **#138** is the only unblocked item. #121 (Metis) and #122 (Cactus fork) carry
deferral comments pointing at review note 4; Cactus work re-enters only through F9 (#146).
