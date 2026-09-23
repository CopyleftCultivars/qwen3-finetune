# D203 NORM SWEEP (#222) — post-transform object: is any radial clamp cap viable for reserved rows?

Milestone #14 (Epic #179), parent #179, sibling of #216/#203/#204. Decides
whether #223 (Design A engine work: `inputs_embeds` media injection) is
**optional** or **mandatory**.

**Headline: NO VIABLE CAP FOUND. Design A is MANDATORY, not merely preferred.**

Sweeping the corrected post-transform object `out_ln[h](v + mlp(v))` (norm
≈ 37, #216) down through M1 #186's norm ≤ 4.0 tied-row safe zone and below, on
the exact paired #216 protocol:

- The **reserved-row path never carries above-chance recall at any cap** —
  not at 37, not at 4.0, not at 1.0. Its recall scatters around chance
  (0.0–0.21 over 24 evals) with no trend; the only thing the sweep changes is
  how badly the tied rows contaminate (no-memory loss 321 → 4.16).
- The **dynamic soft-prompt path shows the content is real** (0.17–0.29
  unclamped, inside #216's band) **but that recall itself collapses with the
  clamp**: at the literal M1 boundary cap = 4.0 it is 1/24 (default) and 2–3/24
  (m193) — at/below chance (0.0833).
- The two requirements never overlap: tied-row contamination only clears at
  **norm ≤ 2**, while the dynamic content signal is already gone by **norm ≈ 4**.
  Scaling the object into the embedding-compatible norm band destroys exactly
  the post-LayerNorm magnitude the backbone reads as memory.

## Method (reuse, not rebuild)

New `l1_sweep` subcommand in `scripts/lpm_quant_gap.py`, built entirely from
#216's pieces, unchanged:

- `fit_val_slots` → `s0_select` → `l1_transform_selected` — the SAME fitted
  `val_module` and the SAME S0 selection (#211 math, **recomputed in-run**;
  slot fitting is GPU-nondeterministic, so saved indices from another run do
  not address a fresh fit).
- `project_to_ball(transformed, cap)` (`lpm/memory.py:123`) — the exact radial
  clamp M1 already uses (`vectors * min(1, cap/||v||)`). `cap=None` is the
  unclamped #216 object.
- `fixed_prompt_recall_matrix` — the dynamic soft-prompt scorer.
- `score_reserved_rows` — #216's `reserved_arm` closure extracted verbatim:
  writes the `[K,H,d]` vectors into `model.get_input_embeddings().weight.data`
  at `PACK_BASE + persona*H + head` **in memory** (no disk conversion), then
  scores with `recall_matrix` + `no_memory_loss` + `summarize`.

Both paths are scored for **every cap in one run, on the same fitted module** —
paired, not cross-run. The dynamic path is measured with the tied rows restored
to their original values (patched rows perturb the lm_head logits even for
soft-prompt inputs); the reserved path's `no_memory_loss` is measured *after*
its own patch, so its explosion is the contamination signature.

Caps swept: `None, 32, 24, 20, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0.5`
(run 1 uses the coarser `None,32,24,16,12,8,6,4,3,2,1,0.5` grid; the rerun adds
the intermediate points to localize the transition).

## Protocol (matches #216)

val corpus `data/persona_memory/lpm_corpus_val.jsonl`, checkpoint
`outputs/lpm-persona-memory-192/checkpoint.pt`, base model
`outputs/merged-model-distill-v2-rationale` (bf16), 6-layer encoder, 24 val
personas in two disjoint K=12 slices (offsets 0, 12), both fit budgets
(issue-default 40/128/192 and #193's 20/96/160), seed 0. Per-slice chance
0.0833; 24-way 0.0417. Plus a **full independent re-run of all four
configurations** (the `_rerun` / `_m193fit_rerun` artifacts), which is the
required independent re-run of the recommendation.

## Sanity cross-check vs #216 (unclamped, cap = None)

The unclamped arm reproduces #216's committed JSONs to within run-to-run noise —
the sweep is measuring the same object, and the negative is not a protocol
artifact:

| slice | #216 transformed dyn recall / own NTP | this sweep (run1) | #216 transformed reserved own NTP / no-mem | this sweep (run1) |
|---|---|---|---|---|
| default off0 | 0.2500 / 2.2995 | 0.3333 / 2.2975 | 391.67 / 320.0 | 394.83 / 322.0 |
| default off12 | 0.1667 / 3.2682 | 0.2500 / 3.2695 | 395.00 / 320.0 | 393.83 / 320.0 |
| m193 off0 | 0.2500 / 2.3646 | 0.1667 / 2.3874 | 375.00 / 296.0 | 374.50 / 294.0 |
| m193 off12 | 0.1667 / 3.2917 | 0.1667 / 3.2865 | 390.17 / 308.0 | 390.00 / 308.0 |

Own-target NTP matches to ≤ 0.025 nats in all four slices and the reserved
own-NTP/no-mem reproduce #216's catastrophic band; the recall counts differ
only by the known in-run S0 re-selection (slot fitting is nondeterministic —
#216 documented the same per-slice spread, 0.0–0.33).

Aggregate unclamped dynamic recall: default 0.2917 (7/24), m193 0.1667 (4/24) —
both inside #216's reported 0.19–0.25 band. Aggregate unclamped reserved
own-NTP: 382–394 (default), 382–386 (m193) — #216's 375–400 catastrophic band.

## Full sweep table

Each row aggregates the two K=12 slices → 24 evals (chance 0.0833); the two
entries per cell are the independent `run1` / `rerun`. `no-mem` is the
reserved arm's no-memory NTP (tied-row contamination); the uncontaminated floor
is ≈ 4.16.

### budget default (40/128/192)

| cap | dynamic recall@1 | dynamic own NTP | reserved recall@1 | reserved own NTP | reserved no-mem |
|---|---|---|---|---|---|
| None | 0.2917 (7/24) / 0.2917 (7/24) | 2.784 / 2.787 | 0.1250 (3/24) / 0.1667 (4/24) | 394.33 / 391.33 | 321.00 / 319.00 |
| 32.0 | 0.2917 (7/24) / 0.2917 (7/24) | 2.794 / 2.797 | 0.0833 (2/24) / 0.1667 (4/24) | 335.58 / 333.50 | 281.00 / 278.00 |
| 24.0 | 0.2917 (7/24) / 0.3750 (9/24) | 2.847 / 2.844 | 0.1250 (3/24) / 0.1667 (4/24) | 242.17 / 240.62 | 206.00 / 205.00 |
| 20.0 | — / 0.4167 (10/24) | — / 2.858 | — / 0.0833 (2/24) | — / 168.0 | — / 168.00 |
| 16.0 | 0.1250 (3/24) / 0.1250 (3/24) | 3.011 / 3.016 | 0.0833 (2/24) / 0.1667 (4/24) | 153.40 / 152.38 | 132.00 / 131.50 |
| 14.0 | — / 0.1667 (4/24) | — / 3.047 | — / 0.1250 (3/24) | — / 112.75 | — / 112.75 |
| 12.0 | 0.2083 (5/24) / 0.2083 (5/24) | 3.094 / 3.104 | 0.0833 (2/24) / 0.1250 (3/24) | 111.04 / 110.02 | 95.00 / 94.50 |
| 10.0 | — / 0.1667 (4/24) | — / 3.205 | — / 0.0000 (0/24) | — / 76.0 | — / 76.00 |
| 8.0 | 0.1250 (3/24) / 0.1250 (3/24) | 3.359 / 3.359 | 0.0417 (1/24) / 0.0833 (2/24) | 66.50 / 65.85 | 58.12 / 57.62 |
| 7.0 | — / 0.1667 (4/24) | — / 3.466 | — / 0.1667 (4/24) | — / 48.38 | — / 48.38 |
| 6.0 | 0.1250 (3/24) / 0.1667 (4/24) | 3.615 / 3.573 | 0.1667 (4/24) / 0.0417 (1/24) | 44.62 / 44.46 | 39.75 / 39.38 |
| 5.0 | — / 0.1667 (4/24) | — / 3.601 | — / 0.0833 (2/24) | — / 30.44 | — / 30.44 |
| **4.0** | **0.0417 (1/24) / 0.0417 (1/24)** | **3.642 / 3.601** | **0.1250 (3/24) / 0.1250 (3/24)** | **22.97 / 22.73** | **21.88 / 21.56** |
| 3.0 | 0.0833 (2/24) / 0.0833 (2/24) | 3.651 / 3.666 | 0.0417 (1/24) / 0.0000 (0/24) | 13.42 / 13.39 | 13.41 / 13.22 |
| 2.0 | 0.0000 (0/24) / 0.0833 (2/24) | 3.621 / 3.599 | 0.0417 (1/24) / 0.0417 (1/24) | 6.29 / 6.18 | 6.44 / 6.36 |
| 1.0 | 0.0833 (2/24) / 0.1250 (3/24) | 3.629 / 3.631 | 0.0833 (2/24) / 0.0417 (1/24) | 3.64 / 3.64 | 4.16 / 4.16 |
| 0.5 | 0.2083 (5/24) / 0.0833 (2/24) | 3.627 / 3.621 | 0.2083 (5/24) / 0.0833 (2/24) | 3.63 / 3.62 | 4.15 / 4.15 |

### budget m193 (20/96/160)

| cap | dynamic recall@1 | dynamic own NTP | reserved recall@1 | reserved own NTP | reserved no-mem |
|---|---|---|---|---|---|
| None | 0.1667 (4/24) / 0.2500 (6/24) | 2.837 / 2.828 | 0.1667 (4/24) / 0.0833 (2/24) | 382.25 / 385.92 | 301.00 / 302.00 |
| 32.0 | 0.2500 (6/24) / 0.2083 (5/24) | 2.841 / 2.840 | 0.1667 (4/24) / 0.0833 (2/24) | 321.58 / 323.29 | 260.00 / 260.50 |
| 24.0 | 0.2917 (7/24) / 0.1667 (4/24) | 2.875 / 2.876 | 0.0833 (2/24) / 0.0833 (2/24) | 231.79 / 232.71 | 190.50 / 191.00 |
| 20.0 | — / 0.2500 (6/24) | — / 2.857 | — / 0.0833 (2/24) | — / 157.0 | — / 157.00 |
| 16.0 | 0.2083 (5/24) / 0.1250 (3/24) | 3.037 / 3.034 | 0.0833 (2/24) / 0.0833 (2/24) | 145.46 / 146.02 | 121.75 / 122.25 |
| 14.0 | — / 0.1250 (3/24) | — / 3.086 | — / 0.1250 (3/24) | — / 104.75 | — / 104.75 |
| 12.0 | 0.0417 (1/24) / 0.0833 (2/24) | 3.111 / 3.117 | 0.0833 (2/24) / 0.0833 (2/24) | 104.62 / 105.19 | 87.50 / 87.50 |
| 10.0 | — / 0.0833 (2/24) | — / 3.219 | — / 0.0833 (2/24) | — / 70.50 | — / 70.50 |
| 8.0 | 0.1667 (4/24) / 0.2083 (5/24) | 3.332 / 3.339 | 0.1667 (4/24) / 0.0833 (2/24) | 62.79 / 62.82 | 53.12 / 53.25 |
| 7.0 | — / 0.0833 (2/24) | — / 3.432 | — / 0.0833 (2/24) | — / 44.62 | — / 44.62 |
| 6.0 | 0.2083 (5/24) / 0.0833 (2/24) | 3.530 / 3.547 | 0.0833 (2/24) / 0.0833 (2/24) | 42.40 / 42.39 | 36.25 / 36.12 |
| 5.0 | — / 0.0417 (1/24) | — / 3.612 | — / 0.0833 (2/24) | — / 27.88 | — / 27.88 |
| **4.0** | **0.1250 (3/24) / 0.0833 (2/24)** | **3.611 / 3.648** | **0.0000 (0/24) / 0.1250 (3/24)** | **21.55 / 21.62** | **19.81 / 19.69** |
| 3.0 | 0.0417 (1/24) / 0.0833 (2/24) | 3.664 / 3.675 | 0.1667 (4/24) / 0.0833 (2/24) | 12.06 / 12.20 | 12.09 / 12.06 |
| 2.0 | 0.0000 (0/24) / 0.0833 (2/24) | 3.605 / 3.600 | 0.0833 (2/24) / 0.0417 (1/24) | 5.75 / 5.75 | 6.03 / 6.00 |
| 1.0 | 0.0417 (1/24) / 0.0000 (0/24) | 3.628 / 3.630 | 0.0417 (1/24) / 0.0000 (0/24) | 3.63 / 3.64 | 4.16 / 4.16 |
| 0.5 | 0.0000 (0/24) / 0.0417 (1/24) | 3.642 / 3.628 | 0.0000 (0/24) / 0.0417 (1/24) | 3.64 / 3.63 | 4.16 / 4.16 |

Raw (pre-transform) in-run references: dynamic recall 0.083–0.125, own NTP
3.85–3.89; reserved recall 0.042–0.167, own NTP 9.6–11.9, no-mem 8.4–10.5
(reproducing #216's raw rows).

## Divergence between the two paths (explicitly, not averaged away)

The two paths disagree in a way that is itself the result:

- **High norm (cap ≳ 24): the dynamic path holds up but reserved still
  contaminates.** Dynamic recall is 0.17–0.42, yet reserved recall is at chance
  and reserved no-mem is 190–321 (45–77× the 4.16 floor). The tied-row softmax
  is saturated by the patched rows, so the reserved path cannot discriminate
  even though the object demonstrably carries content on the injection path.
  This is #216's finding reproduced, now across a norm grid.
- **Low norm (cap ≤ 2): contamination is gone but so is the signal.** Reserved
  no-mem reaches the 4.16 floor, but dynamic recall is at/below chance and
  reserved recall is at chance. The clamp has scaled the post-LayerNorm
  magnitude the backbone reads as memory down to nothing.
- There is **no cap where both hold**. The dynamic recall's collapse (≈ norm 4)
  is *above* the contamination-clearing threshold (≈ norm 2).

## Transition / collapse point

- Tied-row contamination falls monotonically with the cap: reserved no-mem
  321 → 206 (cap 24) → 132 (16) → 95 (12) → 58 (8) → 40 (6) → **21.9 (4.0)** →
  13.4 (3) → 6.4 (2) → **4.16 (1)**. It clears the raw-reserved band
  (8.4–10.5) only at cap ≤ 2 and reaches the uncontaminated floor at cap ≤ 1.
- **Even at norm 4.0 — the literal M1 #186 boundary — 8 of these vectors still
  contaminate: tied-row no-mem is ~20–22, ≈ 5× the 4.16 floor and ≈ 2× the
  raw-reserved rows.** As the issue warned, 4.0 is not automatically safe for
  this quantity (a LayerNorm-affine output, not the raw `w_v` vector #193's
  norm-vs-contamination curve was calibrated on).
- Dynamic recall is stable at 0.29–0.42 down to cap 24, becomes noisy/marginal
  (0.04–0.21) from cap 16 to cap 5, and is **at/below chance by cap 4.0**
  (1/24 default; 2–3/24 m193). So the content signal collapses in the norm
  4–6 range — before, not after, the safe zone.

## Verdict on #223

**No viable reserved-row cap exists. Design A (`inputs_embeds` media injection,
`cactus-engine/src/model.cpp:3738 write_media_row_at` into
`lm_encoder_media_chunk_`) is MANDATORY, not optional.** #223 is required and
must not be dropped.

The reason is structural, not a tuning failure: the object the backbone was
trained to read is a post-LayerNorm activation (norm ≈ 37 = `sqrt(1024)` scaled
by learned affine `gamma`). To fit a tied embedding row it must be radially
clamped to the embedding norm band (≤ ~4), but that clamp removes exactly the
magnitude the backbone uses; below the contamination threshold the memory is
inert. Conversely, at the norm where the memory is legible, the tied lm_head
rows are catastrophically contaminated. The corrected object can only be
delivered on a surface that never touches the tied embedding table — the
engine's existing `inputs_embeds` media path.

This is a sharper, more specific confirmation than #216 alone: "even at the M1
boundary norm 4.0, tied-row no-memory NTP is still ~20–22 (vs a 4.16 floor),
while dynamic recall has already fallen to chance." The #205–207 suppression
patch remains valid engineering, but it suppresses reserved-ID *emission* — it
cannot make a 20× contaminated tied row a legible memory prompt.

## Reproduce

```bash
PY=[LOCAL_PATH]

# issue-default fit budget (40/128/192), two slices, coarse grid
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 0
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 12

# #193's exact fit budget (20/96/160), two slices
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 0  --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 12 --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit

# independent stability re-run, finer grid to localize the transition
CAPS="none,32,24,20,16,14,12,10,8,7,6,5,4,3,2,1,0.5"
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 0  --caps "$CAPS" --out-tag _rerun
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 12 --caps "$CAPS" --out-tag _rerun
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 0  --fit-steps 20 --fit-max-length 96 --max-length 160 --caps "$CAPS" --out-tag _m193fit_rerun
$PY scripts/lpm_quant_gap.py l1_sweep --n-personas 12 --persona-offset 12 --fit-steps 20 --fit-max-length 96 --max-length 160 --caps "$CAPS" --out-tag _m193fit_rerun

# independent comparator/report over the committed artifacts (exits 0 on PASS)
$PY scripts/norm_sweep_report.py
```

Artifacts: `results/D203_NORM_SWEEP_offset{0,12}[_m193fit][_rerun].json` (8
files) — each records the cap grid, per-cap clamped norm stats, and both the
dynamic and reserved arm (recall@1, own/offdiag NTP, no-memory NTP) for the
same fitted module, plus the in-run raw references and S0 slot indices.
`scripts/norm_sweep_report.py` re-derives the tables and verdict from those
files alone and asserts the invariants (caps respected, monotone
decontamination, #216 unclamped cross-check, no viable cap).
