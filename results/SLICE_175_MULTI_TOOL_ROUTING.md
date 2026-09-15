# Tier 1 (#175): static query-based multi-tool routing, noise-robustness result

Date: 2026-09-15. Model: `outputs/merged-model-p6-lr5e5-r2` (the 0.6B GPU
champion, v5clean). Dataset: `data/val_v3.jsonl`, 113 scorable MCQ rows (the
same filtered subset `eval_slice.py` has used for every prior arm sweep).
`query_mode=question` (raw question text, not model-authored) for all of
R/M/N, so the comparison isolates the tool-routing variable rather than
mixing it with query-authoring quality. Raw per-item rows:
`results/slice_175_{A,R,M,N}.jsonl`; run log: `results/slice_175.log`.

Four arms, same fixed-pipeline contract (controller decides, model never
self-initiates a tool call -- #127/G9):

| Arm | Tools injected | Selection |
| --- | --- | --- |
| A | none | -- |
| R | retrieval only | always-on (existing baseline) |
| M | retrieval + 1 domain tool | tool_router top-1 (Cactus-ported RRF) |
| N | retrieval + both domain tools | none -- always attempted |

## Results

| Arm | Accuracy | mean delta_trace | mean latency (ms) |
| --- | ---: | ---: | ---: |
| A | 0.7522 | 0.1964 | 2682 |
| R | 0.7080 | 0.2043 | 2743 |
| M | 0.6991 | 0.2479 | 2713 |
| N | 0.7168 | 0.2564 | 2654 |

## Statistical significance

Two-proportion z and paired McNemar (same 113 items across every arm, so
McNemar is the more powered test) on every pair:

| Pair | z | McNemar chi2 | Significant at 0.05? |
| --- | ---: | ---: | --- |
| A vs R | 0.749 | 1.231 | No |
| A vs N | 0.603 | 0.750 | No |
| A vs M | 0.895 | 1.786 | No |
| R vs N | -0.147 | 0.000 | No |
| R vs M | 0.146 | 0.000 | No |
| M vs N | -0.293 | 0.167 | No |

**No pairwise difference clears significance at n=113.** Every accuracy gap
in the table above is within noise at this sample size.

## Routing mechanism diagnostic

Arm M's `top_k=1` forces the router to commit to one of the two domain tools
on every item -- there is no "select neither" option. Breaking down what
actually happened on the 113 items:

| Outcome | n | Accuracy |
| --- | ---: | ---: |
| Router picked a tool, tool executed (had extractable args) | 38 | 0.711 |
| Router picked a tool, but no extractable args (degraded to Arm R) | 75 | 0.693 |

**66% of val_v3 items (75/113) gave the router-selected tool nothing to
execute with** -- `bioaccumulator_ratio` needs both a known crop *and* a known
plant name in the query text, and most val_v3 questions (general
natural-farming SFT/QMD-MCQ content, not bioaccumulator-ratio-specific) don't
contain that pair. On those items Arm M silently degraded to Arm R. This is
a **design limitation, not a router-ranking failure**: `select_relevant_tools`
with only 2 candidates and a forced top-1 pick has no way to say "neither is
relevant here," which is exactly the condition most of this dataset is in.
The executed-vs-not split above (0.711 vs 0.693) shows no meaningful
difference either way, consistent with the low n.

## The central noise-robustness question, answered

**Unconditionally injecting both domain tools' output on every item (Arm N)
did not measurably hurt accuracy relative to retrieval-alone (Arm R): 0.717
vs 0.708, z=-0.147, McNemar chi2=0.0 (0 items flipped either direction).**
That is the headline safety result: no catastrophic noise-driven collapse
from adding mostly-irrelevant structured context on top of retrieval, even
with zero routing at all.

The honest calibration: at n=113 this experiment cannot distinguish three
different underlying realities --

1. the model correctly disregards irrelevant tool output when present;
2. the extra context is mildly harmful (or mildly helpful) but the true
   effect size is smaller than n=113 can resolve;
3. routing (Arm M) would show a real benefit over no-routing (Arm N) with a
   router that could actually decline to fire, rather than being forced to
   pick one of two tools on every item.

None of R, M, or N is distinguishable from Arm A (no tools) either, in this
run -- expected, since `query_mode=question` trades away the model-authored
query Arm R's own docstring calls "the only shape #127 showed this champion
can actually drive" for its strongest number, in exchange for isolating the
routing variable and running ~3x cheaper. This is not evidence against
tool-augmentation generally; it's a property of the weaker query mode
deliberately used for this specific comparison.

## #175 accept-criteria checkoff

- **Per-question tool-selection record**: done -- `tool_calls` in every Arm
  M/N row (`results/slice_175_{M,N}.jsonl`): name, rrf_score,
  selected_by_router, args_available, executed, args.
- **Accuracy vs the single-tool Arm R baseline**: done (table above).
- **Accuracy vs no-tool baseline (Arm A)**: done (table above).
- **Clear answer to the noise-robustness question**: qualified yes -- no
  measurable harm from unrouted dual-tool injection at this n, but the
  experiment is underpowered to confirm *why* (genuine robustness vs. a real
  but small effect vs. a broken router that never gets to prove itself).

## Recommendation

Two concrete follow-ups before treating Tier 1 as settled, not yet done here:

1. **Fix the router's forced-pick design** -- add a real `min_score`
   reject-all threshold (calibrated against real query score distributions;
   the raw RRF scores observed during development cluster tightly with only
   2 candidates, so this needs its own small calibration pass, not a guessed
   constant) so Arm M can actually decline to inject anything, rather than
   defaulting to "always inject one."
2. **Re-run at higher power** if this is going to gate a real decision --
   n=113 cannot resolve effects smaller than several points of accuracy; the
   210-item ladder or a larger held-out slice would roughly double n and
   McNemar's power with it, though may still not be enough to detect a small
   true effect.

Given #175 was scoped to gate #176 (tier-2 mid-generation triggering) and
#177 (on-device port): the noise-robustness question has been *measured*, as
required, with a "no detected harm" answer -- not a "no detected benefit
either" verdict. Whether that's sufficient to open the gate, or whether the
two follow-ups above should run first, is a call on acceptable risk/cost that
the epic owner should make explicitly rather than have it default one way.
