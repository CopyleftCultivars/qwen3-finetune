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

## Update 2026-09-15: router fix + re-run

The owner chose "fix the router, then re-run" over a score-threshold
calibration (see `multi_tool_context.py`'s module docstring for why a score
threshold was tried and rejected: on-topic queries didn't even score higher
than off-topic ones on cosine/BM25 at this scale). The fix: run every tool
whose arguments are available (both are cheap, local, deterministic lookups)
and gate selection on whether it actually found real content, rather than on
the router's score. Full rationale in `rlm_slice/multi_tool_context.py`.

Re-run on the same `val_v3` 113 items, same `query_mode=question`:
`results/slice_175b_{A,R,M,N}.jsonl`, log `results/slice_175b.log`.

| Arm | Accuracy | mean delta_trace | note_counts |
| --- | ---: | ---: | --- |
| A | 0.7522 | 0.1964 | -- |
| R | 0.7080 | 0.2043 | fallback_to_vsearch: 12 |
| M | 0.6991 | 0.2037 | no_domain_tool_fired: 101 |
| N | 0.6991 | 0.2037 | no_domain_tool_fired: 101 |

**M and N are now numerically identical** -- every per-item row matches.
Checking why: of the 113 items, **0 had two domain tools simultaneously
viable**. 101 had none viable (correctly rejected -- the fix's whole point),
12 had exactly one. With only one viable tool ever in play at a time, "pick
the top-1 viable tool" (Arm M) and "use every viable tool" (Arm N) can never
diverge -- confirmed directly: `M and N selected different tool sets on 0/113
items`. **This dataset cannot exercise the actual routing decision #175 set
out to test** (choosing among *multiple* relevant tools), because the two
domain tools' relevance conditions essentially never co-occur on the same
`val_v3` question. Answering "does routing beat no-routing" needs either more
tools with overlapping relevance, or a dataset enriched with dual-relevant
items -- neither of which val_v3 (a general natural-farming validation set)
provides.

What the fix *did* verify: the forced-pick failure mode from the first sweep
is gone. Previously 66% of Arm M items picked a tool with nothing to execute;
now `no_domain_tool_fired` correctly reads 101/113 (89%) -- a materially more
credible number, since it now means "neither tool found real content" rather
than "the router picked something, but it happened to lack arguments."

On the 12 items where a domain tool actually fired (all `regional_recipe_lookup`
except two `bioaccumulator_ratio` items), restricting the comparison to just
those: R accuracy 0.667 vs M accuracy 0.583, one item flipped from correct to
incorrect (`row_000332`, `bioaccumulator_ratio`). **n=12 is far too small to
call this anything but anecdotal** -- one flip either way and the numbers
invert.

McNemar/z on the full 113 (A vs R, A vs M, R vs M): all still far below
significance (max chi2=1.786, A vs M) -- consistent with the first sweep, now
for the right reason (a correctly-gated router) rather than a broken
forced-pick artifact.

### Revised recommendation

The router fix worked as intended and is worth keeping regardless of what
happens next. But **val_v3 structurally cannot answer #175's routing
question** with only two domain tools this rarely co-relevant. Two honest
paths forward, neither started here:

1. **Build a small held-out eval slice weighted toward genuinely dual-relevant
   questions** (both a named crop+plant *and* a region/recipe angle in the
   same item) -- the only way to actually exercise Arm M vs Arm N's
   disagreement, since val_v3 never does.
2. **Accept the current finding as sufficient**: no detected harm from
   domain-tool injection (gated or not), the forced-pick defect is fixed, and
   the routing-vs-no-routing question is deferred as "untestable at this
   tool-set size on general-purpose data" rather than "answered no."

Which of these -- or whether to proceed to #176/#177 on the current
evidence -- is again a call for the epic owner, not a default this report
should pick on its own.
