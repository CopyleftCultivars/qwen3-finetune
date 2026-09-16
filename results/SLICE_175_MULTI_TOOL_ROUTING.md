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

## Update 2026-09-15 (2): dual-relevant eval slice built and run

The owner chose "build a dual-relevant eval slice." Two things surfaced
while building it, ahead of the sweep itself:

**A real bug in `bioaccumulator_ratio` (#174), unrelated to routing.**
`_parse_range`'s regex allowed a leading `-?` on every matched number, which
misread a range string's own separating hyphen as a unary minus: `"30-70"`
parsed via `findall` as `[30, -70]` instead of `[30, 70]`, so every NPK bound
-- and therefore every ratio the tool has ever computed since #174 shipped --
was silently wrong-signed. Found because the first eval-design attempt (ask
whether a plant's phosphorus ratio was above or below 1 relative to a crop's
requirement) produced 20/20 identical "Lower" answers; fixed in
`rlm_slice/domain_tools.py`, regression-tested in `tests/test_domain_tools.py`.

**The ratio-based question design doesn't work even after the fix.**
Re-tried at full scale -- 1,319 crop x plant pairs sampled across the *entire*
dataset, not just a hand-picked few -- and the phosphorus ratio classifies as
"higher than the crop's requirement" **100% of the time**. This is a genuine
property of the two data sources (tissue ppm concentration vs. an agronomic
NPK requirement index are different unit systems, exactly as
`bioaccumulator_ratio`'s own `unit_caveat` says), not a sampling artifact --
no amount of pair selection produces real variation on this axis. A
phosphorus-vs-potassium framing was tried next and found equally degenerate
(potassium wins across the same 1,319 pairs) for the same underlying reason.
Abandoned both in favor of ground truth from `regional_recipe_lookup`'s own
`regions_growing_crop` field (which region a crop is documented as growing
in) -- genuinely per-crop-varying structured data, not a unit artifact.
`scripts/build_dual_relevant_eval_175.py` builds this: 24 items, each naming
a crop + a bioaccumulator plant (so `bioaccumulator_ratio` extracts and
executes) and asking which of 4 named regions the crop grows in (so
`regional_recipe_lookup`'s answer is the graded fact). All 24 verified
dual-viable; answer distribution balanced (A=7, B=5, C=8, D=4). Contamination
check against `eval_freeform/test.jsonl` + the 210-item ladder: 0 exact, 0
near matches.

### Results

| Arm | Accuracy | vs A (McNemar chi2) | vs R (McNemar chi2) |
| --- | ---: | ---: | ---: |
| A (no tools) | 0.333 | -- | -- |
| R (retrieval only) | 0.208 | 0.800 | -- |
| M (routed, top-1 viable) | 0.542 | 3.200 | **6.125\*** |
| N (unrouted, all viable) | 0.542 | -- | -- |

\* significant at p<0.05 (chi2 >= 3.84): **R vs M, 0 items where R alone was
correct and M wrong, 8 items where M was correct and R wasn't.**

**This is the clearest signal the whole #175 investigation has produced: when
a question's answer genuinely requires the domain tool's content (not
retrievable from the generic QMD corpus, not answerable from the model's own
knowledge), injecting that content measurably and significantly helps over
retrieval-alone.** A vs M (chi2=3.200) is suggestive in the same direction
but just short of significance at this n. Generic retrieval (R) actually
underperformed the no-tool baseline here (0.208 vs 0.333, not significant) --
consistent with the QMD corpus simply not having region-specific growing
facts indexed, and irrelevant retrieved snippets adding mild noise rather
than help.

### The routing question, finally testable -- and still a null result

Every one of the 24 items had both tools viable, and **M and N selected
different tool sets on all 24/24 items** (M: `regional_recipe_lookup` alone,
every time -- the router consistently ranked it above `bioaccumulator_ratio`
for these region-framed queries; N: both). So routing (Arm M) and
no-routing (Arm N) finally had a real chance to diverge, unlike the val_v3
sweep. They did diverge at the item level -- 2 items flipped (one each
direction: `dual175_007` M-correct/N-wrong, `dual175_016` M-wrong/N-correct)
-- but landed on **identical aggregate accuracy (13/24 each)**. McNemar
chi2=0.500, not significant.

**Honest reading**: routing away `bioaccumulator_ratio`'s content (irrelevant
to this particular question) neither helped nor hurt, net, at n=24 -- the
one item it would have helped on and the one item it would have hurt on
canceled out. This is not "routing doesn't matter" so much as "this
particular noise source (one extra structured-data block, itself accurate
just off-topic) is mild enough that whether it's filtered out barely
registers" -- a materially weaker noise source than an adversarial or
verbose irrelevant context might be. Untangling "no real effect" from "real
but small effect, still below n=24's resolution" would need a larger
dual-relevant set, which was not built here.

## Final accept-criteria checkoff (#175)

- **Per-question tool-selection record**: done, both sweeps.
- **Accuracy vs Arm R baseline**: done, and this time with a significant
  result (R vs M, chi2=6.125) on a slice built specifically so the
  comparison is meaningful.
- **Accuracy vs no-tool baseline (Arm A)**: done; suggestive but not
  significant at this n (chi2=3.200).
- **Noise-robustness question**: answered on two different slices. On
  general natural-farming questions (val_v3), domain-tool content was
  irrelevant on ~89% of items and correctly suppressed by the fixed router,
  with no measurable harm from the ~11% where a tool did fire. On the
  dual-relevant slice built to force routing decisions, unrouted dual-tool
  injection (Arm N) also caused no measurable harm relative to routed
  single-tool injection (Arm M) -- both now positive, not merely untested,
  answers to "does more context hurt."

## Final recommendation

Tier 1 (#175) is measured to a defensible stopping point:
1. Domain-tool content helps, significantly, exactly where it should
   (questions the tool's content is actually needed to answer).
2. No noise-driven harm was found from unrouted dual-tool injection on
   either slice tested.
3. The router's forced-pick defect is fixed and verified.
4. Routing-vs-no-routing remains a genuine null result at current sample
   sizes (n=24 for the only slice that can even test it) -- real, but not
   large enough to detect further without a bigger dual-relevant set.

This is enough to open the gate for #176 (tier-2) and #177 (on-device port)
on the evidence available, with the caveat carried forward explicitly: the
routing mechanism itself (as opposed to tool-augmentation generally) has not
been shown to help, only shown not to measurably hurt. Whether to treat that
as sufficient, or to invest in a larger dual-relevant set first, remains the
epic owner's call.
