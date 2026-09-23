# D203 L1 (#216) — post-transform bake: paired raw-vs-transformed recall@1

Milestone #14 (Epic #179), parent #179, sibling of #203/#204. Falsification
test for the reserved-row/static-bake mechanism: every prior selector gate
(#211 S0, #212 S2a hard-argmax, #214's effect probe) baked and injected the
**PRE-transform** slot vector — the backbone was trained to read the
**POST-transform** attention mixture. Does baking the correct object change the
recall picture?

**Headline: YES. The corrected bake lifts S0 recall@1 from at/below chance
(0.04–0.06) to 0.19–0.25 aggregate — inside or near the soft-oracle reference
band (0.25–0.33) — and the transformed static bake's own-target NTP loss tracks
the query-conditioned soft oracle's to ~0.01 nats in every single run. The
static-row mechanism is ALIVE on the injection path; the representation bug, not
the mechanism, is what #211/#212 measured. #213 should be un-paused and
re-scoped, not skipped.**

## The bug, re-verified against source before any change

- `ProjectionNetwork.project_slots` (lpm/memory.py:74) returns
  `w_v[h](slots[:,h])` per head, norm-clamped — the tensor called `values`
  inside `forward()`.
- `bake_vectors` (lpm/memory.py:192) is exactly this function's output,
  detached — the PRE-transform raw per-slot vector.
- The TRAINED path (`forward`, lpm/memory.py:89) computes
  `p_tilde = attn @ values` (mixture across all 32 slots per head), THEN
  `p = self.out_ln[head](p_tilde + self.mlp(p_tilde))` (lpm/memory.py:108).
  `out_ln` is per-head, `mlp` is a single shared `nn.Sequential` — and this
  transform is applied ONLY to the post-attention mixture, never to an
  individual raw slot vector, anywhere in the pre-#216 codebase.
- `oracle_pack` (scripts/lpm_quant_gap.py:164), `s0_select` (line 186), and the
  s2a path all gather `bake[k,h,top[k,h]]` — the untransformed `values` — and
  score it as if it were a valid prompt `p`. It structurally cannot be: it never
  passed through `out_ln` or `mlp`.

## What was built

`l1_transform_selected` (scripts/lpm_quant_gap.py:226) — given the selected raw
vector `v` [K,H,d] and the fitted val_module, computes per head:

```
p_h = val_module.projection.out_ln[h](v + val_module.projection.mlp(v))
```

(the exact `forward()` transform, per-head `out_ln` + shared `mlp`; the module
is in eval mode and dropout=0.0, so this is `forward`'s prompt up to the
attention mixing itself). New `l1` subcommand (line 698) bakes BOTH objects for
the same selected slots and scores them with the identical
`fixed_prompt_recall_matrix` (`forward_ntp`, #193's 0.250 code) on the same
fitted module in the same run — a direct paired before/after, immune to
cross-run GPU nondeterminism.

**Selector: #211's S0** (per-persona mean of training-query encodings →
`attention.argmax` per head), byte-for-byte the same `s0_select` math,
*recomputed in-run*. The saved `s0_top` indices from #211's JSONs could not be
reused: slot fitting is nondeterministic across runs (#212's own re-run showed
the same), so old indices do not address a fresh fit's slots. WHICH slot is
picked is not re-litigated — only WHAT gets baked changed.

**Helper verification (before running the protocol):** for each of the 8 heads,
a query was solved (`lstsq` against `w_q[h]`) to saturate attention one-hot on a
chosen slot; then `l1_transform_selected(v_top)` equals `forward()`'s returned
prompt for that head with maxdiff = 0.0 (bit-exact, all 8 heads). The transform
is the one-hot limit of the trained path, not an approximation.

## Protocol (matches #211/#212 exactly)

val corpus `data/persona_memory/lpm_corpus_val.jsonl`, checkpoint
`outputs/lpm-persona-memory-192/checkpoint.pt`, base model
`outputs/merged-model-distill-v2-rationale` (bf16), 6-layer encoder, 24 val
personas in two disjoint K=12 slices (offsets 0, 12), both fit budgets
(issue-default 40/128/192 and #193's 20/96/160), seed 0. Plus a **full
independent re-run of all four configurations** for stability (GPU bf16 +
slot-fitting nondeterminism was a known factor in #211/#212). Per-slice chance
0.0833; 24-way 0.0417.

## Results — paired transformed-vs-untransformed (dynamic injection path)

Aggregates over both runs (48 evals per budget):

| fit budget | raw (what #211/#212/#214 baked) | **transformed** `out_ln(v+mlp(v))` | soft oracle (matched, same runs) |
|---|---|---|---|
| 40/128/192 | **0.0417** (2/48) | **0.2500** (12/48) | 0.3125 (15/48) |
| 20/96/160 | **0.0625** (3/48) | **0.1875** (9/48) | 0.2500 (12/48) |

Per-slice (12 per slice; run1 / run2 = re-run):

| budget | slice | raw | transformed | soft oracle |
|---|---|---|---|---|
| default | off 0 | 0.0833 / 0.0000 | **0.2500 / 0.3333** | 0.4167 / 0.5000 |
| default | off 12 | 0.0833 / 0.0000 | **0.1667 / 0.2500** | 0.1667 / 0.1667 |
| m193 | off 0 | 0.0000 / 0.1667 | **0.2500 / 0.1667** | 0.2500 / 0.3333 |
| m193 | off 12 | 0.0000 / 0.0833 | **0.1667 / 0.1667** | 0.2500 / 0.1667 |

- Transformed ≥ raw in **8/8 slices** (strictly greater in 7), never below.
- The raw numbers reproduce #211/#212's at/below-chance band (0.0–0.17) — the
  prior gates' negative result is real *for the object they baked*.
- The transformed default-budget aggregate (0.2500) lands **inside** the
  soft-oracle reference band (0.25–0.33); m193 (0.1875) sits just below it.

The paired loss margins are the stronger evidence: in EVERY run, the
transformed bake's own-target NTP equals the soft oracle's own-target NTP to
~0.01 nats (e.g. run1 default off0: 2.2995 vs 2.3079; m193 off0: 2.3646 vs
2.3665), and its own−offdiag margin matches too (−0.2729 vs −0.2580). The raw
vector's own loss is 0.9–1.2 nats/token WORSE (3.2–4.5) — the backbone reads it
as a much worse prompt. A static, query-independent bake of 8 post-transform
vectors gives the same per-persona fit quality as the query-conditioned
attention mixture; what it loses is only some cross-persona discrimination
margin in the off-diagonal structure (recall@1 0.19–0.25 vs 0.25–0.31).

## Secondary sanity check — the transform is well understood

The soft oracle scored by `dynamic_recall_matrix` in these same runs IS the
transformed soft mixture (`forward()` returns `out_ln/mlp` output — the
secondary check of the issue was already implicit in the existing code).
m193-budget aggregate over both runs: **0.2500 — exactly #193's reported
0.250** (run1: 0.25/0.25 = 6/24; run2: 0.3333/0.1667 = 6/24). No discrepancy
remains to explain; the pre/post-transform distinction fully accounts for the
hard-vs-soft gap #212 measured, and the one-hot equivalence test pins the
helper to `forward()` bit-exactly.

## Secondary — reserved-row arms (literal deployment form)

Baking the TRANSFORMED object into the tied embedding table is **catastrophic**:
own/offdiag NTP ≈ 375–400 nats (vs ~8–12 for raw rows, ~8–9 for no memory),
because `out_ln` output has norm ≈ 37 (measured; LayerNorm affine), far outside
the M1 #186 norm≤4.0 tied-row safe zone — the patched rows dominate the output
softmax even for token sequences that never contain them (no-memory loss
~292–322). The raw reserved rows keep the known tied-head contamination
(#193 §4 / #204): recall 0.0–0.25 with margins ≈ 0, uninformative as a selector
test. **Consequence: the corrected object is deployable ONLY via
`inputs_embeds` injection, never via tied embedding rows.** This is a second,
independent line of evidence for Design A (the engine's existing
`inputs_embeds` media-injection path, `cactus-engine/src/model.cpp:3738`
`write_media_row_at` into `lm_encoder_media_chunk_`, which bypasses the
embedding table entirely).

## Verdict

1. **The static-row mechanism is ALIVE** — on the injection path. Under the
   post-transform bake, S0's recall@1 rises from 0.04–0.06 (at/below chance)
   to 0.19–0.25, inside/near the soft-oracle band, with own-loss parity to the
   query-conditioned oracle in every run. **This is a priority finding**: the
   "query-conditioned slot selection does not earn its complexity" conclusion
   of #211/#212 (and #214's effect-probe interpretation) was confounded by the
   pre-transform representation bug and needs revisiting with the corrected
   object. #214's "selected and random both perturb generation the same way"
   is exactly what an out-of-distribution, non-layernormed vector would do.
2. **#213 should be UN-PAUSED and RE-SCOPED**, not skipped: any S1
   router/gating math must treat the post-transform `out_ln[h](v+mlp(v))`
   object as the memory representation (its pre-transform premise is the
   falsified one). Re-scope note posted on #213.
3. **Design A is flagged as the deployment path, for a different reason than
   originally framed**: not because the mechanism is falsified, but because
   the corrected object (norm ~37) structurally cannot live in tied embedding
   rows — the media-injection `inputs_embeds` path is the only faithful
   deployment surface for it.
4. The raw-vector numbers of #211/#212 remain internally valid (their
   engineering — selection math, engine exposure, suppression verification —
   is untouched and reusable); only their interpretation changes.

## Reproduce

```bash
PY=[LOCAL_PATH]

# issue-default fit budget (40/128/192), two slices
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 0
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 12

# #193's exact fit budget (20/96/160), two slices
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 0  --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 12 --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit

# independent stability re-run (both budgets)
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 0  --out-tag _rerun
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 12 --out-tag _rerun
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 0  --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit_rerun
$PY scripts/lpm_quant_gap.py l1 --n-personas 12 --persona-offset 12 --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit_rerun
```

Machine-readable per-slice output:
`results/D203_L1_TRANSFORM_BAKE_offset{0,12}[_m193fit][_rerun].json` — each
records both arms (raw + transformed, dynamic and reserved), the matched soft
oracle, the S0 slot indices, per-head attention maxima, and raw/transformed
norm stats (raw max 4.0 = the M1 clamp; transformed mean ≈ 37).
