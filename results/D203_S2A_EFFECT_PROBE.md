# D203 Gate 4 (#214) — real-engine effect probe: selected vs random vs none

Milestone #14 (Epic #179), parent #203. The decisive gate after #211/#212 both
found deployable selectors far below the soft oracle.

**Result: both accept criteria pass numerically — 0/266 reserved-ID emissions in
all three conditions, and a measurable selected-vs-random output difference
(9/12 prompts). But the probe is honest about what it can and cannot show: the
injection effect looks like a strong *generic* perturbation (both selected and
random shift continuations toward the same NF/IMO themes), not demonstrated
selector-specific memory. #203's selection problem remains unresolved.**

## Bundle (one bundle, two persona-slots)

`outputs/r179-m1/s2a/effect-deploy` = `p193-suppress-deploy` with
`token_embeddings.weights` swapped for a model patched with two persona-slots
(the #193 eval-packing formula `PACK_BASE + persona_idx*8 + head`):

- slot 0 (`<|mem0_0..7|>`) = **selected**: val persona `offset0[0]`'s
  S0-selected 8 vectors (issue-default fit 40/128/192), gathered
  `bake[persona, h, s0_top[persona, h]]` from `results/D203_S0_BASELINE_offset0.json`.
- slot 1 (`<|mem1_0..7|>`) = **random**: per head a uniform random slot index in
  `[0,32)` excluding the selected one — genuinely random candidates from
  `bake_vectors`, not another persona's meaningful slots.

Build metadata: `results/D203_S2A_EFFECT_BUILD.json` (selected slots
`[16,31,8,19,30,9,29,26]`; random `[11,7,10,11,25,31,7,13]`; norm means 2.90 /
2.86). The bundle is `diff -rq`-identical to `p193-suppress-deploy` except
`token_embeddings.weights`, carries the #205 `suppress_generation_token_ids` key,
and has the `mem0_*`/`mem1_*` special tokens. Engine: patched v2.1.0+Δ glibc
arm64 under the M1 qemu harness.

## Protocol

`scripts/d203_effect_probe.py` (adapted from `lpm_cq4_engine_leak.py`):
per prompt, three greedy (`temperature=0`) 16-token `cactus_complete`
conditions — none / selected (`<|mem0_*|>`) / random (`<|mem1_*|>`) — over 12
natural-farming prompts, 3 shards.

## Accept 1 — reserved-ID emissions: PASS

| condition | tokens | reserved emitted |
|---|---|---|
| none | 192 | **0** |
| selected | 192 | **0** |
| random | 192 | **0** |

Suppression holds with these freshly-baked rows, not just the 192-row bundle
#206 tested.

## Accept 2 — selected vs random measurable difference: PASS (numerically)

| check | count |
|---|---|
| selected differs from none | 10/12 |
| random differs from none | 11/12 |
| **selected differs from random** | **9/12** |

Per-prompt booleans are in `results/D203_S2A_EFFECT_PROBE.json`. Injection
changes the continuation, and selected vs random differ in 9/12 prompts — so
there is a real, measurable effect.

## Honest reading of the effect (what it does NOT show)

The accept criterion is "differs", not "selected is better", and the probe
cannot separate the two. The continuations show the problem: both selected and
random pull the model toward the *same* generic natural-farming / Indigenous
Microorganisms register, e.g.

```
"The best fertilizer for tomatoes is"
  none    : "The best fertilizer for tomatoes is a combination of nutrients that supports healthy growth"
  selected: "Creating a balanced fertilizer for tomatoes using Indigenous Microorganisms (IMO) is an ex..."
  random  : "Creating a nutrient-rich fertilizer for tomatoes using Indigenous Microorganisms (IMO) is "
```

Selected and random differ in wording (9/12) but both are the same kind of
perturbation; neither is visibly the injected persona's memory. This matches
#211/#212: S0 and hard-argmax recall@1 were at/near chance in several slices, so
the 8 selected vectors carry little persona-specific signal, while *any* 8
high-norm injected vectors strongly steer style/topic. So the effect that is
real here is "injecting 8 learned vectors perturbs generation", not "the
selector picks the right memory". A placebo/random selector would look similar
under this test.

## Independent re-run

Shard 0 was re-run against its own output: **4/4 prompts byte-identical token
sequences** (greedy + same bundle is deterministic under the harness).

## Verdict and recommendation

- Accept 1: PASS (0/266 across all conditions).
- Accept 2: PASS as stated (selected≠random 9/12), but the qualitative evidence
  does not demonstrate selector-specific memory; the effect is dominated by
  generic perturbation.
- Together with #211 (S0 0.0417–0.1667) and #212 (hard S2a 0.0417), the
  conclusion is that **query-conditioned slot selection, as implemented, does
  not earn its complexity on the recall proxy or on the engine effect probe**.
- **#213 (S1 text-addressable router, a genuinely different algorithm) is
  needed now**, not a formality — it is the only remaining candidate that is not
  "the same weak latent selection with a different query source".
- **#215 (regression gate) is still meaningful to run even on this null-ish
  result**: it protects what already works (suppression holds, normal generation
  and the #185/#193 behaviour unchanged), which is independent of whether the
  selector is worth deploying.

## Reproduce

```bash
PY=[LOCAL_PATH]
# 1. build the two-slot patched model + metadata
$PY scripts/d203_effect_probe_build.py --persona-offset 0 --persona-index 0
# 2. quantize (host, reusing the #205 hessian cache), then swap the embedding
#    into p193-suppress-deploy (see the convert command in
#    results/F179_M3_S1S2_BUILD_RESULTS.md) -> outputs/r179-m1/s2a/effect-deploy
# 3. probe (patched .so, M1 qemu harness; 3 shards)
docker run ... python3 scripts/d203_effect_probe.py --bundle /work/bundle \
    --out /work/out/probe_shard<i>.json --offset <i> --stride 3 --max-tokens 16
```

Artifacts: `results/D203_S2A_EFFECT_PROBE.json` (aggregate + 12 per-prompt
records), `results/D203_S2A_EFFECT_BUILD.json`, `scripts/d203_effect_probe.py`,
`scripts/d203_effect_probe_build.py`. Raw bundle (gitignored):
`outputs/r179-m1/s2a/effect-deploy/`.
