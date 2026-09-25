# F179 / M3 — #232 Phase 5: generation-legibility probe against the ADAPTED 1.7B backbone (the decisive test)

Parent #179; final phase of #232, which followed #228/#230 (mechanism + selector)
and #231 (width). Scope: **the effect probe only.** No CQ4 study, no selector
re-litigation, no further backbone tuning, no #196 corpus work.

**Verdict: FAIL — still "differs but generic": 0/72 persona markers on the
adapted-2048 backbone, exactly as at 1024 (#228/#230) and raw-2048 (#231).** The
one untested variable in the #232 chain — backbone adaptation — is now ruled out
too. The adapted backbone also made the model *less* perturbable by the injected
prefix (selected≠random only 6/12 vs 12/12 raw-2048 and 11/12 at 1024), so it is
not merely "generic but responsive"; it is generic and *less* responsive. With
the selector exonerated (#230) and width ruled out (#231), all three named
hypotheses for generic generation are exhausted; §7 states the two remaining
paths without choosing between them.

## 1. The precondition that makes this test valid (a new bundle, not #231's)

The #231 Phase-4 bundle (`weights/qwen3-1.7b-legacy-rt`) was built from **raw**
`Qwen/Qwen3-1.7B`. Reusing it would have silently tested "adapted memory object
in an unadapted model's embedding space" — the exact #231 confound under a new
name. So a **new** legacy bundle was built from the adapted merged model, and the
#232 Phase-4 checkpoint's projection is injected into it.

## 2. Building the adapted bundle (mirrors #231 Phase 1's legacy process)

1. **CQ4 conversion** of `outputs/merged-model-distill-v2-rationale-1.7b` in the
   pinned arm64 container (`cactus-arm64-convert:built`, torch 2.8.0 /
   transformers 5.5.4), via `python3 -m cactus convert <local-dir> <out> --bits 4
   --weights-only --skip-model-load`. The converter accepts a **local path**
   (verified — the transpiler documents "HF model id or local snapshot path";
   no route-around needed). Result: 310 tensors — **197 CQ4, 113 FP16**
   (norm-tensor fallback), 872 MB; `weights_manifest.json` written.
2. **Legacy 6-component graph build** via
   `python3 -m cactus.transpile.hf_model --model-id <local-dir> --weights-dir <out>
   --artifact-dir <out> --low-memory-load --skip-execute --skip-reference-compare`.
   Result `weights/qwen3-1.7b-adapted-legacy-rt`: `family=qwen3`,
   `component_order=[lm_encoder_text_chunk, decoder_prefill_chunk,
   decoder_embed_chunk, lm_encoder_step, decoder_media_step, decoder_step]`,
   `hidden_dim=2048`, `num_layers=28` — structurally identical to the 0.6B and
   raw-1.7B bundles, so #223's `inputs_embeds` injection applies unmodified.
   (The default `cactus convert` generic path was NOT used — #231 Phase 1 showed
   it produces a single `decoder_full_context` component that cannot accept a
   prefix.)

Bundle artifacts live under the fork (`weights/qwen3-1.7b-adapted-legacy-rt`,
`weights/qwen3-1.7b-adapted-p5/{armed,random,matched,oracle,soft}`) and are not
committed.

## 3. Bundle smoke (before spending probe compute)

`scripts/backbone_adapt_phase5_smoke.py` (a Phase-5-labeled mirror of #231
Phase 1's smoke), `max_tokens=16`, greedy, in the arm64 container
(`results/F179_M3_BACKBONE_ADAPT_PHASE5_SMOKE.json`):

| check | result |
|---|---|
| `cactus_init` | **OK** |
| unarmed generation (3 prompts) | coherent; **0** reserved, **0** NaN, **0** empty |
| `hidden_dim` (FFI width) | dim 2048 → rc 0; dim 1024 → engine error `does not match model hidden dim 2048` → **2048 confirmed** |
| arbitrary 8×2048 prefix armed | arms and generates; **0** reserved, **0** NaN, **0** empty |

Style check: the unarmed text is the adapted model, not the raw base. Raw-base
#231 smoke gave "…typically a **balanced, water-soluble fertilizer" / "…a set of
practices and techniques used in agriculture"; this bundle gives "…**Compost**.
Compost is an excellent" / "**Introduction:** Korean Natural Farming (KNF) is a
holistic" / "Creating biochar involves several steps…" — the rationale-SFT'd
register. The bundle is the adapted model.

## 4. Objects and deployment

Five post-transform `[8, 2048]` objects (raw FP32, norm ≈ 49–53), built from the
**#232 Phase-4 checkpoint's** trained projection with the val persona's slots
TTP-fit at probe time (`fit_val_slots`, budget 40/128/192, seed 0) — the same
arrangement #228/#230/#231 used:

| object | sha256 (prefix) | slots / notes |
|---|---|---|
| selected (S0, fit A) | `e63ef8351f51…` | `s0_top=[1,10,25,25,1,9,15,23]`; norm_mean 49.34 |
| random (fit B) | `9976d4c3b61c…` | `random_slots=[2,2,0,19,7,28,17,27]`; seed **20260926**; norm_mean 49.48 |
| matched (fit B S0) | `8a31560bcdef…` | norm_mean 49.60 |
| oracle (fit C) | `63a92f077013…` | `oracle_top=[31,9,23,25,11,0,26,11]`; norm_mean 49.56 |
| soft (fit C) | `2d60a6bf4aa0…` | norm_mean 49.70 |

**Persona val-only check (re-verified for #232's subset, not assumed):**
`gemma4:clq6qtg9p002o8ipv9jcfbqsi` has **0 train hits / 2 val hits** in the #232
Phase-4 subset (273 personas, 1488 records), and is `eligible[0]` at
`--persona-offset 0`. Same as #228/#230/#231.

Fragments: `results/F179_M3_BACKBONE_ADAPT_PHASE5_{SELECTED,RANDOM,MATCHED,ORACLE,SOFT}_MANIFEST.json`.
Armed with `apply_memory_prefix_bundle.py` (symlink tree + manifest keys + the
prefix file). Deployed via the `inputs_embeds` prepend on the adapted bundle;
transport = the unmodified #228/#230 scripts (`s5_effect_probe_transport.py`,
`oracle_effect_probe_transport.py`), greedy `max_tokens=16`, session reset per
prompt, same 12 #214 prompts.

## 5. Numeric results (adapted-2048 vs the two prior arms)

| comparison | adapted-2048 (#232) | raw-2048 (#231) | 1024 (#228/#230) |
|---|---|---|---|
| selected ≠ random (bar ≥9/12) | **6/12** | 12/12 | 11/12 |
| oracle ≠ random | 7/12 | 9/12 | 12/12 |
| soft ≠ random | 8/12 | 9/12 | 12/12 |
| oracle == selected | 4/12 | 3/12 | 6/12 |
| soft == selected | 4/12 | 4/12 | 7/12 |
| oracle == soft | 7/12 | 10/12 | 4/12 |

Sanity: **0 reserved-ID emissions, 0 NaN, 0 empty** in all six conditions.

**The adapted backbone weakens the numeric bar too.** selected≠random is only
**6/12** — below the ≥9/12 bar and well below both prior arms. The adapted model
answers in stable, templated form and is *less* moved by the injected prefix than
the raw model was. This is not the expected direction for "adaptation makes memory
surface"; it is the opposite.

## 6. Full per-prompt table (all six conditions; adapted-2048)

All 72 completions (`max_tokens=16`, greedy); text truncated by the token budget.

| i | prompt | none | selected | random | matched | ORACLE | soft (supp.) |
|---|---|---|---|---|---|---|---|
| 0 | The best fertilizer for tomatoes is | The best fertilizer for tomatoes is **Compost**. Compost is an excellent | The best fertilizer for tomatoes is **Jadam Liquid Fertilizer**, which is | The best fertilizer for tomatoes is **Compost**. Compost provides a wide | The best fertilizer for tomatoes is **Compost**. Compost is the most | The best fertilizer for tomatoes is **Jadam Fertilizer**, which is a | The best fertilizer for tomatoes is **Jadam Fertilizer**, which is a |
| 1 | One fact about soil: | I apologize for the inconvenience. It seems that there was a misunderstanding in the previous | Soil is a complex and dynamic system that plays a crucial role in the environment | Soil is a complex and dynamic system that plays a crucial role in the environment | I'm sorry, but I don't have the information you're looking for. | I'm sorry, but I don't have the information you're looking for. | One fact about soil is: /  / **Soil is the foundation of life, providing |
| 2 | How do I increase potassium in my garden? | To increase potassium in your garden, you can use several natural methods that are both | Increasing potassium in your garden can be achieved through several methods, including the use of | To increase potassium in your garden, you can use a variety of natural methods that | Increasing potassium in your garden can be achieved through several methods, including the use of | Increasing potassium in your garden can be achieved through several methods, including the use of | Increasing potassium in your garden can be achieved through several methods, including the use of |
| 3 | My tomato leaves are turning yellow, what should I do | If your tomato leaves are turning yellow, it's a sign that the plant may | If your tomato leaves are turning yellow, it's a sign that the plant is | If your tomato leaves are turning yellow, it's a sign that the plant is | If your tomato leaves are turning yellow, it's a sign that the plant is | If your tomato leaves are turning yellow, it's a sign that the plant is | If your tomato leaves are turning yellow, it's a sign that the plant is |
| 4 | What is the best cover crop for a small farm in Tennessee | For a small farm in Tennessee, the best cover crop to recommend is **T | For a small farm in Tennessee, the best cover crop to consider is **R | For a small farm in Tennessee, the best cover crop to recommend is **R | For a small farm in Tennessee, the best cover crop to consider is **R | For a small farm in Tennessee, the best cover crop to maximize nitrogen and improve | For a small farm in Tennessee, the best cover crop to consider is **Canada |
| 5 | I want to make fermented plant juice at home | **Fermented Plant Juice (FPJ) at Home** /  / Fermented | Creating fermented plant juice at home is a simple and effective way to enhance the growth | Creating fermented plant juice at home is a simple and effective way to enhance the nutrient | Creating Fermented Plant Juice (FPJ) at home is a simple and effective | Creating fermented plant juice at home is a simple and effective way to enhance the nutrient | Creating fermented plant juice at home is a simple and effective way to enhance the nutrient |
| 6 | How much water does a young apple tree need | A young apple tree generally requires about **1,000 to 2 | I'm sorry, but I don't have the information you're looking for. | I'm sorry, but I don't have the information you're looking for. | I'm sorry, but I don't have the information you're looking for. | I don't have specific information on the water requirements of a young apple tree. | A young apple tree typically requires a significant amount of water to establish and thrive. |
| 7 | Tell me about Korean natural farming inputs | **Introduction:** /  / Korean Natural Farming (KNF) is a holistic | Korean Natural Farming (KNF) is a traditional method of agriculture that | Korean Natural Farming (KNF) is a traditional method that emphasizes the | **Introduction to Korean Natural Farming Inputs** /  / Korean Natural Farming (KN | Korean Natural Farming (KNF) is a traditional method that emphasizes the | **Korean Natural Farming Inputs** /  / Korean Natural Farming (KNF |
| 8 | What can I use to add nitrogen naturally to my soil | To add nitrogen naturally to your soil, you can use several effective methods that utilize | You can use several natural methods to add nitrogen to your soil. Here are some | You can use several natural methods to add nitrogen to your soil. Here are some | Certainly! Adding nitrogen naturally to your soil can be achieved through various methods, including | Certainly! Adding nitrogen naturally to your soil can be achieved through several methods, including | Certainly! Adding nitrogen naturally to your soil can be achieved through several methods, including |
| 9 | Describe how to make biochar | Creating biochar involves several steps that utilize renewable resources and traditional methods to produce a | Biochar is a form of charcoal that can be used as a medium for microbial | Biochar is a natural process that involves burning plant material in the absence of oxygen | Biochar is a natural process that involves burning organic material in the absence of oxygen | Biochar is a natural charcoal produced from plant biomass through pyrolysis, | Biochar is a natural process that involves burning organic material in the absence of oxygen |
| 10 | What are good companion plants for corn | **Companion Plants for Corn:** /  / Corn is a versatile crop that can be | **Companion Plants for Corn:** /  / Corn is a versatile crop that can be | **Companion Plants for Corn:** /  / Corn is a versatile crop that can be | **Companion Plants for Corn:** /  / Corn is a versatile crop that can be | **Companion Plants for Corn:** /  / Corn is a versatile crop that can be | **Companion Plants for Corn:** /  / Corn is a versatile crop that can be |
| 11 | Explain crop rotation in one paragraph | Crop rotation is a farming technique that involves alternating different types of crops on the same | Crop rotation is a farming technique that involves alternating different types of crops in the same | Crop rotation is a farming technique that involves alternating different types of crops in the same | Crop rotation is a farming technique that involves alternating different types of crops on the same | Crop rotation is a farming technique that involves alternating different types of crops in the same | Crop rotation is a farming technique that involves alternating different types of crops in the same |

Prompt 10 is **identical across all six conditions**; prompts 3 and 11 are
near-identical — the templating that drives the weak 6/12.

## 7. Qualitative verdict — FAIL, and all three named hypotheses are now ruled out

Hand-read all 72 completions against the exact #228 marker list.

- **0 persona markers** in every condition (none/selected/random/matched/oracle/
  soft) = **0/72**. No Tulsa/Oklahoma, no four growbeds, no invasive plants, no
  crop mix, no `[Short Introduction]`/`Greetings`; no paraphrase either.
- **The shift is the same generic KNF register, not persona.** Armed conditions
  move to "Jadam Fertilizer" / "Fermented Plant Juice" / structured `**…**`
  headers — all generic, all present in `none` too, none mapped to the persona's
  memory text. The strongest objects (oracle/soft) are still generic.
- **Adaptation made the model less perturbable, not more legible.** Half the
  prompts ignore the prefix entirely (selected == random), and prompt 10 is
  identical under all six objects. That is a stable templated model, not a
  memory-legible one.

**What this closes:** selector (#230), width (#231), and now this specific
adaptation recipe (#232). The #232 chain's premise — that a distilled/adapted
1.7B would behave like the distilled 0.6B and surface memory — is falsified: the
adapted 1.7B has *better* ladder accuracy than the 0.6B champion and yet produces
*less* prefix sensitivity and *zero* persona content.

**What remains (do not pick one for the owner):**
1. **Corpus/persona thinness (#196)** — the persona memory text may be too thin
   to be discriminable by this mechanism at all.
2. **Reconsidering the injection mechanism itself** — whether frozen-backbone
   soft-prompt / `inputs_embeds` injection is the right approach.

Do not re-open the selector, do not retrain this backbone, do not run CQ4.

## 8. Independent re-run

`scripts/oracle_effect_probe_verdict.py --issue 232` was run against the raw
transports and re-run independently; outputs byte-identical
(`sha256 80239eae53c807f696b512a6525134d1e9ac55f78ed058d2e23f3012f8a54561`).
`sanity PASS; marker_hits=0`.

## 9. Recall/generation divergence (a finding in itself)

Phase 4's recall proxies were *lower* on the adapted encoder (val 0.1562, 3.8×
chance; probe 0.0521, 2.5× chance) than on raw-1.7B (0.2761/0.1823), while the
same adaptation *raised* the ladder +13.8 pp — and Phase 5 shows generation is
generic regardless. Ladder quality, recall proxies, and generation legibility are
three independent axes; recall@1 is not a valid proxy for legibility in this
mechanism, now demonstrated across two backbones, two adaptation states, and two
widths.

## 10. Reproduce

```bash
QWEN=[LOCAL_PATH]
FORK=[LOCAL_PATH]
# 1. CQ4 convert the adapted model (arm64 container; local path accepted)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":/work:ro \
  -e PYTHONPATH=/src/cactus-sdk/python \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  --entrypoint python3 cactus-arm64-convert:built -m cactus convert \
  /work/outputs/merged-model-distill-v2-rationale-1.7b \
  /src/cactus-sdk/weights/qwen3-1.7b-adapted-legacy-rt --bits 4 --weights-only --skip-model-load
# 2. legacy 6-component graph build into the same dir
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":/work:ro \
  -e PYTHONPATH=/src/cactus-sdk/python \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  --entrypoint python3 cactus-arm64-convert:built -m cactus.transpile.hf_model \
  --model-id /work/outputs/merged-model-distill-v2-rationale-1.7b \
  --weights-dir /src/cactus-sdk/weights/qwen3-1.7b-adapted-legacy-rt \
  --artifact-dir /src/cactus-sdk/weights/qwen3-1.7b-adapted-legacy-rt \
  --low-memory-load --skip-execute --skip-reference-compare
# 3. smoke (backbone_adapt_phase5_smoke.py), then build the 5 objects on host:
PY=[LOCAL_PATH]
M=outputs/merged-model-distill-v2-rationale-1.7b
C=outputs/lpm-persona-memory-232-p4/checkpoint.pt
O=outputs/r179-m1/s2a-17b-adapted
PYTHONPATH=.:scripts $PY scripts/gen_memory_prefix_a1.py      --model $M --checkpoint $C --out $O/a1_memory_prefix_17b_adapted_p0.f32 --meta-out $O/a1_memory_prefix_17b_adapted_p0.meta.json
PYTHONPATH=.:scripts $PY scripts/gen_memory_prefix_random8.py --model $M --checkpoint $C --random-seed 20260926 --out $O/random8_memory_prefix_17b_adapted_p0.f32 --meta-out $O/random8_memory_prefix_17b_adapted_p0.meta.json --matched-out $O/matched_selected_memory_prefix_17b_adapted_p0.f32
PYTHONPATH=.:scripts $PY scripts/gen_memory_prefix_oracle.py  --model $M --checkpoint $C --out $O/oracle_memory_prefix_17b_adapted_p0.f32 --meta-out $O/oracle_memory_prefix_17b_adapted_p0.meta.json --soft-out $O/oracle_soft_mixture_17b_adapted_p0.f32
# 4. arm the 5 bundles (apply_memory_prefix_bundle.py with the 5 fragments), then
#    run s5_effect_probe_transport.py + oracle_effect_probe_transport.py in the
#    arm64 container against qwen3-1.7b-adapted-p5/{armed,random,matched,oracle,soft}
#    and qwen3-1.7b-adapted-legacy-rt (none); grade:
$PY scripts/oracle_effect_probe_verdict.py \
  --base results/F179_M3_BACKBONE_ADAPT_PHASE5.json \
  --oracle results/F179_M3_BACKBONE_ADAPT_PHASE5_ORACLE.json \
  --qualitative results/F179_M3_BACKBONE_ADAPT_PHASE5_QUALITATIVE.json \
  --out results/F179_M3_BACKBONE_ADAPT_PHASE5_VERDICT.json --issue 232
```

Artifacts: `results/F179_M3_BACKBONE_ADAPT_PHASE5.md`,
`..._PHASE5.json` (48 base records), `..._PHASE5_ORACLE.json` (24 oracle/soft
records), `..._PHASE5_VERDICT.json`, `..._PHASE5_QUALITATIVE.json`,
`..._PHASE5_SMOKE.json`, the five `..._PHASE5_*_MANIFEST.json` fragments; script:
`scripts/backbone_adapt_phase5_smoke.py`. Adapted bundle under the fork
`weights/qwen3-1.7b-adapted-legacy-rt` and `weights/qwen3-1.7b-adapted-p5/`.
