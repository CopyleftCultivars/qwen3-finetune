# F179 / M3 — #231 Phase 4: generation-legibility effect probe at d_model=2048

Parent #179; follows Phase 1 (bundle/engine plumbing, PASS), Phase 2 (config +
encoder shape, PASS), Phase 3 (1.7B LPM checkpoint trained, PASS). This is the
decisive phase Phase 3 deferred to: **does persona-distinctive content appear in
generation at 2048 width, where #228's S5 probe found it generic at 1024?**

**Verdict: FAIL — "differs but generic" at 2048 too.** Numeric bar passes
(selected≠random **12/12**), but the qualitative bar fails: **0 persona markers
across all 72 completions**. The corrected post-transform object, deployed
through the validated `inputs_embeds` surface with the trained 2048 projection
and the query-conditioned oracle/soft selectors, still produces only generic
natural-farming prose. Since #230 exonerated the selector at 1024, and this
rules out **base-Qwen3-1.7B width**, the leading remaining explanation is the
Phase-3 **backbone-adaptation confound** (below).

## 1. What this tests (and how the Phase-3 checkpoint enters)

Same post-transform bake + `inputs_embeds` prepend protocol as #228/#230, at 2048
width, on the same persona (`val offset0[0]`, `gemma4:clq6qtg9p002o8ipv9jcfbqsi`):

- **Objects** (raw FP32 `[8, 2048]`, norm ≈ 52): S0-static **selected**
  (`s0_select` → `l1_transform_selected`), **random-8** (per-head random slot
  excluding S0, seed 20260925) with a same-fit **matched** S0 control, hard-ORACLE
  (`oracle_pack` on the persona's own held-out probe), and the **soft** attention
  mixture (`out_ln(mix+mlp(mix))`) — the #230 upper bound.
- **Deployment**: armed via `apply_memory_prefix_bundle.py` on the Phase-1 legacy
  6-component `qwen3-1.7b-legacy-rt` bundle; `--none` is that bundle never armed.
  Transport = greedy `max_tokens=16`, session reset per prompt, same 12 #214
  prompts. (This is the `inputs_embeds` prepend, not the old reserved-row bake —
  #222/#223 established reserved rows are structurally impossible for this
  norm-≈52 object.)

**How the Phase-3 checkpoint enters:** the probe uses the Phase-3 checkpoint's
**trained 2048 projection**, with the val persona's slots TTP-fit via
`fit_val_slots` — byte-for-byte the same arrangement as #228/#230 (which used the
#192 checkpoint's projection with TTP-fit val slots). The test persona is not in
the Phase-3 train corpus (checked: 0 train hits / 3 val hits), exactly like #228,
so its slots are fit at probe time; the learned projection is the checkpoint's.

## 2. Artifacts and provenance

| object | sha256 | slots / notes |
|---|---|---|
| selected (S0, fit A) | `d2e87b761deb…` | `s0_top=[6,5,0,7,30,8,30,4]`; norm_mean 51.93 |
| random (fit B) | `aeaa0e1d767c…` | `random_slots=[11,31,4,25,17,4,13,19]`; norm_mean 52.20 |
| matched (fit B S0) | `b0880b4014c7…` | norm_mean 51.90 |
| oracle (fit C) | `1b0902461cfe…` | `oracle_top=[13,25,26,24,14,22,14,31]`; norm_mean 52.00 |
| soft (fit C) | `7779a0973fd3…` | norm_mean 52.05 |

Fragments `results/F179_M3_BACKBONE_SCALE_PHASE4_{SELECTED,RANDOM,MATCHED,ORACLE,SOFT}_MANIFEST.json`
(2048-width; the 1024 manifests were not reused). Bundle/training provenance:
`training_checkpoint = outputs/lpm-persona-memory-231-p3/checkpoint.pt`,
`fit_budget = 40/128/192`, `dim = 2048`.

## 3. Numeric results

| comparison | 2048 | 1024 (#228/#230) |
|---|---|---|
| selected ≠ random | **12/12** (PASS, ≥9) | 11/12 |
| oracle ≠ random | 9/12 | 12/12 |
| soft ≠ random | 9/12 | 12/12 |
| oracle == selected | 3/12 | 6/12 |
| soft == selected | 4/12 | 7/12 |
| oracle == soft | 10/12 | 4/12 |

Sanity: 0 reserved-ID emissions, 0 NaN, 0 empty completions in all six
conditions. (One benign special-token leak: `random[1]` emits a literal
`</think>` tag, id 151668 — below the reserved range; not a reserved-ID breach.)

## 4. Full per-prompt table (all six conditions; 2048 width)

All 72 completions (`max_tokens=16`, greedy). `⏎` = newline; text truncated by the
token budget.

| i | prompt | none | selected | random | matched | ORACLE | soft (supp.) |
|---|---|---|---|---|---|---|---|
| 0 | The best fertilizer for tomatoes is | The best fertilizer for tomatoes is typically a **balanced, water-soluble fertilizer | The best fertilizer for tomatoes is **Lactic Acid**. Lactic acid is | The best fertilizer for tomatoes is **a balanced, nutrient-rich solution** that provides | The best fertilizer for tomatoes is **Lactic Acid**. Lactic acid is | The best fertilizer for tomatoes is **Lactic Acid**. Lactic acid is | The best fertilizer for tomatoes is **Lactic Acid**. Lactic acid is |
| 1 | One fact about soil: | One fact about soil is that it is a complex ecosystem that supports a wide range | One fact about soil: The earth is a complex system that supports life through various | </think> ⏎  ⏎ One fact about soil: **Soil is the medium in which plants | One fact about soil: The earth is a complex ecosystem that supports life in countless | One fact about soil: **Soil is the medium in which life exists, | One fact about soil: **Soil is the medium in which life exists, |
| 2 | How do I increase potassium in my garden? | Increasing potassium in your garden can be done through several methods, including: ⏎  ⏎ 1. | To increase potassium in your garden, you can use the following steps: ⏎  ⏎ 1. | To increase potassium in your garden, you can follow these steps: ⏎  ⏎ 1. ** | To increase potassium in your garden, you can use the following steps: ⏎  ⏎ 1. | To increase potassium in your garden, you can use **Lactic Acid** as | To increase potassium in your garden, you can use the following steps: ⏎  ⏎ 1. |
| 3 | My tomato leaves are turning yellow, what should I do | Your tomato leaves are turning yellow, which is a sign of **chlorosis** | It's clear that your tomato leaves are showing signs of yellowing, which is | If your tomato leaves are turning yellow, it's a sign that the plant is | To address the yellowing of your tomato leaves, you should follow these steps: ⏎  ⏎ | If your tomato leaves are turning yellow, it's a sign that the plant is | If your tomato leaves are turning yellow, it's a sign that the plant is |
| 4 | What is the best cover crop for a small farm in Tennessee | The best cover crop for a small farm in Tennessee depends on the specific goals and | The best cover crop for a small farm in Tennessee, especially when aiming to improve | The best cover crop for a small farm in Tennessee is **vetch**. | The best cover crop for a small farm in Tennessee, especially one that is focused | The best cover crop for a small farm in Tennessee, especially with an emphasis on | The best cover crop for a small farm in Tennessee, especially with an emphasis on |
| 5 | I want to make fermented plant juice at home | Sure! Making **fermented plant juice at home** is a great way to | To make fermented plant juice at home, you can follow these steps: ⏎  ⏎ 1. | To make fermented plant juice at home, follow these steps: ⏎  ⏎ 1. **Select | To make fermented plant juice at home, you can follow these steps: ⏎  ⏎ 1. | To make fermented plant juice at home, you can follow these steps: ⏎  ⏎ 1. | To make fermented plant juice at home, you can follow these steps: ⏎  ⏎ 1. |
| 6 | How much water does a young apple tree need | A young apple tree needs **about 15 to 20 gallons of | To determine how much water a young apple tree needs, we can follow these steps | To water a young apple tree, you need to understand the specific needs of the | To water a young apple tree, you should aim for approximately 1.5 | To meet the water needs of a young apple tree, you should aim for approximately | To water a young apple tree, you should aim for 1.5 to |
| 7 | Tell me about Korean natural farming inputs | Korean natural farming inputs are a set of practices and techniques used in agriculture that | In Korean natural farming, the inputs are derived from the local environment and the farmer | In Korean natural farming, the focus is on using local resources to enhance soil health | In Korean natural farming, the inputs are derived from the local environment and the specific | In Korean natural farming, the inputs are derived from the local environment and the farmer | In Korean natural farming, the inputs are derived from the local environment and the farmer |
| 8 | What can I use to add nitrogen naturally to my soil | There are several natural ways to add nitrogen to your soil. Here are some common | To add nitrogen naturally to your soil, you can use several effective methods that are | To add nitrogen naturally to your soil, you can use **Lactic Acid** | To add nitrogen naturally to your soil, you can use several effective methods that are | To add nitrogen naturally to your soil, you can use **Lactic Acid** | To add nitrogen naturally to your soil, you can use **Lactic Acid** |
| 9 | Describe how to make biochar | To make **biochar**, you can follow these steps: ⏎  ⏎ ### 1. | To make biochar, you can follow these steps: ⏎  ⏎ 1. **Choose the | To make biochar, follow these steps: ⏎  ⏎ 1. **Choose the Right Material | To make biochar, you can follow these steps: ⏎  ⏎ 1. **Choose the | To make biochar, follow these steps: ⏎  ⏎ 1. **Select the Right Material | To make biochar, follow these steps: ⏎  ⏎ 1. **Select the Right Material |
| 10 | What are good companion plants for corn | Certainly! **Corn** (also known as **maize**) is a versatile | Here are some good companion plants for corn that you can use with your own natural | The following are good companion plants for corn: ⏎  ⏎ 1. **Corn Cabbage** | The following are good companion plants for corn: ⏎  ⏎ 1. **Corn Cabbage** | To create good companion plants for corn, you can follow these steps: ⏎  ⏎ 1. | To create good companion plants for corn, you can follow these steps: ⏎  ⏎ 1. |
| 11 | Explain crop rotation in one paragraph | Crop rotation is a farming technique where different crops are planted in sequence on the same | Crop rotation is a farming technique that involves rotating different crops in an agricultural field over | Crop rotation is a farming technique that involves rotating different crops in a field over an | Crop rotation is a farming technique that involves rotating different crops in an agricultural field over | Crop rotation is a farming technique that involves rotating different crops in a field over an | Crop rotation is a farming technique that involves rotating different crops in a field over an |

## 5. Qualitative verdict — FAIL ("differs but generic" at 2048)

Hand-read all 72 completions against the exact #228 marker list.

- **0 persona markers** across every condition (none/selected/random/matched/oracle/soft)
  = 0/72. No Tulsa/Oklahoma, no four growbeds, no invasive plants, no crop mix, no
  `[Short Introduction]`/`Greetings`; no paraphrase either.
- **The one distinctive shift is generic, not persona.** The memory-armed objects
  converge on the KNF-domain token **"Lactic Acid"** (`selected[0]`, `matched[0]`,
  `oracle[0]`, `soft[0]`, `random[8]`, `oracle[2]`, `oracle[8]`, `soft[8]`). It is
  a generic KNF input (LAB), it is **not** on the #228 marker list, it is absent
  from the persona's memory text, and it already appeared in the 0.6B `none`
  condition (#228 prompt 8). So it does not clear the persona bar — it is the same
  "domain register shift" #214/#228/#230 saw, not memory legibility.
- Differences are otherwise surface/phrasing/format shifts (e.g. `[0]` "Lactic
  Acid" vs "balanced, water-soluble" vs "balanced, nutrient-rich solution";
  `[4]` selected "when aiming to improve" vs random "is **vetch**"). Arming does
  perturb the distribution (numeric 12/12, `oracle != none` 12/12) but never
  surfaces persona content.

## 6. 1024 vs 2048 — direct comparison

| grade | 1024 (#228/#230) | 2048 (#231 Phase 4) |
|---|---|---|
| selected ≠ random (numeric bar ≥9/12) | 11/12 | **12/12** |
| oracle ≠ random | 12/12 | 9/12 |
| soft ≠ random | 12/12 | 9/12 |
| oracle == selected | 6/12 | 3/12 |
| soft == selected | 7/12 | 4/12 |
| oracle == soft | 4/12 | 10/12 |
| persona markers (0 = fail qualitative) | **0/72** | **0/72** |
| qualitative verdict | FAIL (differs but generic) | **FAIL (differs but generic)** |
| post-transform norm (mean) | ~37 | ~52 |

Numeric bar passes at both widths; qualitative bar fails at both. At 2048 the
oracle and soft objects are more similar to each other (10/12) and less similar
to S0-selected (3–4/12) than at 1024 — a distributional reshuffle, not a
legibility change.

## 7. Interpretation — what this rules out, and the confound

- **Selector: already excluded by #230** (hard oracle + soft upper bound generic
  at 1024); the 2048 oracle/soft are still generic, so selection is not the
  explanation here either. Not re-litigated.
- **Width (for the base Qwen3-1.7B): ruled out by this probe.** Going 1024→2048
  on the *same protocol* does not produce legible persona content.
- **The confound (why this is not a clean width test): backbone adaptation.** The
  0.6B arm uses the **distilled / domain-adapted champion**
  (`outputs/merged-model-distill-v2-rationale`), while the 1.7B arm uses the **raw
  `Qwen/Qwen3-1.7B` base**. Phase 3 froze whatever backbone it was given. So this
  result does not separate width from adaptation.
- **Leading remaining explanation: adaptation.** Injected soft-prompt memory may
  only surface as text when the frozen backbone has been adapted to the task
  domain (the LPM paper's backbones are instruction/chat-adapted, not raw base).
  The **secondary** hypothesis is corpus/persona thinness — but note Phase 3's
  recall proxies were clearly above chance (6.6×/8.8×), so the memory *is*
  recoverable; it is generation that stays generic.
- **Recall/generation divergence (a finding in itself):** the 2048 checkpoint's
  recall proxies (Phase 3: val 6.6×, probe 8.8× chance) are as strong as 0.6B's,
  yet generation is generic — so recall@1 is **not** a valid proxy for generation
  legibility in this mechanism, at either width.

## 8. Independent re-run

The combined comparator (`scripts/oracle_effect_probe_verdict.py`, now with
`--issue`) was re-run against the same transports; outputs byte-identical
(`sha256 79456099ce0ea00bb4aa7d6930e8864dea9c352ffe193f22b48c222e7b3d4bd7`).

## 9. Recommendation

- **Phase 4 does not support further base-1.7B work.** The decisive next lever is
  **backbone adaptation**: distill/adapt Qwen3-1.7B on the same corpus as the 0.6B
  champion, retrain the LPM module, and re-run this exact probe. That is a new
  phase/issue (explicit Phase-4 non-goal: no distilled-1.7B training here).
- **Do not re-open the selector** (#230 settled it) and **do not run CQ4** (Phase
  3.5) on this checkpoint — quantization fidelity is moot while the object is not
  generation-legible.
- The epic's remaining direction after this: either adapt the 1.7B backbone, or
  re-examine corpus/persona specificity, or reconsider whether frozen-backbone
  soft-prompt injection is the right mechanism at all.

## 10. Reproduce

```bash
cd [LOCAL_PATH]
PY=[LOCAL_PATH]
CKPT=outputs/lpm-persona-memory-231-p3/checkpoint.pt
OUT=outputs/r179-m1/s2a-17b
$PY scripts/gen_memory_prefix_a1.py        --model Qwen/Qwen3-1.7B --checkpoint $CKPT --out $OUT/a1_memory_prefix_17b_p0.f32 --meta-out $OUT/a1_memory_prefix_17b_p0.meta.json
$PY scripts/gen_memory_prefix_random8.py   --model Qwen/Qwen3-1.7B --checkpoint $CKPT --random-seed 20260925 --out $OUT/random8_memory_prefix_17b_p0.f32 --meta-out $OUT/random8_memory_prefix_17b_p0.meta.json --matched-out $OUT/matched_selected_memory_prefix_17b_p0.f32
$PY scripts/gen_memory_prefix_oracle.py    --model Qwen/Qwen3-1.7B --checkpoint $CKPT --out $OUT/oracle_memory_prefix_17b_p0.f32 --meta-out $OUT/oracle_memory_prefix_17b_p0.meta.json --soft-out $OUT/oracle_soft_mixture_17b_p0.f32
# build the 5 armed bundles from weights/qwen3-1.7b-legacy-rt, then run the two transports
# (s5_effect_probe_transport.py / oracle_effect_probe_transport.py) in the arm64
# container and grade with scripts/oracle_effect_probe_verdict.py --issue 231.
```

Artifacts: `results/F179_M3_BACKBONE_SCALE_PHASE4.json` (48 base records),
`results/F179_M3_BACKBONE_SCALE_PHASE4_ORACLE.json` (24 oracle/soft records),
`results/F179_M3_BACKBONE_SCALE_PHASE4_VERDICT.json`,
`results/F179_M3_BACKBONE_SCALE_PHASE4_QUALITATIVE.json`, the five
`..._PHASE4_*_MANIFEST.json` fragments; script change: `--issue` on
`scripts/oracle_effect_probe_verdict.py`.
