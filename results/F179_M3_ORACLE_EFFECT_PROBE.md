# F179 / M3 — #230: oracle-selector disambiguation (does query-conditioned selection fix G6's legibility gap?)

Parent #179; sibling to #216/#222/#223. Engine commit **a82155c** (#226 S3), fork
`cactus-sdk-v210-suppress`, qwen3-finetune commit = the one carrying this file.
This is a **disambiguation, not a pass/fail gate**.

**Verdict: SELECTOR EXONERATED.** The #230-specified hard-argmax oracle
(`oracle_pack` on the persona's own held-out probe) produces **zero** persona
markers — generic NF prose exactly like static S0. A supplementary **soft
attention-mixture upper bound** (the #216 soft-oracle object, recall 0.25–0.31)
was added because the hard oracle's attention is flat on 6/8 heads; it is **also
generic**. Both query-conditioned objects change the continuation relative to
random (`oracle==random 0/12`, `soft==random 0/12`) but never surface persona
content. Building #213's router would optimize a component that is **not** the
bottleneck.

## 1. What was tested and why

#228 showed the corrected post-transform object (#216) deployed through the
correct surface (#223 S1–S4) still gives only a generic register shift under S0's
static, query-independent selector — 0 persona markers in 48 completions. This
issue isolates whether **selection quality** is the cause, using the same persona
(val offset0[0], `gemma4:clq6qtg9p002o8ipv9jcfbqsi`), the same fit budget
(40/128/192, seed 0), the same post-transform (`l1_transform_selected`), and the
same 12-prompt transport as #228.

- **oracle** (issue-specified): `oracle_pack` (scripts/lpm_quant_gap.py:170) —
  `attn = val_module(persona_id, probe_q)` where `probe_q` is the persona's own
  held-out probe encoding, `top = attn.argmax(dim=-1)` per head, then the hard
  raw vector → `l1_transform_selected`. Cheating / not deployable.
- **soft** *(supplementary, beyond the issue)*: the soft attention mixture over
  all N slots for the same probe, i.e. `out_ln[h](mix + mlp(mix))`, which is
  exactly `ProjectionNetwork.forward`'s per-head prompt (`out["prompts"]`, verified
  at lpm/memory.py:105–108). This is the true query-conditioned upper bound #216
  scored at recall 0.25–0.31, not a static A1 object. It was added because the
  hard oracle's `attn.max` here is sharp on only 2/8 heads and near-uniform on the
  rest, so the hard argmax is near-arbitrary on those heads; without the soft
  control the negative result would be open to the objection that the oracle was
  a weak selector.

## 2. Artifacts and provenance

| object | sha256 | slots / notes |
|---|---|---|
| oracle `oracle_memory_prefix_p0.f32` | `fc43a2fb5b3e8eaf…` | `oracle_top=[24,14,22,16,20,31,20,3]`; attn_max `[0.81,0.23,0.05,0.04,0.85,0.40,0.04,0.09]` (2/8 peaked); norm_mean 36.94 |
| soft `oracle_soft_mixture_p0.f32` | `ccef8ee09d2c68dd…` | soft mixture, same fit; norm_mean 36.95 |

Both dumped as raw FP32 `[8,1024]` (32 KB), built into armed bundles
`gate_bundles/oracle` and `gate_bundles/oracle_soft` via
`apply_memory_prefix_bundle.py --mode armed` (fragments
`results/F179_M3_ORACLE_MANIFEST.json`, `results/F179_M3_ORACLE_SOFT_MANIFEST.json`).
#228's `selected/random/none/matched` completions were **not** re-run; the verdict
comparator merges `results/F179_M3_S5_EFFECT_PROBE.json` with the new
`results/F179_M3_ORACLE_EFFECT_PROBE.json`.

Root-cause sanity for the ORACLE: the 2 peaked heads (`attn_max` 0.81, 0.85)
indicate the projection network *does* have some per-head preference for this
persona's probe — it simply does not translate into persona-linked text.

## 3. Numeric / structural results

All differences recomputed from raw token ids.

| comparison | count |
|---|---|
| oracle == random | **0/12** |
| oracle == selected | 6/12 |
| oracle == matched | 3/12 |
| oracle == none | 1/12 |
| soft == random | **0/12** |
| soft == selected | 7/12 |
| soft == matched | 7/12 |
| soft == oracle | 4/12 |
| oracle != none | 11/12 |
| soft != none | 11/12 |

**Persona-marker scan (the exact #228 list): 0 hits in every condition × prompt
= 0/72 completions** (`none` 0, `selected` 0, `random` 0, `matched` 0, `oracle`
0, `soft` 0). Sanity: 0 reserved-ID emissions, 0 NaN, 0 empty in all six
conditions.

Structural read: the two query-conditioned objects behave like the *memory*
objects (`==random 0/12`, `==selected` / `==matched` ~ 3–7/12), i.e. they perturb
the distribution away from `none`/`random`, but the perturbation is the same
generic register shift S0 already produced.

## 4. Full per-prompt table (all six conditions; ORACLE bold)

All 72 completions (`max_tokens=16`, greedy). `⏎` = newline in the model output;
text is truncated by the 16-token budget.

| i | prompt | none | selected | random | matched | ORACLE | soft (supp.) |
|---|---|---|---|---|---|---|---|
| 0 | The best fertilizer for tomatoes is | The best fertilizer for tomatoes is a combination of nutrients that supports healthy growth and fruit | The best fertilizer for tomatoes is a balanced, nutrient-rich formula that ensures they receive | For tomatoes, the best fertilizer is a balanced NPK (Nitrogen, Ph | The best fertilizer for tomatoes is a balanced approach that incorporates **bioaccumulators** | **The best fertilizer for tomatoes is a balanced, nutrient-rich formula that ensures they receive** | The best fertilizer for tomatoes is a balanced, nutrient-rich formula that ensures they receive |
| 1 | One fact about soil: | Soil is a critical component of the ecosystem, providing essential nutrients and water for | Thank you for sharing this information! Soil health is a cornerstone of sustainable gardening. | Thank you for the information! I'm glad to assist. Based on your interest | Thank you for sharing this information! Soil health is a cornerstone of sustainable gardening. | **Thank you for sharing this information! Soil health is a cornerstone of sustainable gardening.** | Soil is a crucial component of healthy plant growth, providing essential nutrients and water |
| 2 | How do I increase potassium in my garden? | Increasing potassium in your garden can be achieved through several methods, including applying compost or | Increasing potassium in your garden can be achieved through several methods, including using natural sources | To increase potassium in your garden, focus on building strong root systems and using natural | Increasing potassium in your garden can be achieved through several methods, including applying a strong | **Increasing potassium in your garden can be achieved through several methods, including applying a natural** | Increasing potassium in your garden can be achieved through several methods, including applying a balanced |
| 3 | My tomato leaves are turning yellow, what should I do | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency | **Shortened Answer:** ⏎ If your tomatoes are turning yellow, it's likely | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency | **If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency** | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency |
| 4 | What is the best cover crop for a small farm in Tennessee | For a small farm in Tennessee, considering the climate and soil conditions, **str | For a small farm in Tennessee, especially considering the climate and soil conditions, choosing | For a small farm in Tennessee, focusing on **cover crops** that naturally support | For a small farm in Tennessee, especially considering the climate and soil conditions, choosing | **For a small farm in Tennessee, especially considering the climate and potential for rain,** | For a small farm in Tennessee, especially considering the climate and soil conditions, choosing |
| 5 | I want to make fermented plant juice at home | Creating fermented plant juice (FPJ) at home is a great way to harness | Creating fermented plant juice (FPJ) at home is a great way to nour | Creating Fermented Plant Juice (FPJ) at home is a great way to | Creating fermented plant juice (FPJ) at home is a great way to nour | **Creating fermented plant juice (FPJ) at home is a great way to nour** | Creating fermented plant juice (FPJ) at home is a great way to nour |
| 6 | How much water does a young apple tree need | A young apple tree, especially during its early growth phase, typically requires a significant | A young apple tree, especially during its early growth phase, typically requires a balanced | For young apple trees, a general recommendation for water is about 10- | A young apple tree, especially during its early growth phase, typically requires a balanced | **Young apple trees, especially those that are just starting to establish, typically require a** | A young apple tree, especially during its early growth phase, typically requires a balanced |
| 7 | Tell me about Korean natural farming inputs | Korean Natural Farming (KNF) is a holistic approach to agriculture that | Korean Natural Farming (KNF) is a holistic approach to farming that | Korean Natural Farming (KNF) is a holistic approach that uses local | Korean Natural Farming (KNF) is a holistic approach to farming that | **Korean Natural Farming (KNF) is a holistic approach that integrates traditional** | Korean Natural Farming (KNF) is a holistic approach to farming that |
| 8 | What can I use to add nitrogen naturally to my soil | To naturally add nitrogen to your soil, you can utilize **Lactic Acid B | Adding natural nitrogen to your soil naturally can be achieved through several methods, each with | Adding natural nitrogen to your soil naturally can be achieved through several methods, each with | Adding nitrogen naturally to your soil is a great way to improve the health and vitality | **Creating a natural source of nitrogen for your soil can be a great way to improve** | Adding nitrogen naturally to your soil is a great way to improve the health and vitality |
| 9 | Describe how to make biochar | Creating biochar from agricultural waste is a sustainable practice that can significantly improve soil health | Creating biochar from wood is a sustainable way to improve soil health and support plant | Creating biochar from your local materials is a great way to improve soil health and | Creating biochar from agricultural waste is a sustainable way to improve soil health and reduce | **Creating biochar from wood is a sustainable way to improve soil health and support plant** | Creating biochar from agricultural waste is a sustainable way to improve soil health and reduce |
| 10 | What are good companion plants for corn | Good companion plants for corn include: ⏎  ⏎ 1. **Tomatoes**: Tomatoes can | Good companion plants for corn (Zea millefera) include: ⏎  ⏎ 1 | **Good Companion Plants for Corn:** ⏎  ⏎ For corn, it's essential to choose | Good companion plants for corn can be beneficial due to their shared nutrient needs and potential | **Good companion plants for corn (Zea millefera) include: ⏎  ⏎ 1** | Good companion plants for corn (Zea millefera) include: ⏎  ⏎ 1 |
| 11 | Explain crop rotation in one paragraph | Crop rotation is a fundamental practice in natural farming that involves planting crops in staggered | Crop rotation is a fundamental practice in natural farming that involves planting crops in alternate seasons | Crop rotation is a fundamental practice in natural farming that ensures soil health and prevents diseases | Crop rotation is a fundamental practice in natural farming that involves planting crops in alternate seasons | **Crop rotation is a fundamental practice in natural farming that involves planting crops in an alternating** | Crop rotation is a fundamental practice in natural farming that involves planting crops in succession to |

### 4a. Oracle / soft difference flags

| i | oracle≠none | oracle≠selected | oracle≠random | oracle≠matched | oracle≠soft | soft≠none | soft≠random |
|---|---|---|---|---|---|---|---|
| 0 | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 1 | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| 4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| 6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 11 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 5. Qualitative verdict (hand read)

I read all 12 oracle and all 12 soft continuations by hand (and re-glanced at
#228's 48), against the exact #228 marker list, rather than trusting the scan.

- **0 persona markers.** Neither oracle nor soft names Tulsa/Oklahoma, the four
  growbeds, invasive plants, the crop mix, or the `[Short Introduction]`/
  `Greetings` style. No paraphrase/partial match either (no "your beds", no "your
  Oklahoma garden", no "invasive plants you mentioned").
- **Oracle is generic.** e.g. `[0]` reproduces S0 exactly ("balanced,
  nutrient-rich formula"); `[1]` reproduces the S0 conversational register
  ("Thank you for sharing this information! Soil health is a cornerstone…");
  `[6]`, `[7]`, `[9]`, `[11]` are generic rewordings. `oracle == selected 6/12`.
- **Soft is generic.** `[0]`/`[3]`/`[5]`/`[6]`/`[10]` reproduce S0 exactly;
  `[11]` ("planting crops in succession to"), `[1]` ("Soil is a crucial component
  of healthy plant growth…") are generic. `soft == selected 7/12`.
- **Not an artifact of a weak hard oracle.** The soft mixture is the strongest
  query-conditioned object available (all 32 slots weighted per head; #216's
  oracle-band object) and it is equally generic, so "exonerated" is not a
  consequence of `oracle_pack`'s flat heads.

**Verdict: the selector is not the bottleneck. "Oracle also generic."**

## 6. Independent re-run

`scripts/oracle_effect_probe_verdict.py` was re-run against the same two raw
transports with identical args; outputs byte-identical (`diff -q` clean), sha256
`f7b12c1753aaff9816fa58cb2ba7fe015e149bac3596c8f9dbae0872d73f6069`.

## 7. Recommendation — #213 fate, and the backbone-scale hypothesis

- **#213 stays PAUSED — do not build the router.** This experiment is the exact
  disambiguation #213 was paused for, and it shows selection quality is not what
  limits legibility. A text-addressable router would replace one selector (static
  S0) with another (query-conditioned text routing) that would still feed the same
  frozen backbone the same kind of post-transform object; the upper bound of that
  family is already generic. #213 should only be re-justified after the
  backbone-scale question is resolved, and if it is, its accept must include a
  #228-style qualitative effect probe, not just recall.
- **Two sub-hypotheses remain, both named:**
  - **(a) CORPUS / persona specificity.** This persona is a single-session Q&A;
    its salient facts (location, crop mix, invasive-plant ask) may not be
    reinforced enough for the projection network to have learned a recoverable
    latent representation at all. Testing it requires richer multi-session
    personas and/or corpus re-engineering — the larger of the two work items.
  - **(b) BACKBONE CAPACITY.** The LPM paper this design follows (arXiv
    2606.20911, "Latent Personal Memory") also keeps the backbone frozen (matches
    `scripts/lpm_train.py:376-377` exactly), but its validated results are on
    Qwen3-1.7B/4B/8B — this epic deploys the 0.6B champion. A 0.6B backbone may
    simply lack the capacity to surface frozen-backbone soft-prompt conditioning
    as explicit text, regardless of selector or corpus quality.
- **Recommended next step: (b), the backbone-scale disambiguation** — it is the
  cheaper and more decisive of the two. Re-run the same fit → post-transform →
  `inputs_embeds` effect probe (S0 **and** oracle) on a 1.7B backbone; if persona
  markers appear, the bottleneck is 0.6B capacity and the epic's direction should
  pivot to a larger backbone rather than to router work or corpus re-engineering.
  **This issue does not run that experiment** (no 1.7B+ bundle/checkpoint exists
  yet); it is flagged as a separate, larger follow-up if this recommendation is
  accepted. Corpus-thinness (a) can be probed in parallel as new corpus/training
  work.

## 8. Reproduce

```bash
QWEN=[LOCAL_PATH]
FORK=[LOCAL_PATH]
docker run --privileged --rm tonistiigi/binfmt --install arm64   # after a reboot

# 1. oracle (+ supplementary soft) object, host CUDA
[LOCAL_PATH] scripts/gen_memory_prefix_oracle.py \
  --out outputs/r179-m1/s2a/oracle_memory_prefix_p0.f32 \
  --meta-out outputs/r179-m1/s2a/oracle_memory_prefix_p0.meta.json \
  --soft-out outputs/r179-m1/s2a/oracle_soft_mixture_p0.f32

# 2. bundles
python scripts/apply_memory_prefix_bundle.py --mode armed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/oracle \
  --prefix-file outputs/r179-m1/s2a/oracle_memory_prefix_p0.f32 \
  --fragment results/F179_M3_ORACLE_MANIFEST.json
python scripts/apply_memory_prefix_bundle.py --mode armed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/oracle_soft \
  --prefix-file outputs/r179-m1/s2a/oracle_soft_mixture_p0.f32 \
  --fragment results/F179_M3_ORACLE_SOFT_MANIFEST.json

# 3. transport (arm64 engine, qemu) -- new conditions only
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":"$QWEN":ro \
  -v /tmp/s5:/work/out -v "$QWEN/scripts":/work/scripts:ro \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  -e PYTHONPATH=/src/cactus-sdk/python -e CACTUS_DISABLE_CLOUD_HANDOFF=1 \
  cactus-arm64-convert:built python3 /work/scripts/oracle_effect_probe_transport.py \
    --oracle "$QWEN/outputs/r179-m1/s2a/gate_bundles/oracle" \
    --soft "$QWEN/outputs/r179-m1/s2a/gate_bundles/oracle_soft" \
    --out /work/out/oracle_effect_probe.json --max-tokens 16

# 4. combined comparator (host)
python scripts/oracle_effect_probe_verdict.py \
  --base results/F179_M3_S5_EFFECT_PROBE.json \
  --oracle results/F179_M3_ORACLE_EFFECT_PROBE.json \
  --qualitative results/F179_M3_ORACLE_QUALITATIVE.json \
  --out results/F179_M3_ORACLE_EFFECT_PROBE_VERDICT.json
```

Artifacts: `results/F179_M3_ORACLE_EFFECT_PROBE.json` (raw 24 new completions),
`results/F179_M3_ORACLE_EFFECT_PROBE_VERDICT.json` (combined verdict + 6-condition
rows), `results/F179_M3_ORACLE_QUALITATIVE.json` (hand-authored verdict),
`results/F179_M3_ORACLE_MANIFEST.json`, `results/F179_M3_ORACLE_SOFT_MANIFEST.json`;
scripts `scripts/gen_memory_prefix_oracle.py`,
`scripts/oracle_effect_probe_transport.py`,
`scripts/oracle_effect_probe_verdict.py`. Object files under
`outputs/r179-m1/s2a/` (oracle `fc43a2fb…`, soft `ccef8ee0…`).

## 9. Notes

- The **soft** condition is a deliberate addition beyond the issue's specified
  `oracle_pack` condition, made because the hard oracle is flat on 6/8 heads; it
  is reported as supplementary and changes nothing about the primary oracle
  result (both are generic). Its purpose is to make the selector exoneration
  robust to the "the oracle was a weak selector" objection.
- The oracle is a **cheating upper bound** (uses the persona's own held-out
  content) and is **not deployable**; it is labeled as such in the manifest and
  meta.
- No engine change in this issue; engine is a82155c. NaN detection uses the
  word-boundary regex plus a vocab/`U+FFFD` check (same as #227/#228).
