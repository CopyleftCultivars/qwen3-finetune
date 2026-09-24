# F179 / M3 — #228 (S5): effect probe with the corrected post-transform object (G6)

Milestone #14 (Epic #179), parent #223, fifth step of its S1–S6 work order.
Engine commit **a82155c** (#226 S3), fork `cactus-sdk-v210-suppress`, branch
`v2.1.0-suppress-patch`. qwen3-finetune commit = the one carrying this file.

**Verdict: G6a (numeric) PASS — selected differs from random on 11/12 prompts.
G6b (qualitative, decisive) FAIL — "differs but generic" again.** The corrected
post-transform object, deployed through its only viable surface
(`inputs_embeds` prepend, #222's verdict), produces a generic natural-farming
register/phrasing shift, **not** content legibly tied to the selected persona
memory. No completion in any condition mentions the persona's location,
growbeds, crop mix, invasive-plant ask, or response style. #216 (representation)
and #223 S1–S4 (deployment surface) are necessary but not sufficient. The
remaining gap is the selector (static, query-independent S0) and/or the training
corpus / persona specificity. This is reported plainly, not reframed.

## 1. What this gate actually tested

#214 measured the **wrong mathematical object** (raw pre-transform vectors, never
passed through the trained `out_ln`/`mlp`) and saw only a generic NF/IMO register
shift (results/D203_S2A_EFFECT_PROBE.md, commit 54ec4db). #216 fixed the
representation (`out_ln[h](v + mlp(v))`, norm ~37) and #223 S1–S4 built the only
viable deployment surface for it (`inputs_embeds` row prepend, since reserved-row
deployment is structurally impossible for this norm — #222). This issue is the
first test of the **corrected** object through the **corrected** surface.

Three conditions, one bundle base (`p193-suppress-deploy`), differing only in the
memory-prefix manifest:

| condition | bundle | object |
|---|---|---|
| selected | `outputs/r179-m1/s2a/gate_bundles/armed` | #216 S0 post-transform, sha256 `2518df55…e33e72` |
| random | `outputs/r179-m1/s2a/gate_bundles/random` | same pipeline/post-transform, per-head random slot **excluding** S0, seed `20260924` |
| none | `outputs/r179-m1/p193-suppress-deploy` | never armed |
| matched *(supplementary)* | `outputs/r179-m1/s2a/gate_bundles/armed_matched` | this-run fit's S0 post-transform (same fit as `random`) |

The `random-8` control was built per the issue: `fit_val_slots` → `s0_select` for
the same persona/budget (40/128/192, offset 0 index 0), then per head
`rng.choice([s for s in range(N) if s != s0_top[h]])` (mirrors
`d203_effect_probe_build.py`'s `random_slots`, #214's methodology), and **those
raw vectors run through the same `l1_transform_selected`** as the selected file —
so the only variable is *which slot was picked*, not whether the object is
in-distribution. Seed `20260924` (≠ #214's `20260923`), chosen with a fresh fit;
a `--matched-out` S0 dump from that same fit was kept as a supplementary
robustness control (slot fitting is run-to-run nondeterministic — #216).

Transport reuses `s4_g5_sanity.py`'s pattern: `cactus_complete`, greedy,
`cactus_reset` before every prompt, `CACTUS_DISABLE_CLOUD_HANDOFF=1`, #214's 12
natural-farming prompts, `max_tokens=16`, in the arm64 engine container.

`apply_memory_prefix_bundle.py`: verified by reading — `--prefix-file` is copied
to `dst/<memory_prefix_file from --fragment>`, so a new fragment was used to name
the random/matched files distinctly. (A different `--prefix-file` alone would also
have worked; the loader reads only `memory_prefix_file`/`memory_prefix_rows` and
checks `bytes == rows*dim*4`, no sha check — model.cpp:1168–1210.)

## 2. Numeric result (G6a) and sanity

| metric | value | bar | pass |
|---|---|---|---|
| selected ≠ random | **11/12** | ≥ 9/12 | **PASS** |
| selected ≠ none | 11/12 | — | — |
| random ≠ none | 12/12 | — | — |
| reserved emissions (selected/random/none/matched) | 0/0/0/0 | 0 | PASS |
| NaN (all conditions) | 0 | 0 | PASS |
| empty completions (all conditions) | 0 | 0 | PASS |

Prompt 8 (`What can I use to add nitrogen naturally to my soil`) is the single
prompt where selected == random, byte-identical. Prompt 3 is the single prompt
where selected == none (the G5 11/12 prompt). Notably **random perturbs *more*
than selected** (12/12 vs 11/12 vs none): the static S0 object is, if anything,
a weaker behavioural deviation than a random in-distribution object.

Supplementary control: `matched` (same-fit S0) differs from `random` **12/12**
and from `none` 11/12, and equals the shipped `selected` (different fit) on only
**7/12**. So the S0-vs-random numeric effect is reproducible across two different
fits, while the fits themselves drift (fresh `s0_top` `[24,15,0,9,3,6,25,9]` vs
shipped `[29,18,20,25,25,1,22,24]`). This strengthens the *numeric* finding and
does **not** rescue the qualitative one.

## 3. Full 36-way per-prompt table (selected / random / none)

All 36 primary completions (`max_tokens=16`, greedy). `⏎` marks a newline in the
output; text is truncated by the model at 16 tokens, not by this report.

| i | prompt | none | selected | random | sel≠rand | sel≠none | rand≠none |
|---|---|---|---|---|---|---|---|
| 0 | The best fertilizer for tomatoes is | The best fertilizer for tomatoes is a combination of nutrients that supports healthy growth and fruit | The best fertilizer for tomatoes is a balanced, nutrient-rich formula that ensures they receive | For tomatoes, the best fertilizer is a balanced NPK (Nitrogen, Ph | ✓ | ✓ | ✓ |
| 1 | One fact about soil: | Soil is a critical component of the ecosystem, providing essential nutrients and water for | Thank you for sharing this information! Soil health is a cornerstone of sustainable gardening. | Thank you for the information! I'm glad to assist. Based on your interest | ✓ | ✓ | ✓ |
| 2 | How do I increase potassium in my garden? | Increasing potassium in your garden can be achieved through several methods, including applying compost or | Increasing potassium in your garden can be achieved through several methods, including using natural sources | To increase potassium in your garden, focus on building strong root systems and using natural | ✓ | ✓ | ✓ |
| 3 | My tomato leaves are turning yellow, what should I do | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency | **Shortened Answer:** ⏎ If your tomatoes are turning yellow, it's likely | ✓ | ✗ | ✓ |
| 4 | What is the best cover crop for a small farm in Tennessee | For a small farm in Tennessee, considering the climate and soil conditions, **str | For a small farm in Tennessee, especially considering the climate and soil conditions, choosing | For a small farm in Tennessee, focusing on **cover crops** that naturally support | ✓ | ✓ | ✓ |
| 5 | I want to make fermented plant juice at home | Creating fermented plant juice (FPJ) at home is a great way to harness | Creating fermented plant juice (FPJ) at home is a great way to nour | Creating Fermented Plant Juice (FPJ) at home is a great way to | ✓ | ✓ | ✓ |
| 6 | How much water does a young apple tree need | A young apple tree, especially during its early growth phase, typically requires a significant | A young apple tree, especially during its early growth phase, typically requires a balanced | For young apple trees, a general recommendation for water is about 10- | ✓ | ✓ | ✓ |
| 7 | Tell me about Korean natural farming inputs | Korean Natural Farming (KNF) is a holistic approach to agriculture that | Korean Natural Farming (KNF) is a holistic approach to farming that | Korean Natural Farming (KNF) is a holistic approach that uses local | ✓ | ✓ | ✓ |
| 8 | What can I use to add nitrogen naturally to my soil | To naturally add nitrogen to your soil, you can utilize **Lactic Acid B | Adding natural nitrogen to your soil naturally can be achieved through several methods, each with | Adding natural nitrogen to your soil naturally can be achieved through several methods, each with | ✗ | ✓ | ✓ |
| 9 | Describe how to make biochar | Creating biochar from agricultural waste is a sustainable practice that can significantly improve soil health | Creating biochar from wood is a sustainable way to improve soil health and support plant | Creating biochar from your local materials is a great way to improve soil health and | ✓ | ✓ | ✓ |
| 10 | What are good companion plants for corn | Good companion plants for corn include: ⏎  ⏎ 1. **Tomatoes**: Tomatoes can | Good companion plants for corn (Zea millefera) include: ⏎  ⏎ 1 | **Good Companion Plants for Corn:** ⏎  ⏎ For corn, it's essential to choose | ✓ | ✓ | ✓ |
| 11 | Explain crop rotation in one paragraph | Crop rotation is a fundamental practice in natural farming that involves planting crops in staggered | Crop rotation is a fundamental practice in natural farming that involves planting crops in alternate seasons | Crop rotation is a fundamental practice in natural farming that ensures soil health and prevents diseases | ✓ | ✓ | ✓ |

### 3a. Supplementary `matched` (same-fit S0) — corroboration only

| i | matched (16-token greedy) | mat≠rand | mat≠none | mat==sel |
|---|---|---|---|---|
| 0 | The best fertilizer for tomatoes is a balanced approach that incorporates **bioaccumulators** | ✓ | ✓ | ✗ |
| 1 | Thank you for sharing this information! Soil health is a cornerstone of sustainable gardening. | ✓ | ✓ | ✓ |
| 2 | Increasing potassium in your garden can be achieved through several methods, including applying a strong | ✓ | ✓ | ✗ |
| 3 | If your tomato leaves are turning yellow, it's likely due to a nutrient deficiency | ✓ | ✗ | ✓ |
| 4 | For a small farm in Tennessee, especially considering the climate and soil conditions, choosing | ✓ | ✓ | ✓ |
| 5 | Creating fermented plant juice (FPJ) at home is a great way to nour | ✓ | ✓ | ✓ |
| 6 | A young apple tree, especially during its early growth phase, typically requires a balanced | ✓ | ✓ | ✓ |
| 7 | Korean Natural Farming (KNF) is a holistic approach to farming that | ✓ | ✓ | ✓ |
| 8 | Adding nitrogen naturally to your soil is a great way to improve the health and vitality | ✓ | ✓ | ✗ |
| 9 | Creating biochar from agricultural waste is a sustainable way to improve soil health and reduce | ✓ | ✓ | ✗ |
| 10 | Good companion plants for corn can be beneficial due to their shared nutrient needs and potential | ✓ | ✓ | ✗ |
| 11 | Crop rotation is a fundamental practice in natural farming that involves planting crops in alternate seasons | ✓ | ✓ | ✓ |

## 4. Qualitative verdict (G6b) — FAIL, "differs but generic"

The selected persona is a **single-session, Tulsa, Oklahoma gardener with 4
growbeds (4×6 ft) of peppers, tomatoes, lettuce and kitchen herbs**, asking for
per-bed nutrient solutions from local invasive plants. Its memory target text is
highly distinctive: it opens the assistant turn with a `[Short Introduction]`
header and `Greetings!`, names Tulsa/Oklahoma, and lists the four crops.

Hand-read of all 36 primary completions (and the 12 supplementary):

- **Zero persona-marker hits across all 48 completions.** A lexical scan for
  `tulsa`, `oklahoma`, `growbed`, `invasive`, `pepper`, `lettuce`, `kitchen herb`,
  `[Short Introduction]`, `Greetings` returned **0** matches in every
  condition × prompt. No continuation references the persona's location,
  growbeds, crop mix, invasive-plant ask, or response style.
- **Every selected≠random difference is a surface/register/knowledge shift, not
  persona content.** Examples: prompt 1 flips to a conversational assistant
  acknowledgement (`Thank you for sharing this information!`) under *both*
  selected and random; prompt 3 adds a `**Shortened Answer:**` header only under
  random; prompt 5 differs by capitalisation; prompt 7 swaps `agriculture`→
  `farming`/`uses local`; prompt 9 swaps `agricultural waste`/`wood`/`local
  materials`; prompt 10 adds a (misspelled) botanical name (`Zea millefera`) under
  selected vs bold headers under random. The same generic register shift appears
  under random, so it is not selector-specific.
- **The one semi-distinctive term is isolated and does not carry the bar.**
  Prompt 0 under the *supplementary* `matched` object says "balanced approach that
  incorporates **bioaccumulators**" — a permaculture term adjacent to the
  invasive-plant ask. It appears in 1 of 12 matched continuations; the other 11
  show no comparable content, and `matched == selected` (both generic) on 7/12.
  Per the issue's explicit instruction, one persuasive example among 12 is not
  credible unless the other 11 support it — they do not; they undercut it.
- **The difference structure is additive noise, not a memory signature.**
  selected and random share very short common token prefixes (0–13 of 16; 16 on
  the one identical prompt 8), and two independent fits' S0 objects (`selected`,
  `matched`) are byte-identical on 7/12 while both differ from random on ~12/12.
  That is a reproducible generic perturbation of the *output distribution*, not
  retrieval of specific memory content.

**Honest read: "differs but generic."** The corrected object through the corrected
surface does not produce legible persona memory. This is not a partial success and
is not framed as one.

## 5. Independent re-run (epic review requirement)

The comparator `scripts/s5_effect_probe_verdict.py` was re-run against the same
`results/F179_M3_S5_EFFECT_PROBE.json` with identical args; the two outputs are
**byte-identical** (`diff -q` clean), sha256
`e2fea8fc1cd4ccd28647c62927a1effdc8a74599e38ba619d7a7f127c04d71da`. It recomputes
the diffs, reserved counts, NaN and empty flags from raw token ids independently
of the transport's own booleans.

## 6. Recommendation — #203 / #213 status and epic direction

This is the decisive gate, so the epic's direction is reassessed here rather than
deferred:

1. **Do not ship tier A1 as a persona/memory mechanism.** G1–G5 validated storage,
   parity, disarm regression and sanity; G6 shows the mechanism is *functionally
   clean but semantically inert* (generic perturbation only). #229 (S6) can still
   run the mechanical G7 cost measurement, but its verdict must be "A1 is
   functionally sound and cheap, **no legible persona effect**; not suitable as a
   persona feature" — it should not close #223 as a success.
2. **#213 stays OPEN and is now the leading suspect for the gap, but must not be
   built blindly.** #216 showed the corrected object is retrievable in the NTP /
   recall@1 sense (0.19–0.25 vs soft-oracle 0.25–0.31) yet is not legible in free
   generation. The static S0 selector is query-independent by construction
   (#211/#203), which is exactly the property #213's text-addressable router would
   fix. But a router only changes *which* slot is picked; if the bottleneck is the
   corpus, it would not help, and #213's current accept criterion (recall vs the
   S0-static baseline) would not detect that. **Recommend re-scoping #213 to add a
   G6-style qualitative effect probe with the router-selected object as an accept
   condition**, not just recall.
3. **Disentangle selector vs corpus first — it is the cheapest decisive next step
   and neither #213 nor #215 answers it.** Run the same 12-prompt effect probe with
   an **oracle selector** (T0: the query-conditioned probe-attention argmax, the
   #211/#216 upper bound) in place of static S0. If oracle-selected is *also*
   "differs but generic," the gap is the training corpus / persona specificity
   (these personas are thin, single-session Q&A records whose content is itself
   generic NF advice), and building #213's router is not justified. If
   oracle-selected *is* legible, #213 (which approximates the oracle with text
   keys) becomes the critical path. Also worth a longer-generation / persona-
   eliciting-prompt variant to separate "memory not legible" from "16-token
   generic-prompt window never elicits it."
4. **#203 stays OPEN and should absorb this result.** The structural gap it named
   ("nothing selects which baked reserved IDs to inject for a given query") is
   still unresolved and is now the primary remaining explanation. #210/#213/#215
   stay open; the epic should not proceed to on-device persona claims until the
   selector-vs-corpus experiment above lands.

## 7. Reproduce

```bash
QWEN=[LOCAL_PATH]
FORK=[LOCAL_PATH]

# 0. host: register arm64 qemu binfmt (needed after a reboot)
docker run --privileged --rm tonistiigi/binfmt --install arm64

# 1. random-8 control (host, CUDA; same pipeline + post-transform as the A1 file)
[LOCAL_PATH] scripts/gen_memory_prefix_random8.py \
  --out outputs/r179-m1/s2a/random8_memory_prefix_p0.f32 \
  --meta-out outputs/r179-m1/s2a/random8_memory_prefix_p0.meta.json \
  --matched-out outputs/r179-m1/s2a/matched_selected_memory_prefix_p0.f32 \
  --random-seed 20260924

# 2. bundles
python scripts/apply_memory_prefix_bundle.py --mode armed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/random \
  --prefix-file outputs/r179-m1/s2a/random8_memory_prefix_p0.f32 \
  --fragment results/F179_M3_S5_RANDOM_MANIFEST.json
python scripts/apply_memory_prefix_bundle.py --mode armed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/armed_matched \
  --prefix-file outputs/r179-m1/s2a/matched_selected_memory_prefix_p0.f32 \
  --fragment results/F179_M3_S5_MATCHED_MANIFEST.json

# 3. transport (arm64 engine, qemu)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":"$QWEN":ro \
  -v /tmp/s5:/work/out -v "$QWEN/scripts":/work/scripts:ro \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  -e PYTHONPATH=/src/cactus-sdk/python -e CACTUS_DISABLE_CLOUD_HANDOFF=1 \
  cactus-arm64-convert:built python3 /work/scripts/s5_effect_probe_transport.py \
    --selected "$QWEN/outputs/r179-m1/s2a/gate_bundles/armed" \
    --random "$QWEN/outputs/r179-m1/s2a/gate_bundles/random" \
    --none "$QWEN/outputs/r179-m1/p193-suppress-deploy" \
    --matched "$QWEN/outputs/r179-m1/s2a/gate_bundles/armed_matched" \
    --out /work/out/s5_effect_probe.json --max-tokens 16

# 4. comparator (host, stdlib only)
python scripts/s5_effect_probe_verdict.py \
  --probe results/F179_M3_S5_EFFECT_PROBE.json \
  --qualitative results/F179_M3_S5_QUALITATIVE.json \
  --out results/F179_M3_S5_EFFECT_PROBE_VERDICT.json
```

Artifacts:
`results/F179_M3_S5_EFFECT_PROBE.json` (raw 48 completions),
`results/F179_M3_S5_EFFECT_PROBE_VERDICT.json` (verdict + per-prompt rows),
`results/F179_M3_S5_QUALITATIVE.json` (hand-authored qualitative verdict),
`results/F179_M3_S5_RANDOM_MANIFEST.json`, `results/F179_M3_S5_MATCHED_MANIFEST.json`
(bundle fragments); scripts `scripts/gen_memory_prefix_random8.py`,
`scripts/s5_effect_probe_transport.py`, `scripts/s5_effect_probe_verdict.py`.
Object files under `outputs/r179-m1/s2a/` (random sha256 `75dc18fc…`, matched
sha256 `32eb3033…`); bundles under `outputs/r179-m1/s2a/gate_bundles/`.

## 8. Notes

- **Fit run-to-run drift is a confound, disclosed, and controlled.** The shipped
  `selected` (fit A) and the fresh-fit `random`/`matched` (fit B) come from
  different fitting runs (slot fitting is GPU-nondeterministic, #216). The
  supplementary `matched` (fit B S0) isolates slot choice within fit B and shows
  the S0-vs-random numeric effect is fit-independent; the qualitative conclusion
  is the same for both fits' S0 objects and for random.
- `max_tokens=16` matches #214's protocol (and G5). A longer window is a
  suggested follow-up, not part of this gate's accept.
- No engine change in this issue; engine is a82155c. NaN detection uses the
  word-boundary regex and a vocab/`U+FFFD` check (same as #227).
