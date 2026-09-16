# #177: on-device multi-tool routing -- implementation complete, ladder re-run DONE

Date: 2026-09-15. Scope: `LocalLLM/natural_farming_chat` (Flutter app,
Cactus SDK, local 0.6B model).

## What #177 actually asked for, and why the plan changed mid-flight

The original ticket (chainlink #177) assumed Cactus's native
`select_relevant_tools` (cactus-engine/src/rag.cpp) was ready-to-use,
on-device, model-free routing infrastructure and should be registered
against directly. Investigation this session found that assumption false
on both ends of the stack:

- `cactus-engine/src/rag.cpp`'s C++ `select_relevant_tools` (hybrid
  BM25+embedding+RRF) is **not exposed through the Dart FFI layer** --
  `tool_rag_top_k` never appears outside the C++ source. It is unreachable
  from Dart today.
- `packages/cactus/lib/services/tool_filter.dart`'s `ToolFilterService` is a
  full Dart implementation (keyword + embedding-based filtering) that
  **exists but is never called anywhere in the app**.
  `model_service.dart` computed `effectiveTools = tools ?? farmingTools` --
  the raw, unfiltered list, always, before this session's changes.

Per owner decision (recorded as a standing project memory,
`feedback_dont_use_cactus_native_routing.md`): do not wire up either of
these. Instead, port #175's own validated router (`tool_router.py`) faithfully
to Dart and use that. This is what was built; the rest of this doc reports
on that build, not on registering Cactus's native mechanism.

## What was built (all committed, `LocalLLM/natural_farming_chat` commit
`7e41fed`)

1. **`lib/services/tool_router.dart`** -- faithful Dart port of
   `rlm_slice/tool_router.py`: same tokenizer (`[a-z0-9]+`, length > 2),
   same BM25 constants (k1=1.5, b=0.75) and formula, same RRF fusion
   (k=60, embedding weight 0.8, BM25 weight 0.2). Scores every tool, exposes
   `RoutingResult.selected(topK, minScore)` mirroring the Python
   `RoutingResult.selected`.
2. **Two domain tools**, ported from `rlm_slice/domain_tools.py`:
   - `bioaccumulator_ratio` (`lib/services/bioaccumulator_ratio_service.dart`),
     backed by a new bundled asset `assets/plant_data/Crop_Requirements.json`
     (copied from this repo's `data/sources/gemma4/spaces/data/
     crop_requirements.json`) plus the already-bundled
     `plant_mineral_profiles.json` via `PlantLookupService` (given a new
     `byCommonName` getter). Carries forward the negative-ratio fix from
     #175 (no leading `-?` in the NPK range regex) -- regression-tested
     against the real bundled Corn/Okra data (`test/domain_tools_test.dart`).
   - `regional_recipe_lookup` (`lib/services/recipe_lookup_service.dart`),
     backed by a new bundled asset `assets/plant_data/Enriched_Recipes.json`
     (copied from `enriched_recipes.json`, 14 KNF/JADAM recipes with
     `primary_nutrient`/`suggested_bioaccumulators`) plus a new reverse
     lookup added to `RegionPlantService` (`regionsGrowingCrop`). Deliberately
     does NOT touch the older, still-validated `NF_Recipes.json`.
3. **Both registered as `CactusTool`s** in `model_service.dart`'s
   `farmingTools` (now 5 tools total) and wired into `ToolExecutor`'s
   dispatch switch, following the existing tools' exact pattern.
4. **Router wired into `effectiveTools`**:
   `ModelService.generateCompletionWithTools` now routes `farmingTools`
   through `tool_router.dart` using the app's existing (but previously
   never-initialized-in-`main.dart`) `EmbeddingService`, selecting the
   top-2 tools by RRF score (comment in `model_service.dart` records the
   reasoning: #175 found top-1 routing and inject-all-viable statistically
   indistinguishable, so an exactly-tuned k isn't load-bearing; top-2 keeps
   the now-5-tool list narrow for a 0.6B model already known (G9/#156) not
   to reliably self-initiate tool calls). Retrieval is not part of the
   routed list -- there was no retrieval step wired into the live chat flow
   in this app to begin with (`RagService`/`EmbeddingService` were both
   fully unused prior to this change), so nothing needed to be kept
   separate from routing; the router is additive.
   Fails open to the full tool list if no embedding model is available.

## What is tested, and how

- `flutter test` (no device needed): **26/26 passing**.
  - `test/tool_router_test.dart` mirrors
    `qwen3-finetune/tests/test_tool_router.py`'s cases exactly (tokenizer,
    BM25 zero/positive-overlap, cosine similarity edge cases, RRF constants,
    `selectRelevantTools` with a deterministic stub embedder -- no live
    model required).
  - `test/domain_tools_test.dart` exercises `BioaccumulatorRatioService` and
    `RecipeLookupService` against the **real bundled assets** (97 crops, 14
    recipes, 660 plants, region data) -- including the negative-ratio
    regression case (Corn's Phosphorus requirement is literally `"30-70"`,
    the exact string that broke the GPU-side regex).
- `flutter analyze`: clean (only pre-existing `avoid_print` lint infos,
  same as the rest of the codebase).
- `flutter build apk --debug`: **builds clean**.

## The harness that had to be built first

The existing `ladder_benchmark_test.dart` harness does not exercise
tool-calling at all -- it calls `generateCompletion` with a raw MCQ prompt,
no `tools` argument, matching `evaluate.py`'s protocol exactly (this is why
F9's numbers are directly comparable to the GPU harness's think-off
baseline). A new `integration_test/ladder_benchmark_multitool_test.dart`
was built instead. Since the ladder's MCQ items carry no structured
`crop`/`candidate_plant`/`region`/`problem` fields (unlike #175's GPU
harness, where `multi_tool_context.py`'s controller had those), a new
`lib/services/tool_arg_extractor.dart` ports `multi_tool_context.py`'s
`extract_args`/`_find_in_text` (closed-vocabulary substring match over the
question+options text) so both domain tools can still fire deterministically,
no model call needed. Per item: extract args for each tool, run whichever
tools have extractable args, keep only "viable" ones (real content found --
same `_has_real_content` gate as the GPU side), inject all viable tools'
output into the prompt (not top-1 routing -- #175's GPU-side dual-relevant
eval already found top-1 and inject-all-viable statistically indistinguishable,
so this keeps the simpler design), then generate the answer with the same
prompt/generation params as the baseline harness.

The embedding-based routing audit (RRF scores) is skipped in this harness:
initializing a second live `CactusLM` context (the embedding model)
alongside the already-initialized chat model reproducibly crashed the
native engine on-device ("C++ error: Failed to create model" immediately
after the chat model finished initializing) -- the same failure mode
`model_service.dart` already documents for why semantic tool filtering is
disabled in the live chat path. This costs the audit's RRF-score field
only, not tool coverage: selection here never depended on rank anyway.

35/35 `flutter test` passing throughout (`test/tool_arg_extractor_test.dart`
added for the new extraction logic).

## Champion baseline for comparison

Per `F9_ON_DEVICE_LADDER_RESULTS.md`'s corrected (calibrated) numbers, and
`model_service.dart`'s `modelSlug` comment ("the promoted champion,
chainlink #163"): the deployed champion is **distill-v2-rationale
(calibrated)**, on-device baseline **0.8381 (176/210)** -- not v5clean's
0.7810/0.8095 line.

## The on-device run

Pixel 7 connected via USB (`2C090DLH2001QL`). Full 210-item
`natural_fertilizers_benchmark_v1_graded.jsonl`, greedy decoding, same
generation params as the F9 baseline harness. Raw per-item results:
`results/ladder210_multitool_distill-v2-rationale.json`.

| | Accuracy | vs. baseline (176/210) |
| --- | --- | --- |
| Multi-tool augmented | **177/210 = 0.8429** | two-proportion z = 0.133 |
| Wilson 95% CI | [0.7875, 0.8859] | baseline point estimate falls well inside |

**Not a measurable difference.** +1 item over baseline, z=0.133 is nowhere
near significance, and the baseline's own 0.8381 sits comfortably inside
this run's 95% CI. This matches the qualitative shape of #175's own
val_v3 finding (general-purpose questions mostly don't need these tools),
not the dual-relevant eval's finding (a significant win when the tool's
content is load-bearing) -- because the natural_fertilizers ladder, like
val_v3, is general-purpose natural-farming content, not purpose-built to
require a bioaccumulator ratio or a region lookup.

**Tool-firing breakdown**: 29/210 items (13.8%) had at least one domain
tool fire with real content -- 28 via `regional_recipe_lookup`, 10 via
`bioaccumulator_ratio` (some items had both). On those 29 items, accuracy
was 24/29 = 0.828 -- *lower* than the 181 items where nothing fired
(153/181 = 0.845). At n=29 this is not a real signal either direction (one
or two items flipping would reverse it), but it is worth stating plainly:
this run found **no evidence that firing a domain tool helped**, on this
benchmark, in either direction.

## Verdict

**Multi-tool augmentation does not measurably change on-device ladder
accuracy for the deployed app, in this run.** No regression (ruling out a
concern that the extra prompt content or routing overhead could hurt), but
also no improvement -- consistent with #175's own finding that
tool-augmentation's benefit is real but conditional on the tool's content
actually being what a question needs, and most natural_fertilizers ladder
questions (like most of val_v3) don't need this content. This is the
correct on-device counterpart to #175's honest GPU-side conclusion, not a
contradiction of it: the "does routing/augmentation ever help" question was
already answered affirmatively on a purpose-built eval; this run answers
the separate "does it help *this specific general benchmark*" question,
and the honest answer is no, not yet -- not because the mechanism is
broken (it fired correctly on 13.8% of items, exactly as designed and
verified by 35 passing unit tests), but because this benchmark mostly
doesn't ask questions these two tools are built to answer.

Recommendation: do not block deployment of the multi-tool augmentation on
this result (it caused no regression), but do not claim it as an
accuracy win for the app either. If a future benchmark or real usage
pattern skews toward bioaccumulator-ratio or region-specific questions,
re-run this harness against that slice -- this result does not rule out a
benefit there, it only rules out one on the general ladder.
