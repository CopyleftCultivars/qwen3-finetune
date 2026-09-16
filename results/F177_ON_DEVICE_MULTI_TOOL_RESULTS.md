# #177: on-device multi-tool routing -- implementation complete, ladder re-run BLOCKED (no Pixel 7 connected)

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

## What is BLOCKED

`adb devices` returned an empty device list throughout this session -- no
Pixel 7 (or any device) connected. Per the process this ticket specified,
work stopped here rather than guessing at on-device behavior. **The
210-item `natural_fertilizers` on-device ladder re-run
(`natural_fertilizers_benchmark_v1_graded.jsonl` via
`integration_test/ladder_benchmark_test.dart`) has not been run with
multi-tool augmentation active, and no on-device score exists yet.**

Two things are true and worth being explicit about for whoever resumes
this:

1. **The existing `ladder_benchmark_test.dart` harness does not exercise
   tool-calling at all** -- it calls `generateCompletion` with a raw MCQ
   prompt, no `tools` argument, matching `evaluate.py`'s protocol exactly
   (this is why F9's numbers are directly comparable to the GPU harness's
   think-off baseline). Re-running *that* harness unmodified would
   reproduce F9's numbers, not measure anything about the new tools --
   it would not exercise `generateCompletionWithTools`, the router, or
   either domain tool at all.
2. To actually get a multi-tool-augmented on-device ladder score
   comparable to #175's GPU-side dual-relevant eval, the harness needs a
   variant that, per MCQ item: runs the router against the question text,
   executes any domain tools it selects (which requires an argument-
   extraction step from the MCQ's `question` text -- the ladder items don't
   carry structured `crop`/`candidate_plant`/`region`/`problem` fields the
   way #175's GPU harness's controller-owned `multi_tool_context.py` did),
   formats the results into context, and generates the answer as
   `_generateWithTools`'s phase 2 does. That controller-owned argument-
   extraction step is not yet built for the on-device harness and is the
   next piece of work before a ladder run is possible, device access aside.

## Champion baseline for comparison, once a device is available

Per `F9_ON_DEVICE_LADDER_RESULTS.md`'s corrected (calibrated) numbers, and
`LocalLLM/natural_farming_chat/lib/services/model_service.dart`'s
`modelSlug` comment (currently `qwen3-nf-distill-v2-cact-int4-ios`, "the
promoted champion, chainlink #163"): the currently-deployed champion is
**distill-v2-rationale (calibrated)**, on-device baseline **0.8381
(176/210)**, not v5clean's 0.7810/0.8095 line -- the deployed app is not
running v5clean. Whatever multi-tool score comes out of a future run should
be compared against **0.8381 (176/210)**, not the v5clean numbers restated
in #177's ticket text.

## What's needed to resume

1. Connect a Pixel 7 over USB; confirm with `adb devices` (must show a
   device, not an empty list).
2. Decide on and implement the argument-extraction step for the on-device
   ladder harness (point 2 above) -- likely a new
   `integration_test/ladder_benchmark_multitool_test.dart` that mirrors
   `ladder_benchmark_test.dart` but calls `ModelService.generateCompletionWithTools`
   (with `embeddingService` wired) instead of the raw `generateCompletion`,
   executes any selected tools via `ToolExecutor`, and generates the final
   answer the same two-phase way `main.dart`'s `_generateWithTools` does.
3. `adb push` the model bundle, embedding model bundle, and the 210-item
   benchmark JSONL as `ladder_benchmark_test.dart`'s header comment
   describes; `adb shell svc power stayon usb` first (F9's replication note:
   this avoids USB power-management timeouts on long-running on-device
   tests).
4. Run the new integration test, `adb pull` `ladder_results.json`, score it
   the same way F9 did, and update this doc with the real number and a
   verdict against 0.8381 (176/210).

## Verdict

**Not yet determined.** Implementation is complete and unit-tested; the
on-device accuracy read this ticket's accept criterion requires is blocked
on (a) device access and (b) a small amount of additional harness work
(argument extraction for the ladder's un-structured MCQ text) that was not
part of the original ticket's scope but is a real prerequisite discovered
while trying to actually wire this up. Do not treat the implementation
work above as evidence either way about whether multi-tool augmentation
helps the deployed app -- that question is exactly what remains open.
