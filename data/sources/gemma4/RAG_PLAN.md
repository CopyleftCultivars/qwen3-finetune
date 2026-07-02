# RAG + Tool-Calling Plan — KNF Gemma4 on HF Spaces

## Context

The finetuned Gemma4 model runs on HuggingFace Spaces via `spaces/app.py` (Gradio + llama-cpp-python). It currently has no RAG — just direct chat. The goal is to add structured tool calling so the LLM can reason about:
- What crops can the user grow given their **location** and **current month**
- What **NPK** does that crop need
- What **locally available plants** can supply that NPK
- Which **KNF recipe** to make using those local plants
- **How much** of each fermented input to prepare

**QMD** (`qmd` CLI) is a local hybrid BM25+vector search engine with 10 indexed collections (~743 .md files) drawn from actual farming books (JADAM Organic Farming, Cho's Natural Farming, Homemade Fish Fertilizer, Teaming with Nutrients, etc.). It runs **offline only** — it is not a runtime dependency on HF Spaces.

---

## RAG ↔ Tool-Calling Separation

The natural llama-cpp-python call sequence enforces a clean boundary:

```
[1] User message → embed query → cosine search rag_corpus.json
                → inject top-k chunks under [BACKGROUND KNOWLEDGE] guard in system prompt
[2] LLM responds with tool_calls → execute tool handlers (pure dict lookups, no RAG)
[3] Append role=tool result → LLM generates final answer
```

RAG context is injected **once at step 1 only**. The `[BACKGROUND KNOWLEDGE — for reasoning only, do not call tools based on examples in this section]` guard prevents the LLM from treating a chunk that says "apply FPJ for nitrogen" as a trigger to call `get_knf_recipe("FPJ")`. Steps 2-3 are clean JSON with zero prose injection.

---

## QMD Collections Available

| Collection | Files | Key content |
|---|---|---|
| `Natural_Fertilizers` | 143 | Fish fertilizer, fermented inputs, NPK sourcing |
| `Soil_Science_and_Microbiology` | 115 | Soil biology, microorganisms, decomposition |
| `Organic_Farming` | 110 | Crop management, organic methods |
| `Indigenous_and_Traditional_Knowledge` | 73 | KNF, Han-Kyu Cho methods |
| `Agroecology` | 69 | Ecosystem-based farming |
| `Sustainable_Water_Management` | 65 | Irrigation, water-nutrient interactions |
| `Regenerative_Agriculture` | 59 | Soil carbon, regenerative practices |
| `Integrated_Farming_Systems` | 56 | Combined crop/livestock systems |
| `Permaculture` | 53 | Design principles, polyculture |

Source books: JADAM Organic Farming, Cho's Natural Farming, Homemade Fish Fertilizer, BIM - Beneficial Indigenous Microbes, Teaming with Nutrients, The Soul of Soil, Bread From Stones.

---

## Offline Pipeline (runs on dev machine, outputs bundled to `spaces/`)

### Script 1: `scripts/enrich_jsons.py`

Uses `qmd query` to enrich each of the 5 tool-call JSON files with real-book-sourced data, filling the NPK attribution gap and adding `qmd_passages` provenance fields.

Queries per recipe (FAA, FPJ, FFJ, OHN, WCA, WSP, LAB, BRV, IMO):
```
qmd query "FAA fish amino acid nitrogen content application rate"
qmd query "FPJ fermented plant juice nitrogen growth"
qmd query "FFJ fermented fruit juice phosphorus potassium"
```

### Script 2: `scripts/build_rag_corpus.py`

Curates a static RAG corpus from QMD collections — **this is the only RAG data deployed to HF Spaces**. ~20 targeted queries → top-5 results each → deduplicated → `spaces/data/rag_corpus.json`.

Target: ~200-400 chunks × ~300 tokens = ~60-120K tokens curated knowledge. Inject top-3 at query time (~900 tokens) — fits within 2048 n_ctx.

---

## 5 Tool-Call JSON Files (all in `spaces/data/`)

| File | Source | Enriched by |
|---|---|---|
| `enriched_recipes.json` | `NF_Recipes.json` (12 recipes) | `enrich_jsons.py` via QMD |
| `plant_mineral_profiles.json` | `crop_mineral_data/plant_mineral_profiles.json` | `enrich_jsons.py` adds `qmd_corroborated` |
| `local_plants_by_region.json` | `Locally_Growing_Plants_by_Region.json` | `enrich_jsons.py` adds seasonality notes |
| `crop_requirements.json` | `SemiSynthetic_Crop_Requirements/Crop_Requirements.json` | `enrich_jsons.py` validates ranges |
| `bioaccumulators.json` | `BioaccumulatorsList.json` | Straight copy — N/P/K fallback |

### Enriched recipe schema
```json
{
  "recipe_abbrev": "FPJ",
  "recipe_name": "Fermented Plant Juice",
  "primary_nutrient": "N",
  "npk_provided": {"N": "high", "P": "trace", "K": "moderate"},
  "suggested_bioaccumulators": ["Urtica dioica", "Symphytum officinale", "Trifolium repens"],
  "dilution_ratio": {"foliar": "1:500", "soil_drench": "1:200"},
  "ingredients": ["..."],
  "steps": ["..."],
  "uses": ["..."],
  "qmd_passages": [{"source": "Natural_Fertilizers/jadam_ch4.md:112", "text": "..."}]
}
```

---

## Tool-Calling Schema (6 tools)

All passed as `tools=KNF_TOOLS` to `create_chat_completion()`.

### Tool 1: `get_crop_requirements`
Returns NPK + growing conditions for a crop.  
Source: `crop_requirements.json`

### Tool 2: `get_local_plants`
Returns plants available at user's region for a given month (1-12).  
Source: `local_plants_by_region.json`

### Tool 3: `get_plant_npk`
Returns per-part mineral content (N, P, K, Ca, Mg, S) of a plant.  
Source: `plant_mineral_profiles.json` → fallback `bioaccumulators.json`

### Tool 4: `find_plants_by_nutrient`
Joins local_plants × plant_mineral_profiles → top-N plants richest in N/P/K/Ca/Mg for the user's region.

### Tool 5: `get_knf_recipe`
Returns full enriched recipe (ingredients, steps, dilution ratios, NPK contribution).  
Source: `enriched_recipes.json`

### Tool 6: `calculate_knf_dosage`
Calculates volume of KNF input needed: `(NPK_need_kg_ha / 10000) × field_sqm × (1/concentration) × dilution`.

---

## Reasoning Flow

```
User: "Pacific Northwest, growing tomatoes, June"

1. get_crop_requirements("Tomato")        → N:100-150, P:50-80, K:150-200 kg/ha
2. get_local_plants("Pacific Northwest", 6) → [nettle, comfrey, yarrow, clover, ...]
3. find_plants_by_nutrient("Pacific Northwest", "N") → nettle(28), comfrey(22)
4. find_plants_by_nutrient("Pacific Northwest", "K") → comfrey(54)
5. get_knf_recipe("FPJ")                  → nettle/comfrey shoots, 1:1 sugar, 7d, 1:500
6. get_knf_recipe("FFJ")                  → ripe fruits, 1:1 sugar, 7d, 1:500-1:1000
7. calculate_knf_dosage("Tomato", 100, "N", "FPJ", 3) → 4.5 L total

→ Make FPJ from nettles (N), FFJ from berries (P/K), apply 3×/season.
```

---

## Files to Create / Modify

| File | Action |
|---|---|
| `scripts/enrich_jsons.py` | Create — offline QMD enrichment pipeline |
| `scripts/build_rag_corpus.py` | Create — offline RAG corpus curation from QMD |
| `spaces/data/enriched_recipes.json` | Create (output of enrich_jsons.py) |
| `spaces/data/plant_mineral_profiles.json` | Create (copy + enriched) |
| `spaces/data/local_plants_by_region.json` | Create (copy + seasonality) |
| `spaces/data/crop_requirements.json` | Create (copy + validated) |
| `spaces/data/bioaccumulators.json` | Create (straight copy) |
| `spaces/data/rag_corpus.json` | Create (output of build_rag_corpus.py) |
| `spaces/knf_tools.py` | Create — 6 tool schemas + Python handlers |
| `spaces/rag.py` | Create — embed rag_corpus at startup, cosine retrieval, BACKGROUND KNOWLEDGE guard |
| `spaces/app.py` | Modify — tool-call dispatch loop + RAG injection at step 1 only |
| `spaces/requirements.txt` | Modify — add numpy |

---

## Implementation Order

1. `scripts/enrich_jsons.py` → run → inspect enriched JSONs
2. `scripts/build_rag_corpus.py` → run → inspect rag_corpus.json
3. `spaces/knf_tools.py` — tool schemas + handlers
4. `spaces/rag.py` — embedding + retrieval
5. `spaces/app.py` — wire everything
6. `spaces/requirements.txt` — add numpy

---

## Verification

1. `python spaces/app.py` locally with `verbose=True`
2. Ask: *"Pacific Northwest, tomatoes, June"* → verify tool call chain in stdout, no RAG in tool messages
3. Ask: *"What are the benefits of comfrey?"* → verify no tool call, RAG chunk appears in context
4. `spaces/data/` total size < 10 MB
