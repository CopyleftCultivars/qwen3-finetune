# KNF Gemma4 — Current Status & TODO

## Phase 1 — COMPLETE
All core files implemented and data generated:
- `spaces/knf_tools.py` — 6 tool schemas + handlers (pure dict lookups)
- `spaces/rag.py` — embedding, cosine retrieval, BACKGROUND KNOWLEDGE guard
- `spaces/app.py` — Gradio UI wired with tool-call dispatch loop and RAG injection
- `spaces/data/enriched_recipes.json` — 14 recipes
- `spaces/data/plant_mineral_profiles.json` — 660 plants with ppm mineral data
- `spaces/data/local_plants_by_region.json` — 127 regions
- `spaces/data/crop_requirements.json` — 97 crops
- `spaces/requirements.txt` — includes numpy

---

## Phase 2 — IN PROGRESS

### ✅ Change 1: Remove `bioaccumulators.json` (DONE)
`spaces/knf_tools.py` — all 3 bioaccumulators references removed:
- `_bioaccumulators` global var
- `_load()` line
- Fallback blocks in `handle_get_plant_npk()` and `handle_find_plants_by_nutrient()`

### ✅ Change 2: Expand RAG Corpus (DONE)
- `scripts/build_rag_corpus.py` — expanded from 20 → 50 queries; `TOP_PER_QUERY` 5→10; `MIN_SCORE` 30→20
- `spaces/data/rag_corpus.json` — regenerated: **161 chunks, ~16,249 tokens** (was 65 chunks)
- Format: plain `list[str]` — no metadata fields
- `spaces/rag.py` — updated to index `_corpus[i]` directly (no `.text` key lookup)

### ⏳ Change 3: Pre-compute Embeddings Offline (BLOCKED)
`scripts/precompute_embeddings.py` is ready but cannot run — `llama_cpp` is not installed in the active Python environment.

**To unblock:**
```bash
# Option A — activate the venv/conda env that has llama_cpp, then:
python scripts/precompute_embeddings.py

# Option B — install in current env:
pip install llama-cpp-python

# The script downloads gemma4.Q4_K_M.gguf (~3 GB, cached in ~/.cache/huggingface/)
# and writes spaces/data/rag_embeddings.npy (~1-5 MB)
```

Once `rag_embeddings.npy` exists, `rag.py` loads it at startup (instantaneous).
Without it, `rag.py` falls back to runtime embedding (slow on HF Spaces CPU).

---

## Deployment Checklist

- [ ] Run `scripts/precompute_embeddings.py` → produces `spaces/data/rag_embeddings.npy`
- [ ] Commit `spaces/data/rag_corpus.json` + `spaces/data/rag_embeddings.npy` to HF Space repo
- [ ] Verify startup: `RAG: loaded 161 pre-computed embeddings` in Space logs
- [ ] Test tool-call flow: *"I'm in the Pacific Northwest, growing tomatoes, it's June"*
  - Should fire: `get_crop_requirements` → `find_plants_by_nutrient` → `get_knf_recipe`
- [ ] Test RAG-only flow: *"What are the benefits of comfrey as a fertilizer plant?"*
  - Should NOT fire any tool calls; RAG chunk should appear in system context
- [ ] Confirm `spaces/data/` total size < 10 MB
