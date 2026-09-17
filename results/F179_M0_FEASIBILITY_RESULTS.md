# F179 / M0 — Phase 0 Feasibility: reserved-token embedding injection in Cactus

Read-only investigation into epic #179 ("on-device long-term persona memory via
reserved-token embedding injection in Cactus"). No training code, no model
changes, no Cactus edits, no conversion runs.

Champion under test: **distill-v2-rationale / "distill-calibrated"**
- HF/merged source: `outputs/merged-model-distill-v2-rationale` (safetensors)
- Deployed Cactus bundle: `[LOCAL_PATH]`
- Engine source: `[LOCAL_PATH]`
- Graph runtime: `[LOCAL_PATH]`
- Convert tooling: `[LOCAL_PATH]`
  and `cactus-sdk/python/cactus/convert/`

Method note (per the #172 postmortem landmine): every "does the source expose X"
claim below was cross-checked against the **compiled symbol table** of
`cactus-engine/build/libcactus_engine.so` (the ARM64 build that ships) and the
**actual bundle artifacts** (`conversion_manifest.json`, `components/manifest.json`,
`.weights` headers), not just by reading `.cpp`/`.h`. A native `generate()` call
was not possible on this host: the shipped engine is `ELF aarch64` only, and this
host is `x86_64` (no cross sysroot for the aarch64 `libcurl`/`libstdc++` deps), so
the strongest available runtime checks were symbol-table + artifact-header
inspection. That limitation is called out where it matters.

---

## #180 (P0.1) — Is there a C FFI / engine entry point that accepts embeddings or raw tensors instead of token IDs?

**Answer: No entry point at the model level accepts embeddings or raw tensors.
Every model-level generate/prefill entry takes text (`messages_json`) or token
IDs (`uint32_t*`), never a tensor of hidden states.**

The complete model-level FFI surface (`cactus-engine/cactus_engine.h`, confirmed
against `nm -D --defined-only libcactus_engine.so`):

| Entry point | Prompt input | File:line |
|---|---|---|
| `cactus_complete(model, messages_json, ...)` | chat messages (tokenized internally) | `cactus_engine.h:37-48` |
| `cactus_prefill(model, messages_json, ...)` | chat messages (tokenized internally) | `cactus_engine.h:50-59` |
| `cactus_submit(model, const uint32_t* prompt_tokens, num_tokens, ...)` | **raw token IDs** | `cactus_engine.h:66-73`, impl `batch_scheduler.cpp:186` |
| `cactus_score_window(model, const uint32_t* tokens, ...)` | raw token IDs (scoring, not prefill) | `cactus_engine.h:94-103` |
| `cactus_tokenize` / `cactus_render_prompt` | text -> token IDs | `cactus_engine.h:77-92` |
| `cactus_embed(model, text, float* out, ...)` | **text -> embedding output** (the other direction) | `cactus_engine.h:146-153`, impl `embed.cpp:32` |

There is no `cactus_prefill_embeds`, no `inputs_embeds` parameter, and no
`"input_ids"`/`"embeds"` key in the inference-options JSON parser
(`parse_inference_options_json`, `utils.h:1404-1473`). `InferenceOptions`
(`utils.h:506-525`) is sampling-only — no tensor field.

### The one place raw hidden-state vectors are fed to the decoder (internal only)

`Model::prefill_with_media(tokens, image_paths, audio_features_per_message, ...)`
(`engine.h:661-664`, impl `model.cpp:2882`) is the multimodal warm-injection path.
For each prompt position whose token id equals the model's image/audio placeholder
token (`image_tok` / `audio_tok`), it writes the encoded media row directly into
the decoder's `inputs_embeds` graph input (`model.cpp:2519-2535`,
`model.cpp:2741-2774`). This proves the decoder graph **can** accept precomputed
hidden states via `inputs_embeds` — but:

- it is **not exposed through the C FFI** (no header entry; no `nm` symbol),
- it only fires for media placeholder token IDs, not arbitrary text positions,
- it is gated by `supports_warm_media_injection()` (`engine.h:708-712`), which
  requires the decoder graph to have an `inputs_embeds` input node (the Qwen3
  text path here goes input_ids -> in-graph `aten.embedding` instead).

### The lower-level graph API does accept arbitrary tensors — but it is not the model path

`cactus-graph/cactus_graph.h` (bundled into the same `.so`, 115 exported
`cactus_graph_*` symbols) exposes the primitive building blocks:
`cactus_graph_set_input` (`:871`), `cactus_graph_set_external_input` (`:874`),
`cactus_graph_embedding_from_tensor` (`:949`), `cactus_graph_embedding_from_file`
(`:951`), `cactus_graph_mmap_embeddings` (`:953`). These are what the engine uses
internally to build/bind components. Using them to inject custom embeddings means
**rebuilding/reimplementing the model execution graph**, i.e. the "fork Cactus"
option — not a supported injection hook.

### Landmine confirmed (source looked wired; symbol is absent)

The Python binding ships a convenience `cactus_decode_tokens(model, tokens, ...)`
(`python/cactus/bindings/cactus.py:1246`) that takes raw token IDs. Its C symbol is
registered via `_bind_optional` (`:771`), which **silently no-ops when the symbol is
missing**, and `cactus_decode_tokens` is **not exported by the shipped `.so`**
(`nm` shows only `cactus_graph_gated_deltanet_decode`). Calling that helper raises
`AttributeError`. The working raw-token path is the lower-level
`cactus_batch_start` + `cactus_submit` + `cactus_batch_stop` trio (all exported).

**Consequence for the epic:** the reserved-token mechanism cannot feed vectors as
embeddings. It must reference reserved **token IDs**, and those IDs must resolve
to pre-existing rows in the embedding table. Raw token IDs *are* injectable
(`cactus_submit`, exported), so the "reference the reserved IDs" half of the trick
is live. The "learned vector values" half must be carried by the embedding table.
