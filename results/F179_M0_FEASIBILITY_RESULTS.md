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

---

## #181 (P0.2) — Is the embedding table mmap'd + mutable, or baked immutable?

**Answer: it is mmap'd, but the mapping is `PROT_READ` (read-only) and the tensor
is CQ4-quantized at convert time. It is effectively baked immutable. Reserved
rows cannot be overwritten post-load without an engine change.**

### How `token_embeddings.weights` is loaded

The deployed bundle's `components/manifest.json` binds the embedding as a graph
**input** node with `kind: "embedding"`:

```
lm_encoder_step          node_id 3   -> token_embeddings.weights  (module.backbone.embed_tokens.weight)
decoder_step             node_id 312 -> token_embeddings.weights  (module.backbone.embed_tokens.weight_1)
decoder_prefill_chunk    node_id 312 -> token_embeddings.weights
lm_encoder_text_chunk    node_id 3   -> token_embeddings.weights
```

The Qwen3 text path performs the lookup in-graph: `lm_encoder_step/raw_ir.json`
contains `aten.embedding.default` (`padding_idx: 151654`) taking the weight
constant and `input_ids`. `Model::load_component_graph` (`model.cpp:762-790`)
loops over `bound_constant_bindings` and calls
`comp.graph->bind_mmap_weights(node_id, path)` for every binding — the `kind`
field (`"embedding"` vs `"weight"`) is **not** branched on.

### `bind_mmap_weights` maps the file and points the node at it

`cactus-graph/src/io.cpp:523-607`:

```cpp
auto mapped_file = std::make_unique<GraphFile::MappedFile>(resolved_filename);
...
set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);
```

i.e. the graph input node's data pointer *is* the mapped file. The mapping is
created in the `MappedFile` constructor (`io.cpp:860-885`):

```cpp
fd_ = open(filename.c_str(), O_RDONLY);          // io.cpp:862
mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);  // io.cpp:874
```

`PROT_READ` means any write through that pointer faults; the process never
`mprotect`s or re-maps it writable. `MAP_SHARED` is irrelevant for in-process
writes here because the protection forbids them (it would only matter if some
*other* process modified the file, which is not a supported/app-store-safe path).
`mmap_embeddings` (`io.cpp:399-460`) uses the identical `PROT_READ` pattern.

### It is also quantized (baked), not raw FP16

Parsed the deployed file header directly:

```
token_embeddings.weights: magic 0x54434143  flags 0x6 (orthogonal | interleaved_4row)
dims [151936, 1024]  precision enum 6 = CQ4  group_size 1024  num_groups 1
byte_size 77,791,232 (4-bit packed)  scales_bytes 2,405,152   file 80,196,480 B
```

`Precision` enum is `INT8=0, FP16=1, FP32=2, CQ1..CQ4=3..6`
(`cactus-kernels/cactus_kernels.h:10-18`). So the table is a single CQ4 blob
(codebook + per-group input scales + per-row norms), **not** an fp16 array.

### Why per-row precision is impossible in the current runtime format

A graph tensor carries exactly one scalar `precision` (`BufferDesc.precision`,
`cactus_graph.h:226`), and `bind_mmap_weights` rejects any file whose precision
differs from the pre-built node's precision and whose shape differs
(`io.cpp:539-544`). The `aten.embedding` node reads the whole table at one
precision. There is no per-row / per-range precision concept anywhere in the
`.weights` header or the graph buffer descriptor.

### Runtime-write paths that exist (all require a fork)

The graph API could point a node at a different buffer
(`cactus_graph_set_external_input`) or at an in-memory tensor
(`cactus_graph_embedding_from_tensor`), so a fork *could* hold an owned writable
copy and rewrite rows. But the model FFI does not expose the component graphs, and
re-quantizing a CQ4 row in place means recomputing the per-row norm and writing
packed 4-bit codes — not a drop-in `memcpy`. The app-level alternative (rewrite
`token_embeddings.weights` on disk and rely on `MAP_SHARED` coherency) depends on
the model living in a writable location and is fragile/unsupported.

**Consequence for the epic:** reserved-row vectors must be **baked at convert
time** (edit the merged model's `embed_tokens` rows, then convert) unless the
engine is forked to support a writable embedding buffer. Per-session in-place
overwrite of reserved rows is **not** possible with the stock engine.
