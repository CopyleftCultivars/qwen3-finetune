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

---

## #182 (P0.3) — Does cactus convert expose a per-tensor exclude/precision override?

**Answer: no. Granularity stops at whole-tensor / whole-component policy. There is
no row-range override, no per-tensor-name config/rules file, and the existing
per-component `--embedding-bits` flag is actually dead for this model because the
embedding policy hardcodes CQ4.**

### What the project's wrapper exposes

`conversion/convert_to_cactus.sh` accepts only `--merged-model`, `--output`,
`--bits {1,2,3,4}`, `--cache-context-length`, `--cactus-dir` (`:11-17`), and shells
out to `cactus convert ... --bits "$BITS" --reconvert ...` (`:104-111`). No
per-tensor or per-range option.

### What the underlying CLI exposes

`cactus-sdk/python/cactus/convert/cli.py` `build_parser()` (`:584-611`):

```
--bits / --language-bits / --vision-bits / --audio-bits / --embedding-bits   (choices 1-4)
--calibration-manifest  --device  --model-family  --strict  --force
--max-*-examples  --hessian-cache-in/out  --calibration-batch-size ...
```

`_bits_for_component(component, args)` (`cli.py:122-131`) maps a *component*
(whole tensor group) to a bit width. The convert loop applies this to every tensor
of that component (`cli.py:439-440`, `:498-499`). **There is no**
`--precision-map`, `--policy-file`, `--exclude`, or `name -> precision` config
anywhere in the pipeline (`grep` over `cactus/convert/**` finds no such loader;
the only `json.load` sites are hessian/provenance/config passthrough).

### The policy is hardcoded per tensor-type in Python

`cactus-sdk/python/cactus/convert/model_adapters/policy.py:20` —
`policy_for_tensor(match, shape, user_bits, family) -> TensorPolicy` is the single
decision point, a hardcoded `if/elif` ladder over output name / shape / family.
`TensorPolicy` (`policy.py:8-17`) has `action, precision, bits, component,
use_gptq, rotation, fallback_reason, layout` — **no row/range field**.

The embedding branch (`policy.py:62-71`) is:

```python
if component == "embedding" or output_head_or_tied_embedding:
    ...
    return TensorPolicy("convert", "CQ4", 4, component, False, "orthogonal", layout=layout)
```

It returns `CQ4`/`4` **unconditionally and ignores `user_bits`**. For this Qwen3
bundle the embedding component is exactly `"embedding"`, so `--embedding-bits 3`
(or any value) still produces a CQ4 tensor — the flag is a no-op here. There is
also no way via the CLI to request FP16 for the embedding: a `fallback`/`FP16`
policy can only be produced by the hardcoded branches (norms, biases, non-2D,
positions, etc.).

### Where row-range support would have to be added

1. **Policy.** Extend `TensorPolicy` and `policy_for_tensor`
   (`model_adapters/policy.py:8-17`, `:20`) to carry a row range / segment list
   (e.g. `segments: [(start, end, precision, bits)]`), or add a user-supplied
   `name -> segments` rules file and a matching `--precision-map` CLI flag in
   `cli.py:build_parser()` (`:584-611`).
2. **Emit/quantize.** In the convert emit loop (`cli.py:487-535`), split the
   tensor into row segments, call the quantizers per segment
   (`quantization/cq.py:quantize_orthogonal` `:296`,
   `quantize_hadamard` `:245`) and write each segment
   (`quantization/cq.py:write_cq_tensor` `:321`; FP16 segments go through
   `cli.py:_save_fallback_tensor` `:198`). This means a per-segment file layout,
   or a new container.
3. **Runtime.** Even with (1)+(2), the `.weights` header and graph buffer carry a
   single precision, and `bind_mmap_weights` rejects whole-file precision
   mismatch (`cactus-graph/src/io.cpp:539-544`) — so mixed-precision rows inside
   one tensor are **not representable at runtime** without an accompanying engine
   change (see #181). A cleaner fork-free alternative for the epic: keep the whole
   table at one precision — either accept CQ4 for the memory rows (train them
   quantization-aware) or hold the **entire** embedding tensor at FP16
   (`151936*1024*2 ≈ 311 MB` vs the current ≈ 80 MB, i.e. ≈ +231 MB bundle size).

**Consequence for the epic:** Phase 1's planned "N embedding rows held at FP16"
is not achievable via the existing convert options, and cannot be done by the
current runtime format even with a convert patch. It has to be re-scoped to
"whole-tensor precision" (CQ4-trained rows, or FP16 whole table) or to
"convert + engine fork for mixed precision".

---

## #183 (P0.4) — Tied embeddings, and the safe reserved vocab IDs

**Answer: embeddings are tied (verified against the actual weight file, not just
`config.json`). The unused, tokenizer-unreachable vocab range is
`[151670, 151935]` — N = 266 IDs — and that is the reservation pool.**

### Tied-embedding verification (weight file + manifest, not the config flag)

Parsed the safetensors header of
`outputs/merged-model-distill-v2-rationale/model.safetensors`:

```
num tensors: 310
model.embed_tokens.weight   shape [151936, 1024]  dtype BF16   <- the only embedding/head tensor
lm_head.weight              ABSENT
config.json tie_word_embeddings: true
```

`conversion_manifest.json` (deployed bundle) likewise has **310 entries, 0
containing `lm_head`**, with `components: {embedding: 1, language: 309}`. The
single embedding entry is:

```
hf_name      model.embed_tokens.weight
output_file  token_embeddings.weights
policy       convert      precision  CQ4      shape [151936, 1024]
```

So `lm_head` is tied to `model.embed_tokens.weight` — one physical table, used
both for input embedding lookup and (transposed) for output logits. Two
implications: (a) baking reserved rows also changes the output head for those IDs,
and (b) the reserved IDs must be suppressed from sampling or the model could emit
memory tokens into its visible text (the engine already has a vocab-bias hook,
`Model::set_vocab_bias` / `clear_vocab_bias`, `engine.h:760-763`; that is the
lever for M1 #187).

### Enumeration of unused vocab IDs

Analyzed the actual tokenizer of both the merged model and the deployed bundle
(they are identical):

| Source | IDs | Count |
|---|---|---|
| `vocab.json` base BPE vocab | `0 … 151642` | 151643 |
| `tokenizer.json` / `added_tokens.json` special/added tokens | `151643 … 151669` | 27 |
| **union (everything the tokenizer can emit)** | `0 … 151669` | 151670 |
| `config.json vocab_size` (embedding rows) | `0 … 151935` | 151936 |
| **unused / unreachable** | **`151670 … 151935`** | **266** |

- Every base BPE token (including all 256 byte fallbacks) lives in `0…151642`.
- The only added tokens are the 27 in `added_tokens.json`, max id `151669`
  (`<|PAD_TOKEN|>`); see also bundle `special_tokens.json` (`vocab_size: 151670`).
- Therefore no tokenizer call can produce any id in `151670…151935`, so no such
  id can appear in training/eval data built with this tokenizer.
- The rows physically exist in the tied table and are non-zero but untrained-scale
  (bf16 row L2 norms ≈ 0.30–0.36 across the whole range, vs ≈ 0.9–1.0 for real
  base rows), consistent with never having been indexed. Overwriting them is safe.

**Concrete reservation list: 266 IDs, `151670..151935` inclusive.** A smaller
working subset (e.g. `151670..151701` for N=32, matching PersonaMem's N=32) is the
first N of these. Note these IDs are reachable as raw prompt tokens via the
exported `cactus_submit` path (#180), which is what makes the reserved-token trick
addressable without any tokenizer change.

**Caveat for Phase 1:** because the head is tied, reserved IDs may receive
competitive logits at generation time. A logit-bias/ban on `151670..151935`
(`set_vocab_bias`, or the banned-token sampling planned in #187) is required, and
should be verified on-device — per the #172 landmine, do not assume the bias hook
is wired until a real generate confirms it.
