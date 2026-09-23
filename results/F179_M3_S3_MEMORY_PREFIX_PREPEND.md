# F179 / M3 — #226 (S3): token-level prepend + prefill row overwrite + G1/G3

Milestone #14 (Epic #179), parent #223, third step of its S1–S6 work order.
Engine fork `cactus-sdk-v210-suppress`, branch `v2.1.0-suppress-patch`, commit
**a82155c** (parent `1ededf0` = #225 S2).

**The armed memory prefix is now injected end-to-end on the real p193 bundle:
N pad placeholders are prepended to the prefill token sequence and the first N
rows of the first chunk's `inputs_embeds` are overwritten with the armed
vectors. G1 (storage) PASSES with cos = 0.99999998 and max-delta inside the
FP16 half-ULP bound on all test prompts; G3 (directional HF parity) PASSES.**

## 0. Route mismatch found before editing (important)

The issue's three cited fresh-context sites are all
`DecodeRoute::FULL_CONTEXT_TEXT` (`context_tokens_.insert` +
`run_full_context_text`), but the armed p193-suppress-deploy bundle is
**`CACHED_STEP`**, not FULL_CONTEXT_TEXT. Verified directly: its
`component_order` is `[lm_encoder_text_chunk, decoder_prefill_chunk,
decoder_embed_chunk, lm_encoder_step, decoder_media_step, decoder_step]`, which
satisfies `has_chunked_prefill` (`model.cpp:797-807`) and selects `CACHED_STEP`.
`cactus_complete` for text-only prompts calls
`prefill_and_sample_first_token` → `run_chunked_prefill` → `execute_prefill_chunk`.

So the mechanism described (pad tokens + a chunk `inputs_embeds` row overwrite)
is inherently the **chunked-prefill** path, and the token prepend had to be
implemented there for G1/G3 to be reachable at all. Both routes were handled:
the CACHED_STEP prepend (the one the armed bundle exercises) and the three
FULL_CONTEXT_TEXT sites as the issue instructed. The FULL_CONTEXT_TEXT route
has no chunked `inputs_embeds` buffer in the text-encoder path, so
`run_full_context_text` now attempts an `inputs_embeds` overwrite and
**hard-errors** if the input component has none (no silently-unarmed run).

## 1. Insertion points (line numbers AS OF COMMIT a82155c, 2026-09-23)

`model.cpp` has drifted on every prior S-issue; these are pinned to a82155c.
The next editor must re-grep, not trust these blindly.

| what | location at a82155c |
|---|---|
| `execute_prefill_chunk` def | `model.cpp:2242` |
| row overwrite (after encoder copy, before mask+execute) | `model.cpp:2276` |
| G1 readback capture (after `graph->execute()`) | `model.cpp:2286` |
| `run_chunked_prefill` def | `model.cpp:2298` |
| H6 capacity hard-error | `model.cpp:2342` |
| H2 force-padded-chunk | `model.cpp:2393` |
| `run_full_context_text` def | `model.cpp:2474` |
| FULL_CONTEXT_TEXT overwrite / hard-error | `model.cpp:2503-2507` |
| `prefill_and_sample_first_token` def | `model.cpp:3337` |
| CACHED_STEP prepend (this fn) | `model.cpp:3388-3397` |
| `prefill` def | `model.cpp:3438` |
| CACHED_STEP prepend (`prefill`) | `model.cpp:3455-3459` |
| `decode` fresh scalar guard (H2) | `model.cpp:4790-4794` |
| helpers `prepend_memory_prefix_tokens` / `overwrite_prefill_prefix_rows` / `capture_memory_prefix_readback` | `model.cpp:538 / 543 / 555` |
| G1 readback accessors (engine.h) | `engine.h:833-844` |

The three FULL_CONTEXT_TEXT token-prepend sites are the `context_tokens_.empty()`
guards in `prefill_and_sample_first_token` (~`3364`), `prefill` (~`3446`), and
`decode` (~`4767`).

## 2. What was implemented

- **Token prepend** (`prepend_memory_prefix_tokens`): insert N copies of
  `config_.pad_token_id` (151669) at the front of the fresh token sequence.
  CACHED_STEP: applied to a local `prefill_tokens` copy at the two prefill entry
  points (and `cache_token_ids_` updated to match the KV rows). FULL_CONTEXT_TEXT:
  applied to `context_tokens_` when it was empty. No new position/offset
  arithmetic — position_ids, attention_mask, chunk arithmetic, `last_logit_position_`
  and KV length all keep reading the (now longer) token sequence.
- **Row overwrite** (`overwrite_prefill_prefix_rows`): in
  `execute_prefill_chunk`, when `processed == 0 && start_position == 0 &&
  memory_prefix_rows_ > 0`, for each row 0..N-1 call `write_embeds_row(chunk_comp,
  input_index(chunk_comp,"inputs_embeds"), chunk_tokens, row, memory_prefix_.data()
  + row*row_bytes, row_bytes, Precision::FP32)`. The `inputs_embeds` index is
  looked up per component (same pattern as the media path), never hardcoded.
- **H6**: hard error when `component_tokens < memory_prefix_rows_ + 1`.
- **H2**: an armed fresh prefill shorter than one chunk is **forced through a
  single padded chunk** so the overwrite runs; if the cache type cannot be
  padded (conv/recurrent/sliding-window), hard-error instead. A fresh scalar
  `decode()` (which uses `run_step` and can never inject) is **rejected** when
  armed. Rationale for forcing over rejecting on the prefill path: rejecting
  would make ordinary short prompts unusable on an armed bundle, and the padded
  chunk is already a supported code path (`pad_tail`); the padding is discarded
  when the cache moves to the decoder, so cache-length accounting is unchanged.
  The `decode()` scalar path is a different entry point that `cactus_complete`
  never uses for a fresh prompt (it uses `prefill_and_sample_first_token`), so
  rejecting there is safe and explicit.

## 3. Gate G1 — storage (decisive): PASS

`cactus-engine/tests/test_memory_prefix_g1.cpp` runs a real armed
`prefill_and_sample_first_token` for each test prompt, then reads rows 0..N-1 of
the first prefill chunk's `inputs_embeds` immediately after `graph->execute()`
(captured by `capture_memory_prefix_readback`) and compares to #225's source
FP32 file after the buffer's conversion. **Buffer precision confirmed FP16
(elem_bytes = 2), not assumed.**

| prompt | cos | max abs delta | FP16 half-ULP bound | pass |
|---|---|---|---|---|
| capital of France | 0.99999998 | 0.0018659 | 0.0048712 | ✓ |
| photosynthesis | 0.99999998 | 0.0018659 | 0.0048712 | ✓ |
| name/greet | 0.99999998 | 0.0018659 | 0.0048712 | ✓ |

ACCEPT (cos ≥ 0.99999 AND max-delta below the FP16 bound, all prompts): **PASS
3/3**. The identical readback across prompts confirms the rows are the fixed
memory vectors, independent of the prompt (C1/C8).

**Independent comparator:** `scripts/g1_compare_readback.py` re-derives cos and
max-delta from the persisted `.readback` dumps (the engine test writes them via
`CACTUS_TEST_G1_DUMP_DIR`) without importing the engine test. Re-run twice →
byte-identical output; re-run after the clean rebuild → identical to the
pre-rebuild run. PASS 3/3.

## 4. Gate G3 — directional parity vs HF: PASS

The issue's G3 accept text was corrupted; operationalized as: engine armed vs
disarmed first-token shift, HF next-token shift under the same A1 vectors vs
none, and directional consistency (the engine's armed token is weakly more
likely under the HF memory prefix than under no memory).

- Engine side (`tests/test_memory_prefix_g3.cpp`): armed vs disarmed greedy
  first token, raw prompt tokens (same tokenization as HF, no chat template).
- HF side (`scripts/g3_engine_vs_hf.py`): prepend the same A1 vectors to the
  same raw prompt token embeddings and take the next-token logits
  (`forward_ntp` geometry), memory vs none.

| metric | result |
|---|---|
| engine conditioning shift (armed ≠ disarmed first token) | 3/4 |
| HF conditioning shift (memory ≠ no-memory argmax) | 4/4 |
| engine armed token rank improves under HF memory | 3/4 |
| engine armed first == HF memory argmax | 2/4 |
| engine disarmed first == HF no-memory argmax | 3/4 |

ACCEPT (engine shift ≥ majority AND HF shift ≥ majority AND rank improvement ≥
majority): **PASS**. Parity is directional, not bitwise: the engine bundle is
CQ4-quantized while the HF reference is bf16, so argmax divergence is expected
(#193's quantization gap). On the two prompts where the engine shifted cleanly,
the engine armed first token **exactly equals** the HF memory argmax
(prompt 2: 358 = 358; prompt 3: 20678 = 20678) and the disarmed token equals the
HF no-memory argmax on prompt 2 — i.e. the same conditioning direction on both
sides. On the third shifting prompt the engine's armed token rank under HF
memory is 11 vs 29 under no memory (improved, not exact).

## 5. Notes / limitations

- The FULL_CONTEXT_TEXT overwrite path is implemented but **not exercised** (no
  full-context bundle in the workspace; the armed bundle is CACHED_STEP). It
  hard-errors rather than silently skipping if armed without an `inputs_embeds`
  input.
- The engine G3 path prepends to raw prompt tokens (matching `forward_ntp`); the
  app-level chat-template rendering path is S4/S6. Multi-turn session reuse and
  persona switch (H4: `cactus_reset` + clear/set) are S4's regression scope.
- `cactus_embed` is untouched (C5) — the prefix is only read by the text
  prefill/decode paths.

## Reproduce

```bash
PY=[LOCAL_PATH]
FORK=[LOCAL_PATH]
QWEN=$(pwd)

# clean build + G1 + G3 engine side (qemu arm64)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":"$QWEN":ro \
  -v /tmp/g1:/g1 -v /tmp/g3:/g3 \
  -e CACTUS_TEST_BUNDLE_ARMED="$QWEN/outputs/r179-m1/s2a/gate_bundles/armed" \
  -e CACTUS_TEST_BUNDLE_DISARMED="$QWEN/outputs/r179-m1/p193-suppress-deploy" \
  -e CACTUS_TEST_PREFIX_FILE="$QWEN/outputs/r179-m1/s2a/a1_memory_prefix_p0.f32" \
  -e CACTUS_TEST_G1_DUMP_DIR=/g1 -e CACTUS_TEST_G3_OUT=/g3/engine.txt \
  cactus-arm64-convert:built bash -lc '
    rm -rf /src/cactus-sdk/cactus-engine/build && cd /src/cactus-sdk && cactus build --python
    cd /src/cactus-sdk/cactus-engine
    for t in g1 g3; do g++ -std=c++20 -O1 -pthread -I . -I libs -I ../cactus-graph \
      -I ../cactus-kernels -I ../cactus-kernels/src -I ../cactus-kernels/libs \
      tests/test_memory_prefix_${t}.cpp tests/test_utils.cpp \
      build/libcactus_engine.a -o /tmp/test_${t} -lcurl -ldl -lm; done
    /tmp/test_g1; /tmp/test_g3'

# host-side comparators
$PY scripts/g1_compare_readback.py --dump-dir /tmp/g1 --prefix-file outputs/r179-m1/s2a/a1_memory_prefix_p0.f32
$PY scripts/g3_engine_vs_hf.py --engine-out /tmp/g3/engine.txt --prefix-file outputs/r179-m1/s2a/a1_memory_prefix_p0.f32
```

Artifacts: engine commit **a82155c**
(`model.cpp` +148, `engine.h` +20, `tests/test_memory_prefix_g1.cpp`,
`tests/test_memory_prefix_g3.cpp`); qwen3-finetune:
`scripts/g1_compare_readback.py`, `scripts/g3_engine_vs_hf.py`.

## Anchors for #227 (S4), as of a82155c

Same table as §1. S4's regression work will additionally need: `reset_cache`
(`model.cpp`), `cactus_reset` (`init.cpp:449`), and the disarmed bundle
`outputs/r179-m1/p193-suppress-deploy` vs the armed
`outputs/r179-m1/s2a/gate_bundles/armed`. Re-grep before trusting.
