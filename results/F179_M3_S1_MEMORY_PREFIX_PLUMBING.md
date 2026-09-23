# F179 / M3 — #224 (S1): memory-prefix plumbing (write_embeds_row lift + FFI + bindings)

Milestone #14 (Epic #179), parent #223, first step of its S1–S6 work order.
Engine fork `cactus-sdk-v210-suppress`, branch `v2.1.0-suppress-patch`.
Commit **341d09b** (parent `ae33052`, which carries #212's
`CACTUS_QWEN_EMBED_LAYERS`; #205's suppression patch `82d8b30` is further back
and untouched).

**Pure plumbing: state + setter + FFI + Python binding + build. No
prefill/generation behavior changes. Disarmed (rows=0, the default) is a
complete no-op — the only new state is an empty byte buffer and three
zero-initialised size fields, and nothing reads them unless armed.**

## 1. Lambda → reusable helper

`build_lm_encoder_outputs_dynamic_media` defined a local capturing lambda
`write_media_row_at` (`model.cpp:3738` pre-change) that wrote one row into the
media chunk's token-aligned `inputs_embeds` buffer, doing row-offset math,
token-alignment/bounds checks, and FP32→target-precision conversion via
`write_typed_buffer`. It captured `lm_encoder_media_chunk_` (a member) plus two
locals (`media_embeds_idx`, `media_chunk_tokens`), so it could not be lifted
verbatim.

It is now a **static private member** of `Model`:

```cpp
static void Model::write_embeds_row(Component& comp, int embeds_idx,
                                    size_t chunk_tokens, size_t row,
                                    const uint8_t* row_data, size_t row_bytes,
                                    Precision row_precision);
```

declared next to `write_media_embeds_row` (`engine.h:969`) and defined before
`build_lm_encoder_outputs_dynamic_media` (`model.cpp:3638`). The body is the
lambda body with the three captures turned into explicit parameters and the one
mechanical adaptation that follows from the capture being a pointer member
(`lm_encoder_media_chunk_->x` → `comp.x`); the three throw branches and their
string literals are preserved verbatim.

**Deviation, flagged:** the prompt's signature was `static void
write_embeds_row(Component&, ...)`. `Model::Component` is a *private* nested
type (`engine.h:824`→`1188`), so a namespace-scope free function cannot name it
without a friend declaration. A static member function is the reusable form that
compiles; it still takes every input explicitly (no captures).

## 2. Existing media call site — unchanged behavior confirmed

Both call sites in the `can_chunk` branch (audio and image) now call the helper
with the same modality arguments plus the three now-explicit context args:

```cpp
write_embeds_row(*lm_encoder_media_chunk_, media_embeds_idx, media_chunk_tokens,
                 i, audio_feature.row(audio_idx), audio_feature.row_bytes, audio_feature.precision);
```

Because this is a pure extraction and the media path needs a VLM/audio bundle to
exercise at runtime (not available in this repo's text-only bundles), the
regression check is a **static equivalence proof**,
`qwen3-finetune/scripts/check_write_embeds_row_refactor.py`. It extracts the
pre-refactor lambda body (from the introducing commit's parent) and the new
function body, applies the documented capture→parameter substitutions, and
asserts they are line-for-line identical; it also asserts both call sites pass
identical modality expressions with only the three context args prepended.
Result:

```
base ref: 341d09b5dcc195025df680f687d8719e3826af3^
lambda body lines: 26 | function body lines: 26
call sites: old=2 new=2
  media row: (i, audio_feature.row(audio_idx), audio_feature.row_bytes, audio_feature.precision)
  media row: (i, image_feature.row(image_idx), image_feature.row_bytes, image_feature.precision)
PASS: write_embeds_row is a verbatim lift; media call sites unchanged in effect.
```

## 3. New API surface

**Model state** (`engine.h`, private, beside `media_placeholder_token_id_`):
`std::vector<uint8_t> memory_prefix_; size_t memory_prefix_rows_ = 0;
size_t memory_prefix_dim_ = 0;` — default disarmed.

**Model methods** (public):
- `set_memory_prefix(const float* data, size_t n_rows, size_t dim)` — stores the
  payload as raw **FP32 bytes** (no pre-conversion; S3's row-overwrite converts
  to the target buffer precision via `write_embeds_row`). `n_rows == 0` disarms.
  **Hard-errors** (`std::invalid_argument`) on a null pointer, `dim == 0`,
  `dim != config_.hidden_dim`, or a size overflow — a wrong-width prefix would
  silently corrupt every downstream row, so clamping was rejected.
- `clear_memory_prefix()` — frees the buffer, resets rows/dim to 0.
- `memory_prefix_rows()/dim()/bytes()` — const readback accessors (used by G2).

**FFI** (`cactus_engine.h`, additive only — no existing symbol changed):
```c
int cactus_set_memory_prefix(cactus_model_t model, const float* data, int32_t n_rows, int32_t dim);
int cactus_clear_memory_prefix(cactus_model_t model);
```
Implemented in new `src/memory_prefix.cpp`, matching `cactus_embed`'s pattern
exactly: null/invalid → `CACTUS_LOG_ERROR` + `-1`; body in try/catch with
`last_error_message = e.what()`. Return 0 on success.

**Python** (`python/cactus/bindings/cactus.py`): `cactus_set_memory_prefix(model,
vectors, n_rows, dim)` (accepts a flat or nested row-major payload, validates
length client-side, raises `RuntimeError` via `cactus_get_last_error` on engine
failure) and `cactus_clear_memory_prefix(model)`, registered next to
`cactus_embed`'s argtypes.

## 4. G2 — instrumented readback (this issue's gate)

Pure host-side storage; no FP16 conversion happens until S3, so **bit-exact
memcmp is the bar**. `cactus-engine/tests/test_memory_prefix.cpp` drives the FFI
entry points against a default-constructed `Model` handle (no bundle needed) and
reads back through the accessors:

| test | checks |
|---|---|
| default_disarmed | fresh model: rows=0, dim=0, empty buffer (no-op default) |
| set_readback_bit_exact | 8×1024 FP32 payload with non-trivial bit patterns → `memcmp == 0`, rows/dim exact |
| clear_disarms | set → clear → rows/dim 0, buffer empty |
| nrows_zero_disarms | direct `Model::set_memory_prefix(data, 0, dim)` disarms (not an error) |
| dim_mismatch_rejected | dim=1023 → FFI returns -1, no partial state |
| null_rejected | null data / null model / n_rows=0 via FFI / null clear all rejected |
| overwrite_roundtrip | second set fully replaces the first, byte-for-byte |

Result: **7/7 pass** (clean rebuild from an empty `build/`):

```
✓ PASS │ default_disarmed
✓ PASS │ set_readback_bit_exact
✓ PASS │ clear_disarms
✓ PASS │ nrows_zero_disarms
✓ PASS │ dim_mismatch_rejected
✓ PASS │ null_rejected
✓ PASS │ overwrite_roundtrip
✓ All 7 tests passed!
```

**Python-binding E2E** (stronger: a real bundle handle), run under qemu with the
built glibc arm64 `.so` against `outputs/r179-m1/p193-suppress-deploy`:
`cactus_init` → flat set → nested set → wrong-length `ValueError` → wrong-dim
`RuntimeError("memory prefix dim 1023 does not match model hidden dim 1024")` →
clear → destroy. **PASS**, which also confirms the added state/methods did not
regress model loading.

## 5. Build

`cactus build --python` in `cactus-arm64-convert:built` (`--platform
linux/arm64`, qemu), **clean rebuild from an empty `build/`: 4m03s**. The
resulting `libcactus_engine.so` is ELF 64-bit ARM aarch64 and exports
`cactus_set_memory_prefix` / `cactus_clear_memory_prefix` (alongside
`cactus_embed`). **No new warnings**: 10 warnings total, all pre-existing in
untouched files (`attention_hybrid.cpp`, `builder.cpp`, `metal_plan.cpp`,
`bpe.cpp`, `telemetry_impl.cpp`); none in `model.cpp`, `memory_prefix.cpp`,
`engine.h`, or the test.

## Reproduce

```bash
FORK=[LOCAL_PATH]

# static refactor equivalence (qwen3-finetune)
python scripts/check_write_embeds_row_refactor.py

# clean build + G2 (arm64 via qemu)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk \
  cactus-arm64-convert:built bash -lc '
    rm -rf /src/cactus-sdk/cactus-engine/build
    cd /src/cactus-sdk && cactus build --python
    cd /src/cactus-sdk/cactus-engine
    g++ -std=c++20 -O1 -pthread -I . -I libs -I ../cactus-graph \
      -I ../cactus-kernels -I ../cactus-kernels/src -I ../cactus-kernels/libs \
      tests/test_memory_prefix.cpp tests/test_utils.cpp \
      build/libcactus_engine.a -o /tmp/t -lcurl -ldl -lm && /tmp/t'
```

Artifacts: engine commit `341d09b` (7 files, +363/−32);
`cactus-engine/tests/test_memory_prefix.cpp`;
`qwen3-finetune/scripts/check_write_embeds_row_refactor.py`.
