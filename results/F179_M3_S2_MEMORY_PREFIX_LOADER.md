# F179 / M3 — #225 (S2): manifest keys + loader + A1 vector file + loader gate

Milestone #14 (Epic #179), parent #223, second step of its S1–S6 work order.
Engine fork `cactus-sdk-v210-suppress`, branch `v2.1.0-suppress-patch`, commit
**1ededf0** (parent `341d09b` = #224 S1).

**S2 arms a bundle with a static memory prefix at model init from the manifest
alone (zero app-side code for tier A1), ships the real 8×1024 post-transform A1
vector file, and proves the loader is byte-exact when armed and a true no-op
when the keys are absent. G1 (the live `inputs_embeds` readback) is #226/S3's
gate, not this one.**

## 1. Manifest keys + loader

Followed the existing `load_manifest()` pattern exactly (no new JSON code).
Added, right before `load_manifest` returns:

```cpp
const std::string prefix_file = read_string_key("memory_prefix_file");
const size_t prefix_rows = parse_size_or_zero(read_string_key("memory_prefix_rows"));
```

Both keys are **string-encoded** — `read_string_key` only returns JSON strings,
and the composition `parse_size_or_zero(read_string_key(...))` is the same one
`media_min_new_tokens` uses (the issue called `memory_prefix_rows` an int; in
this manifest format ints read through `read_string_key` are stored as JSON
strings, e.g. `"memory_prefix_rows": "8"`). The apply script and fragment emit
that form.

Policy when the keys are present (documented choice):
- **Both must be present together.** Either one alone is a malformed bundle.
- The file is resolved as `fs::path(bundle_dir_) / prefix_file` (same convention
  as `manifest.json` itself) and must be exactly `rows * config_.hidden_dim *
  sizeof(float)` bytes.
- **Hard-fail load** (`return false`, logged) on: keys not paired, unreadable
  file, size mismatch, or `set_memory_prefix` throwing. Rationale: a bundle that
  declares a memory prefix it cannot deliver is a packaging bug; silently
  running disarmed would produce a model that behaves differently than it
  claims to, which is exactly the kind of quiet wrongness the A1 tier exists to
  avoid. `set_memory_prefix`'s own `dim != hidden_dim` hard-error is therefore
  also caught here at load time.
- **Neither key present** (the common case): the block is skipped entirely,
  `set_memory_prefix` is never called, the model stays at S1's `rows=0` default.
  This is a true no-op.

## 2. A1 vector file (Part 3)

`scripts/gen_memory_prefix_a1.py` does fit → S0 select → post-transform in one
pass, reusing `scripts/lpm_quant_gap.py`'s #216 machinery unchanged
(`fit_val_slots` → `s0_select` → `l1_transform_selected`).

- **Persona/budget choice:** val persona **offset 0, index 0**
  (`persona_id=gemma4:clq6qtg9p002o8ipv9jcfbqsi`), issue-default fit budget
  **40/128/192**, seed 0 — the same test persona #214's effect probe used, for
  continuity with the #214/#216/#222 evidence chain.
- **Object:** `out_ln[h](v + mlp(v))` per head — the #216-corrected
  post-transform object, **not** `bake_vectors`' raw pre-transform output (the
  falsified #211/#212 object) and **not** norm-clamped (#222 found no viable
  clamp; this path never touches the tied embedding table, so the true object is
  shipped).
- **Nondeterminism, stated explicitly:** slot fitting is GPU-nondeterministic
  run-to-run (#216). This file is a **new shipped artifact**, not a
  bit-reproduction of any prior JSON. One run was committed to; its output is
  pinned by sha256 in the manifest fragment.

Generated file (raw, no header):

| property | value |
|---|---|
| path | `outputs/r179-m1/s2a/a1_memory_prefix_p0.f32` |
| format | raw FP32, row-major, H=8 rows × d_model=1024 floats, no header |
| size | 32768 bytes (32 KB) |
| sha256 | `2518df55533a8a8f536065b50d2fbe22be0452787eeeee683e1b5dd029e33e72` |
| norm mean / max | 36.93 / 37.83 (the expected #216 norm~37; unclamped) |
| S0 slots | `[29, 18, 20, 25, 25, 1, 22, 24]` |

Checked-in manifest fragment: `results/F179_M3_S2_A1_MANIFEST.json` (the two
keys + provenance/sha256). `scripts/apply_memory_prefix_bundle.py` consumes it
to build an armed (or disarmed/malformed) bundle variant as a cheap symlink tree
(no weight copy), placing the `.f32` at `components/a1_memory_prefix_p0.f32` —
this is the armed bundle #226/S3 consumes.

## 3. Loader-correctness gate

`cactus-engine/tests/test_memory_prefix_loader.cpp` drives `cactus_init` on four
bundle variants and reads the stored prefix back through #224's
`memory_prefix_rows()/dim()/bytes()` accessors (host-side, no inference):

| check | bundle | expectation | result |
|---|---|---|---|
| `armed_memcmp` | symlink tree + keys + `.f32` | rows/dim = 8/1024, `memory_prefix_bytes()` memcmp the source file | **PASS** |
| `disarmed_noop` | symlink tree, keys absent | rows=0, dim=0, empty buffer (no setter call) | **PASS** |
| `malformed_hard_fail` | keys present, file missing | `cactus_init` returns null | **PASS** |
| `ordinary_regression` | real `p193-suppress-deploy` (no keys) | loads, tokenizer present, rows=0 | **PASS** |

`ordinary_regression` runs against the **real** bundle, not a synthetic
manifest, to catch anything a hand-made test bundle might miss. Result on the
required **clean rebuild from an empty `build/`** (4m16s, qemu arm64):

```
✓ PASS │ armed_memcmp
✓ PASS │ disarmed_noop
✓ PASS │ malformed_hard_fail
✓ PASS │ ordinary_regression
✓ All 4 tests passed!
```

Build: `cactus build --python` in `cactus-arm64-convert:built`, clean rebuild;
**10 warnings, all pre-existing in untouched files** (`attention_hybrid.cpp`,
`builder.cpp`, `metal_plan.cpp`, `bpe.cpp`, `telemetry_impl.cpp`) — none in
`model.cpp` or the new test. The gate was re-run independently from the clean
build (and the bundle variants regenerated from scratch beforehand).

## 4. What #226 (S3) consumes

- Armed bundle on disk: `outputs/r179-m1/s2a/gate_bundles/armed` (keys in
  `components/manifest.json`, `.f32` at `components/a1_memory_prefix_p0.f32`).
- Source FP32 for G1's comparison:
  `outputs/r179-m1/s2a/a1_memory_prefix_p0.f32` (sha256 pinned in the fragment).
- The loader arms `Model::memory_prefix_` at init; S3 still needs to implement
  the row-overwrite that copies it into the first prefill chunk's
  `inputs_embeds` via the #224 `write_embeds_row` helper, then run **G1**
  (cos ≥ 0.99999 AND max-abs-delta below the FP16 bound on the live buffer) —
  `memory_prefix_bytes()` is the host-side stored FP32, not the engine buffer,
  so G1 needs its own test-only export.

## Reproduce

```bash
PY=[LOCAL_PATH]
FORK=[LOCAL_PATH]

# 1. generate the A1 file + provenance
$PY scripts/gen_memory_prefix_a1.py \
  --out outputs/r179-m1/s2a/a1_memory_prefix_p0.f32 \
  --meta-out outputs/r179-m1/s2a/a1_memory_prefix_p0.meta.json

# 2. build the bundle variants the gate (and #226) use
$PY scripts/apply_memory_prefix_bundle.py --mode armed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/armed \
  --prefix-file outputs/r179-m1/s2a/a1_memory_prefix_p0.f32
$PY scripts/apply_memory_prefix_bundle.py --mode malformed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/malformed
$PY scripts/apply_memory_prefix_bundle.py --mode disarmed \
  --src-bundle outputs/r179-m1/p193-suppress-deploy \
  --out-bundle outputs/r179-m1/s2a/gate_bundles/disarmed

# 3. clean build + gate (qemu arm64)
QWEN=$(pwd)
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk -v "$QWEN":"$QWEN":ro \
  -e CACTUS_TEST_BUNDLE_ARMED="$QWEN/outputs/r179-m1/s2a/gate_bundles/armed" \
  -e CACTUS_TEST_BUNDLE_DISARMED="$QWEN/outputs/r179-m1/s2a/gate_bundles/disarmed" \
  -e CACTUS_TEST_BUNDLE_MALFORMED="$QWEN/outputs/r179-m1/s2a/gate_bundles/malformed" \
  -e CACTUS_TEST_BUNDLE_ORDINARY="$QWEN/outputs/r179-m1/p193-suppress-deploy" \
  -e CACTUS_TEST_PREFIX_FILE="$QWEN/outputs/r179-m1/s2a/a1_memory_prefix_p0.f32" \
  cactus-arm64-convert:built bash -lc '
    rm -rf /src/cactus-sdk/cactus-engine/build
    cd /src/cactus-sdk && cactus build --python
    cd /src/cactus-sdk/cactus-engine
    g++ -std=c++20 -O1 -pthread -I . -I libs -I ../cactus-graph \
      -I ../cactus-kernels -I ../cactus-kernels/src -I ../cactus-kernels/libs \
      tests/test_memory_prefix_loader.cpp tests/test_utils.cpp \
      build/libcactus_engine.a -o /tmp/t -lcurl -ldl -lm && /tmp/t'
```

Artifacts: engine commit `1ededf0` (`model.cpp` +53,
`tests/test_memory_prefix_loader.cpp`); qwen3-finetune:
`scripts/gen_memory_prefix_a1.py`, `scripts/apply_memory_prefix_bundle.py`,
`results/F179_M3_S2_A1_MANIFEST.json`, and the generated
`outputs/r179-m1/s2a/a1_memory_prefix_p0.f32` (+ `.meta.json`).
