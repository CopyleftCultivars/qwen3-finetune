# F179 / M3 — #231 Phase 1: does Qwen3-1.7B load + generate + accept a 2048-wide memory prefix on the existing engine?

Parent #179; sibling to #216/#222/#223/#230. Fork `cactus-sdk-v210-suppress`,
branch `v2.1.0-suppress-patch`, engine commit **a82155c** (#226 S3). This note
covers **Phase 1 only** (bundle load / generate / injection plumbing); phases 2–4
(retrain the LPM module at d_model=2048, etc.) are not started here.

**Verdict: Phase 1 PASS — but only for a bundle built in the 0.6B component
format.** With a Qwen3-1.7B bundle built through the **legacy Qwen3 graph
builder** (the same 6-component layout as the 0.6B bundles), the existing engine
loads it, generates coherent text, reports `hidden_dim = 2048` (confirmed
empirically), and **accepts an arbitrary 8×2048 memory prefix via
`cactus_set_memory_prefix` and generates without crashing** — so #223's
`inputs_embeds` injection is backbone-width-agnostic as designed.

**But the documented acquisition path is broken, and this changes Phase 2:** the
`cactus download` path cannot fetch the hosted Qwen3-1.7B bundle, and the fork's
**default** transpile path produces a structurally *incompatible* bundle
(`family=generic_text`, single `decoder_full_context` component, no
`inputs_embeds`) on which arming **hard-fails**. Phase 2 must standardise on the
legacy 6-component build (details below); otherwise phases 3–4 would train
against a bundle the memory mechanism cannot use.

## 1. Obtaining the bundle — the documented path does not work

- Hosted repo `huggingface.co/Cactus-Compute/Qwen3-1.7B` exists (completion /
  tools / embeddings tags) and publishes exactly two archives:
  `weights/qwen3-1.7b-int4.zip` (1,006,051,493 B) and
  `weights/qwen3-1.7b-int8.zip` (1,727,531,959 B). All tags (`v1.6`…`v1.14`) and
  `main` carry the same names.
- `cactus download Qwen/Qwen3-1.7B` **cannot fetch it.** The fork's archive-name
  parser (`python/cactus/cli/utils.py`, `CQ_VARIANT_RE = -cq(\d+)$`) only accepts
  `-cqN`; the hosted names are `-int4`/`-int8`, so `list_hf_cq_archives()` returns
  empty and `resolve_archive` raises `No CQ archives found` (`cactus download`
  then falls back to a local build).
- The hosted archive is **flat**: 317 entries, no `components/` directory, no
  `components/manifest.json` (the fork's `validate_extracted_bundle` requires
  one). It is *converted CQ weights* (`config.txt`, `layer_*.weights`,
  `token_embeddings.weights`, `output_norm.weights`, tokenizer sidecars,
  `chat_template.jinja2`), **not a runnable bundle**. It must be transpiled into
  a runtime bundle before the engine can load it.
- (Workaround used here: fetched `qwen3-1.7b-int4.zip` directly with
  `huggingface_hub` and extracted it — the payload size matches the LFS metadata
  exactly.)

## 2. Building a runnable bundle — two paths, one incompatible

Two transpile paths exist in this fork and they produce **different runtime
formats**:

| path | output | engine compatibility |
|---|---|---|
| default `cactus convert` (→ `cactus.transpiler` generic profile) | `family=generic_text`, **one** component `decoder_full_context`, `runtime_plan.json`, `runtime_execution_strategy=full_context_recompute`, `runtime_cache_persistence=none` | loads, but **incompatible** with #223 |
| legacy Qwen3 graph builder (`cactus.transpile.hf_model`, `family=qwen3`) | **6** components, `component_order=[lm_encoder_text_chunk, decoder_prefill_chunk, decoder_embed_chunk, lm_encoder_step, decoder_media_step, decoder_step]` | **matches the 0.6B bundles; #223 injection works** |

The default path was run first. Its `components/manifest.json` has
`component_order: null` and a single component whose FULL_CONTEXT_TEXT input
component has **no `inputs_embeds`**, so arming a prefix makes generation
hard-fail:

```
[ERROR] [complete] Exception: memory prefix armed but FULL_CONTEXT_TEXT input
component has no inputs_embeds input to overwrite
```

Its unarmed generation is also degenerate and pathologically slow under qemu
(6 generated tokens took 449 s and decoded to the prompt verbatim:
`ids=[785,1850,64291,369,40513,374] == tokenizer(prompt)`), consistent with the
`full_context_recompute` / no-KV-cache plan. This bundle is **not usable** for the
epic's mechanism.

The legacy builder was then run **inside the arm64 conversion container**
(`cactus-arm64-convert:built`, which is pinned to torch 2.8.0 / transformers
5.5.4 — the fork's `python/requirements.txt`). Note the engine only builds for
arm64 (NEON; `cactus-kernels/CMakeLists.txt:36` hardcodes
`-march=armv8.2-a…`), and #223's container build overwrote the host
`cactus-engine/build/` shared lib with the arm64 one — so the transpiler must run
in the arm64 container (qemu), not on the host. Invocation (worked, ~10 min under
qemu with `--low-memory-load`, capture of all 6 components):

```
docker run --platform linux/arm64 -v FORK:/src/cactus-sdk \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/src/cactus-sdk/python \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  cactus-arm64-convert:built python3 -m cactus.transpile.hf_model \
    --model-id Qwen/Qwen3-1.7B \
    --weights-dir  /src/cactus-sdk/weights/qwen3-1.7b-cq4-conv \
    --artifact-dir /src/cactus-sdk/weights/qwen3-1.7b-legacy \
    --low-memory-load --skip-execute --skip-reference-compare
```

A runnable dir was assembled from the legacy `components/` + the converted flat
weights (`weights/qwen3-1.7b-legacy-rt`, 1.1 GB). Full conversion stats: 312
tensors — 198 CQ4, 113 FP16 (norm-tensor fallback), 1 ignored; total 1.06 GB.

## 3. Engine smoke test (the decisive cheap check)

Run in the arm64 container against `weights/qwen3-1.7b-legacy-rt`, greedy
(`temperature=0`), `max_tokens=16`, with
`CACTUS_DISABLE_CLOUD_HANDOFF=1`.

| check | result |
|---|---|
| `cactus_init` | **OK** (≈1.9 s) |
| unarmed generation (3 prompts) | coherent text; **0** reserved-ID admissions, **0** NaN, **0** empty |
| `hidden_dim` probe via FFI width | `set_memory_prefix(dim=2048)` → rc=0; `dim=1024` → engine error `memory prefix dim 1024 does not match model hidden dim 2048` → **hidden_dim is 2048, empirically** |
| arbitrary 8×2048 prefix armed via `cactus_set_memory_prefix` | **arms OK and generates**; 3 prompts, **0** reserved, **0** NaN, **0** empty |

Generation samples (`max_tokens=16`, truncated by the token budget):

| prompt | unarmed | armed with arbitrary 8×2048 prefix (random, plumbing only) |
|---|---|---|
| The best fertilizer for tomatoes is | `The best fertilizer for tomatoes is typically a **balanced, water-soluble fertilizer` | `The "best fertilizer for tomatoes" can vary depending on the specific needs of the` |
| Tell me about Korean natural farming inputs | `Korean natural farming inputs are a set of practices and techniques used in agriculture that` | `Korean natural farming inputs are a set of practices that emphasize the use of natural` |
| Describe how to make biochar | `To make **biochar**, you can follow these steps:\n\n### 1.` | `Biochar is a carbon-rich material produced from biomass through thermal decomposition (pyro` |

The output is **not** about memory content (the vectors are random, as intended) —
this is purely the plumbing check the issue requested. **#223's injection
mechanism is backbone-width-agnostic when the bundle uses the 0.6B component
format.**

Determinism: the smoke script was re-run once against the same bundle; output
byte-identical (`sha256 539d5b20cd4ce5b1f67c3d37c3245c879f67d7e285de8da02d6c310e08ad12e5`,
artifact `results/F179_M3_BACKBONE_SCALE_PHASE1_SMOKE.json`).

## 4. hidden_dim and component structure (direct inspection, not assumption)

- Real Qwen/Qwen3-1.7B `config.json` (HF cache): `hidden_size=2048`,
  `num_hidden_layers=28`, `num_attention_heads=16`, `num_key_value_heads=8`,
  `intermediate_size=6144`, `tie_word_embeddings=true`, `vocab_size=151936`,
  `model_type=qwen3`.
- Bundle `config.txt`: `hidden_dim=2048`, `num_layers=28`.
- Engine `config_.hidden_dim` is read from the bundle (same mechanism #224/#225
  established for 0.6B); confirmed at runtime by the FFI width rejection above.
- `components/manifest.json` (legacy build) `component_order` is **identical** to
  the 0.6B bundles' six components and `family=qwen3` — so #223's per-component
  injection (`lm_encoder_text_chunk` → `decoder_prefill_chunk`) applies
  unmodified. The default build's single `decoder_full_context` component does
  **not**.

## 5. What this changes for Phases 2–4

1. **Phase 2 must target the legacy 6-component bundle format, not whatever
   `cactus download`/`cactus convert` yields by default.** The default generic
   bundle cannot accept the memory prefix at all, so training an LPM module
   (Phase 3) and re-running the #228/#230 effect probe (Phase 4) against it would
   be wasted compute.
2. **The bundle-acquisition toolchain needs a small fix before phases 2–4 are
   cheap:** either (a) extend `cactus download`'s archive-name parser to accept
   `-int4`/`-int8` (and publish/inspect a manifest), or (b) add/keep a
   Qwen3-1.7B registered profile so the default transpiler emits the 6-component
   layout, or (c) standardise on the legacy builder invocation above. Until then,
   building the 1.7B bundle requires the arm64 container + legacy builder
   (~10 min under qemu), which is acceptable but worth automating.
3. **The engine itself needs no change.** No engine patch was required for load,
   generate, or the 2048-wide prefix; the workaround needed was in the *bundle
   build*, not the engine. This preserves the phase-4 assumption that the
   #223 mechanism can be reused.

## 6. Recommendation (go/no-go for Phase 2)

**GO — with the scope change above.** Phase 1 shows the core plumbing is sound at
2048 width, which was the biggest unknown. Before spending Phase 3 compute,
either fix/extend the bundle build so the 1.7B runtime bundle is reproducible in
the 0.6B format, or document the legacy-build command as the standard. Do **not**
build a 1.7B LPM module against the default generic runtime.

Phase 2 (Qwen3-1.7B `LPMConfig` profile at `d_model=2048` + encoder forward-shape
smoke test) can proceed once the bundle-format decision is made; it was not
started here.

## 7. Reproduce

```bash
QWEN=[LOCAL_PATH]
FORK=[LOCAL_PATH]
docker run --privileged --rm tonistiigi/binfmt --install arm64   # after a reboot

# 0. the existing #205/#212/#223 engine build (arm64) is at $FORK/cactus-engine/build/

# 1. hosted bundle (int4, 1.0 GB) -> flat converted weights
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("Cactus-Compute/Qwen3-1.7B","weights/qwen3-1.7b-int4.zip",
                local_dir="/tmp/qwen17"); print("fetched")
PY
mkdir -p "$FORK/weights/qwen3-1.7b-cq4-conv" && \
  (cd "$FORK/weights/qwen3-1.7b-cq4-conv" && unzip -q /tmp/qwen17/weights/qwen3-1.7b-int4.zip)

# 2. build the 6-component runtime bundle in the arm64 container (see §2 command),
#    then assemble: legacy components/ + converted flat weights -> weights/qwen3-1.7b-legacy-rt

# 3. smoke test
docker run --rm --platform linux/arm64 -v "$FORK":/src/cactus-sdk \
  -v "$QWEN/scripts":/work/scripts:ro -v /tmp/out:/work/out \
  -e PYTHONPATH=/src/cactus-sdk/python \
  -e CACTUS_LIB_PATH=/src/cactus-sdk/cactus-engine/build/libcactus_engine.so \
  -e CACTUS_DISABLE_CLOUD_HANDOFF=1 --entrypoint python3 cactus-arm64-convert:built \
  /work/scripts/backbone_scale_phase1_smoke.py \
    --bundle /src/cactus-sdk/weights/qwen3-1.7b-legacy-rt \
    --out /work/out/phase1_smoke.json --max-tokens 16
```

Artifacts: `results/F179_M3_BACKBONE_SCALE_PHASE1_SMOKE.json` (raw smoke records);
script `scripts/backbone_scale_phase1_smoke.py`. Bundle artifacts under the fork
`weights/` (`qwen3-1.7b-cq4-conv` generic, `qwen3-1.7b-legacy`,
`qwen3-1.7b-legacy-rt`) are not committed (fork is a separate repo).

## 8. Notes / caveats

- The default-path experiment used an intermediate host IR (torch 2.10 /
  transformers 4.57) plus identity-elimination of 29 `aten.alias` nodes, because
  the pinned Generator lacked that rule. That path is reported as a negative
  finding only; the positive Phase-1 result uses the legacy builder in the pinned
  container, so it is not affected by that workaround.
- qemu timings: init ≈1.9 s; ~35–45 s per 16-token greedy generation on the
  legacy bundle; the generic no-cache bundle took 449 s for 6 tokens.
- The prefix vectors in the armed test are random (seeded 231); they are a
  plumbing check, not a memory-content test (there is no trained 1.7B LPM module
  yet — that is Phase 3).
