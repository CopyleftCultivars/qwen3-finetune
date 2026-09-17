# F179 / M1 — Phase 1 Plumbing spike: reserved-token injection end to end

Milestone #12. Builds directly on `results/F179_M0_FEASIBILITY_RESULTS.md`; the
M0 findings are cited, not re-derived. Scope is plumbing only — no LPM training,
no encoder work.

Champion: **distill-v2-rationale / "distill-calibrated"**. The merged source
`outputs/merged-model-distill-v2-rationale` is byte-identical (safetensors md5
`2bf7044e…`) to the model that produced the shipped bundle, so all conversions
below reproduce the shipped quantization.

## Method — local ARM execution + byte-exact conversion reproduction

M0 could not run the engine (shipped `.so` is aarch64; host is x86_64; a native
build is impossible because `cactus-kernels/CMakeLists.txt:36` hardcodes
`-march=armv8.2-a+fp16+simd+dotprod+i8mm`). M1 solved this locally:

- **ARM64 emulation.** `docker run --privileged tonistiigi/binfmt --install arm64`
  registers qemu-user binfmt on the host kernel, so an `arm64v8/ubuntu:24.04`
  container (glibc ≥ 2.38, which the `.so` needs — Debian bookworm's 2.36 is too
  old) can `dlopen` the shipped `libcactus_engine.so`. The container runs the real
  engine via the repo's Python ctypes binding.
- **Conversion on the host (x86).** The convert path is pure Python and works
  natively. The project's `outputs/distill-calibrated-weights/` holds the GPTQ
  `hessians.safetensors` cache, and `calibration/manifest.json` + the 256-row
  `calibration/agronomy_calibration.jsonl` are present, so
  `python -m cactus.convert`'s `convert()` can be re-driven with
  `--hessian-cache-in` and reproduces the shipped quantization exactly (see #185).

---

## #185 (P1.1) — Reserved-row bake does not hurt the baseline

**Result: PASS.** Reserved rows `151670..151935` were overwritten with
norm-matched random noise in a copy of the merged model, converted with the exact
shipped pipeline, and the resulting bundle (a) loads and generates on the real
ARM engine, and (b) scores item-for-item identically to the control on the
210-item ladder (182/210 vs 182/210; **0/210 items differ**).

### Plumb-byte-exact control

Re-running the shipped conversion (per M0 #181/#182 the table stays whole-tensor
CQ4 — no FP16 request):

```
main(['convert', '--model', outputs/merged-model-distill-v2-rationale,
      '--out', outputs/r179-m1/control-weights, '--bits', '4',
      '--calibration-manifest', LocalLLM/finetune/calibration/manifest.json,
      '--hessian-cache-in', LocalLLM/finetune/outputs/distill-calibrated-weights,
      '--force'])
```

reproduced the shipped bundle exactly:

- `token_embeddings.weights` md5 `3452a896800b5ad115643e9195071c6c` == shipped
- `conversion_manifest.json` identical, 0 differing entries
- `conversion_summary.json` identical (`total_bytes 308743136`, 197 CQ4 / 113 FP16)

### The bake

`outputs/r179-m1/model-p185` — a copy of the merged model with **only**
`model.embed_tokens` rows `151670..151935` replaced. Each row got fresh Gaussian
noise rescaled to that row's *original* L2 norm (so the norm distribution is
unchanged: row 151670 orig 0.3614 → new 0.3614), fixed seed `20260917`. Nothing
else in the state dict touched.

Conversion of the patched model differs from control in **exactly one file**:

```
DIFF token_embeddings.weights     (all 309 other .weights files byte-identical)
md5 control 3452a896…  p185 b2c8c640…
```

### Bundle loads + generates (real ARM engine, qemu)

`outputs/r179-m1/p185-deploy` = the shipped bundle with only
`token_embeddings.weights` swapped for the p185 tensor; loaded in the arm64
container. Same three greedy prompts, control vs p185:

| Prompt | control (`distill-calibrated-deploy`) | p185 |
|---|---|---|
| The capital of France is | …**Paris**. | …**Paris**. |
| Is compost tea a good natural fertilizer?… | "Yes, compost tea **can be** a good…" | "Yes, compost tea **is** a good…" |
| Name one cover crop… | …**straw**. | …**straw**. |

The bundle loads and runs at the same speed (≈23–26 s per 16-token generate under
qemu, both bundles). Two of three open prompts are identical; one flipped a single
mid-generation token (`can be` → `is`).

### Baseline quality: 210-item ladder

`scripts/ladder_score_local.py` (the frozen #140 protocol), bf16 merged model,
greedy, seed 42, run on the host GPU (allowed by the M1 brief; device access was
the bottleneck — no Pixel 7 connected):

| model | correct | accuracy | ambiguous |
|---|---|---|---|
| control (`merged-model-distill-v2-rationale`) | 182/210 | 0.8667 | 0 |
| p185 (reserved rows noise) | 182/210 | 0.8667 | 0 |

**Every item matches**: 0/210 differ in `model_answer` *and* 0/210 differ in
`raw_response`. Note the absolute number here (0.8667) is one item below the
historically recorded 0.8714 for this checkpoint — a harness/environment drift
(`qwen3-train` env, transformers 4.57.6, current chat template), not a regression;
the M1 claim rests on the *paired* control-vs-p185 comparison, which is exact.

### Interpretation (and the one honest caveat)

Because `lm_head` is tied to `embed_tokens` (#183), the reserved rows also
contribute logits to the output softmax denominator. So baking them is **not
mathematically inert** even at matched norm: the third-prompt token flip is that
effect, and it is the expected mechanism. At the aggregate/paired level the effect
is nil on this benchmark (0/210), which is what #185 requires. The magnitude will
grow when the rows carry real (larger) memory vectors — which is the point of
#186 — and is exactly why reserved-ID sampling must be suppressed (#187).

### Note for later (not an M1 failure)

The reserved rows are quantized to CQ4 with the rest of the table, per M0's
correction to this issue. Whether CQ4 fidelity is enough for *learned* memory
vectors is deferred to #193/#194 (M3), as planned.

---

## #186 (P1.2) — Injection changes output, end to end

**Result: PASS — but NOT through the FFI path the M0 writeup assumed. The M0
claim that raw token IDs are injectable via `cactus_submit` is true at the symbol
level but false at runtime (#172 pattern again). A working, fork-free transport
was found instead: a reserved-ID entry in the bundle's `special_tokens.json`, fed
through the normal `cactus_complete` path.**

### Part A — `cactus_submit` (the M0-proposed transport) is broken

Real calls on the shipped `distill-calibrated-deploy` with the same chat prompt
("The capital of France is", rendered with `cactus_render_prompt`, tokenized with
`cactus_tokenize`):

| transport | result |
|---|---|
| `cactus_complete` | `"The capital of France is **Paris**."` (coherent) |
| `cactus_batch_start(1)` + `cactus_submit` + `cactus_batch_stop` | deterministic gibberish: `athinghem delkeley *riculum…` (`gen_ids [43561, 29032, 1594, …]`) |

Reproduced with: raw (non-templated) prompt tokens, chat-templated tokens, a
`cactus_reset` first, and on a different bundle (`distill-arm64-cact-int4`). When a
`cactus_complete` call is made first in the same process, the subsequent
`cactus_submit` output becomes an echo of the prompt instead of gibberish — still
not a correct continuation. The likely mechanism is that `cactus_submit` drives
`Model::decode_step_batch` (`model.cpp:1029`) through `encoder_=lm_encoder_step` +
`decoder_=decoder_media_step` **without the chunked prefill** that
`cactus_complete` performs first (`model.cpp:1308`), leaving the step route
incorrectly conditioned; the root cause was not fixed (out of M1 scope).

The Python convenience `cactus_decode_tokens` (the other raw-token-looking entry)
is also dead — no backing C symbol (M0 #180). So **there is no usable raw-token
generation FFI entry in the shipped engine.**

### Part B — a fork-free transport: reserved IDs as bundle special tokens

The engine's BPE tokenizer loads a string→id special-token map from
`<bundle>/special_tokens.json` (`load_special_tokens_map`, `tokenizer.cpp:272-317`)
and matches those strings during encode
(`split_with_special_tokens`, `tokenizer.cpp:319`). Adding entries

```json
"special_tokens": { "151670": "<|mem0|>", … "151675": "<|mem5|>", … }
```

makes the reserved IDs referenceable from ordinary prompt text. Verified on the
real ARM engine:

```
cactus_tokenize("<|mem0|>") == [151670]
cactus_tokenize("The capital of France is <|mem0|>") == [785, 6722, 315, 9625, 374, 220, 151670]
```

No engine change, no fork — only a tiny bundle-file edit (+156 bytes, see #188).
Baseline (non-reserved) tokenization and generation are unaffected.

### Part C — the injection proof (real ARM engine)

`outputs/r179-m1/p186grad-deploy`: the shipped bundle with reserved rows
`151670..151675` holding distinctive vectors of norms `[0.5, 1.0, 1.5, 2.0, 3.0,
4.0]` (same directions as p186, scaled) and the `<|memN|>` special tokens added.
Greedy, `temperature 0`, same base prompt with vs without one reserved token:

| base prompt | no reserved token | `+ <|mem5|>` (norm 4.0) | `+ <|mem0|>` (norm 0.5) |
|---|---|---|---|
| The best fertilizer for tomatoes is | a combination of nutrients that supports | **biofertilizers** | **well-rounded, balanced nutrients** |
| In one word, farming is | the art and science of cultivating | **farming**. | **farming**. |
| My favorite color is | Your favorite color is a delightful way to express your creativity | Your favorite color, while not directly relevant to the bioaccum | Your favorite color, which is a vibrant shade of green, |
| One fact about soil: | Soil is a critical component of the ecosystem, providing essential | Soil is a complex ecosystem that supports plant growth, regulates | Soil is a critical component of our ecosystem, providing essential |

Injecting a reserved ID deterministically changes the continuation, and the change
is **row-specific** (different reserved IDs carry different vectors → different
outputs). This is the load-bearing M1 proof that the reserved-token mechanism
works end to end. The magnitude also steers style (norm-4 `mem5` pulls
"bioaccum…" into an unrelated prompt), as expected for a soft-prompt-like vector.

### Consequence

The epic's mechanism is viable **without** the `cactus_submit` path, via a
`special_tokens.json` mapping. Any M2+ work should use `cactus_complete` (or the
same tokenizer mapping) and treat `cactus_submit` as unavailable/broken.

---

## #187 (P1.3) — Reserved IDs never leak into output

**Result: FAIL. The stock engine has no way to suppress reserved IDs at sampling.
The `set_vocab_bias` hook M0 pointed at is dead, positive-only, and keyed by
tokenizer strings that cannot name the reserved IDs.**

Source-level, fully cross-checked:

- `Model::set_vocab_bias` / `clear_vocab_bias` (`engine.h:760-763`) has exactly one
  caller: `apply_custom_vocabulary_options` (`utils.h:1219-1226`), and **that
  function itself has no callers anywhere in the engine** (`grep` over
  `src/*.cpp`/`src/*.h`). It is dead code.
- Even if called, the builder `build_token_bias_map` (`utils.h:1186-1202`) clamps
  the boost to `[0, 20]` and only ever **adds** bias (the sampler does
  `value += vocab_it->second`, `model.cpp:1931-1932`). There is no negative/ban
  path. `parse_inference_options_json` (`utils.h:1404-1473`) does not parse
  `custom_vocabulary` / `vocabulary_boost` either, so there is no FFI options route.
- The bias is keyed by `tokenizer->encode(word)` — reserved IDs are by definition
  unproducible by the tokenizer, so they cannot even be named as strings.

Real-generate evidence (no source-read-only claim): the **p186** bundle (reserved
rows norm 8.0) emits reserved IDs for **16/16 generated tokens**
(`[151671, 151670, 151670, …]`) even with **no reserved ID in the prompt** — with
tied embeddings the large rows dominate the head. Through `cactus_complete` the
responses are empty strings because those IDs have no decoded text. There is no
FFI knob that changes this.

There is, however, a usable magnitude threshold: the p186grad bundle (norms ≤ 4.0)
produced **no** reserved-ID leak on the four tested prompts. So leak risk is a
function of learned vector norm, not an inevitability — but the engine provides no
guarantee. **Mitigations to carry forward:** (a) train the LPM so reserved-token
logits stay below real tokens and verify at #193/#194, and/or (b) add a small
banned-token-id list to the sampler (a minimal engine change). #187's literal
acceptance ("never appear") is not met by stock Cactus.

---

## #188 (P1.4 GATE) — Decision: **GO to M2, conditional on reserved-ID suppression**

### Overhead (the stated gate criterion): effectively zero

- Bundle `p186grad-deploy` vs shipped `distill-calibrated-deploy`: all 309
  non-embedding `.weights` files byte-identical; `token_embeddings.weights`
  identical size (80,196,480 B); the only delta is `special_tokens.json`
  7893 → 8049 B = **+156 bytes**. No new graph, no new component, no precision
  change (whole table stays CQ4, as M0 required).
- Latency: paired ARM-engine runs are indistinguishable (init ≈ 2 s;
  ≈ 23–26 s per 16-token greedy generate under qemu for both baseline and injected
  bundles). Absolute on-device numbers still need the Pixel 7 (not attached via
  `adb devices`) — reuse `ladder_benchmark_test.dart` for the real measurement in
  the next device session.

### Decision

**GO to M2.** The reserved-token mechanism is proven end-to-end on the real engine
(injection changes output, row-specifically), with zero plumbing overhead and no
Cactus fork. M2 (encoder + LPM) is where the actual research value is; the
plumbing no longer blocks it.

**Two conditions carried into M2:**

1. **Transport.** Use the `special_tokens.json` reserved-ID mapping + the normal
   `cactus_complete` path. Do **not** build on `cactus_submit` — it does not
   produce coherent generations (#186 Part A).
2. **Suppression.** Stock Cactus cannot ban reserved IDs (#187 FAIL). Either
   (a) train the LPM to hold reserved-token logits below real tokens and validate
   no-leak at #193 (preferred; p186grad shows norms ≤ 4.0 already don't leak), or
   (b) add a minimal banned-token-id list at the sampler. If both are rejected and
   a fork is ruled out, this gate must be revisited.

The gate does **not** block on the M0-deferred per-session runtime-write fork
(still M5) or on the per-row FP16 question (still M3/#193/#194).
