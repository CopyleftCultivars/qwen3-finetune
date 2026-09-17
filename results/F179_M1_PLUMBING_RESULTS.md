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
