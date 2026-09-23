# D203 Gate 2 (#212) — S2a exact-latent selector: reference recall + engine exposure

Milestone #14 (Epic #179), parent #203. Two parts: (1) the off-device reference
recall@1 for the literal on-device-computable hard-argmax selector, and (2) a
bounded engine exposure of the same 6-layer pooled hidden state, verified on the
real ARM engine.

**Headline: the deployable hard-argmax selector is far weaker than the soft
attention mixture it is meant to approximate — reference recall@1 ≈ 0.0–0.17 vs
the soft oracle's 0.25–0.33, i.e. at/below #193's 0.250, not "close to" it. The
engine exposure itself lands cleanly (existing `cactus_embed` path + a bounded
transpile change; no new C++, no new `.so`).**

---

## Part 1 — off-device reference (hard argmax from a live query)

#211 had already scored the *soft* probe_q oracle (`dynamic_recall_matrix`: the
attention-weighted mixture through MLP+LN) at 0.3333/0.2083, but never the HARD
selection the deployment actually computes. `oracle_pack` (`lpm_quant_gap.py:157`)
gathers the literal deployable math — live `probe_q` → `w_q/w_k` → per-head
`argmax` → `bake[k,h,top[k,h]]` — but only its `vectors` were consumed by the
reserved-row path (tied-head-dominated, uninformative). New `s2a` subcommand
scores those hard vectors with `fixed_prompt_recall_matrix` (same `forward_ntp`
code as #193's 0.250) and reports the matched soft oracle alongside.

24 val personas, two disjoint K=12 slices, both fit budgets (issue defaults
40/128/192 and #193's actual 20/96/160). Per-slice chance 0.0833, 24-way 0.0417.

| fit budget | selector | slice0 | slice1 | aggregate (24) | own−offdiag |
|---|---|---|---|---|---|
| 40/128/192 | **hard argmax (S2a)** | 0.000 | 0.0833 | **0.0417 (1/24)** | −0.052 / −0.043 |
| 40/128/192 | soft oracle (matched) | 0.3333 | 0.250 | 0.2917 (7/24) | −0.260 / −0.061 |
| 20/96/160 | **hard argmax (S2a)** | 0.0833 | 0.000 | **0.0417 (1/24)** | −0.041 / +0.025 |
| 20/96/160 | soft oracle (matched) | 0.250 | 0.250 | 0.250 (6/24) | −0.201 / −0.032 |

An independent re-run (GPU bf16 + slot-fitting are nondeterministic at n=24)
gave hard 0.0/0.0 (default) and 0.1667/0.1667 (m193) with soft 0.4167/0.25 and
0.3333/0.3333. Combined across both runs the hard selector spans **0.0–0.1667**
and the soft oracle **0.25–0.3333**; the ordering is stable in every run.

**Comparison to #193's 0.250:** the hard S2a reference is at or below 0.250 in
every run and at/below chance in several slices, while the soft oracle
reproduces #193's number (0.25–0.33). The issue's expectation that S2a "should be
close to 0.250" holds only for the SOFT path; the literal hard-argmax deployment
loses most of the recall signal. The own−offdiag margin agrees: hard ≈ 0
(no discriminative benefit), soft −0.03..−0.26 (clear benefit). This is the
load-bearing Part-1 result and it changes the #203 picture: a per-head top-1
selector is a poor approximation of the soft memory, so the effect probe (#214)
is the necessary next measurement, not a formality.

---

## Part 2 — bounded engine exposure

### Investigation (verified with real calls, not source reads)

- The existing bundles **already contain `decoder_embed_chunk`**: the
  `QWEN_PROFILE.text_component_plans` entry for `qwen3forcausallm` auto-selects
  it, and `components/manifest.json` in `p193-deploy` lists all 6 components
  including `decoder_embed_chunk`.
- `cactus_embed` (FFI + Python binding) really works: a live call on `p193-deploy`
  returned a **1024-dim L2-normalized** pooled vector. `get_lm_embeddings`
  (`model.cpp:5308`) mean-pools `last_hidden_state` over the query tokens —
  structurally the same as `Qwen3EarlyLayerEncoder`.
- **But it ran the full 28 layers**, not the deployed 6: cos(engine_28L,
  python_6L) = 0.032 / −0.017, i.e. orthogonal. The `Qwen3EmbedsCausalLMEmbedChunkAdapter`
  loops `self.backbone.layers[: config.num_hidden_layers]` and applies the final
  norm, so the only gap was layer truncation.

### The change (bounded transpile patch, no C++, no new `.so`)

`CACTUS_QWEN_EMBED_LAYERS` caps the layer count for the `decoder_embed_chunk`
component only (decoder/prefill/step keep the full stack); unset → full stack, no
behavior change. Two edits in
`python/cactus/transpile/model_adapters.py` (`_encode_hidden_states` gains an
optional `num_layers`; `Qwen3EmbedsCausalLMEmbedChunkAdapter` reads the env,
passes it, and caps its transpile metadata). Patch:
`results/F179_M3_S2A_ENGINE_PATCH.diff` (30 insertions / 4 deletions, one file).

Conversion ran in the M1 arm64 container with the fork's `python/` mounted over
the image SDK plus a matching built `.so` at the dev path, reusing the existing
CQ weights: `outputs/r179-m1/s2a/enc6b-deploy`.

### Verification (real engine call)

- Structural: `decoder_embed_chunk` **node_count 314 (6-layer bundle) vs 1392
  (28-layer p193-deploy)** — the truncation took effect (the graph is traced from
  `forward`, so only layers 0..5 + norm are in it).
- Numerical: `cactus_embed` on `enc6b-deploy` vs `Qwen3EarlyLayerEncoder(6)`:
  **cos = 0.9869 / 0.9861 / 0.9888** on three prompts (vs 0.032 / −0.017 for the
  28-layer path). The ~1.3% residual is CQ4 quantization of the engine weights
  against bf16 Python, not a layer/computation mismatch. Machine-readable:
  `results/D203_S2A_ENGINE_EMBED.json`.

Verdict: the engine computes the same 6-layer pooled hidden state the off-device
reference uses. The "bounded engine addition" was smaller than expected — a
conversion-pipeline (transpile) change plus verification, not a new inference
path.

---

## Verdict

- Part 1: reference recorded. Hard S2a ≈ **0.0–0.17** (0–4/24), well below the
  soft oracle 0.25–0.33 and #193's 0.250. The deployable hard-argmax selector is
  a weak approximation; this is a real negative result for the S2a design.
- Part 2: engine exposure verified end-to-end (node_count + cosine). Reusable for
  #214's effect probe.
- Recommendation: proceed to **#214** (effect probe) — it is now the decisive
  measurement for whether ANY selector/engine path earns its complexity, and the
  weak hard-argmax number makes it more important, not less. Keep **#213** (S1
  fallback, no engine change) open as the fallback if #214 shows the effect is
  too weak to justify the engine path.

## Reproduce

```bash
PY=[LOCAL_PATH]
# Part 1 (off-device hard-vs-soft reference), 2 slices x 2 fit budgets
$PY scripts/lpm_quant_gap.py s2a --n-personas 12 --persona-offset 0
$PY scripts/lpm_quant_gap.py s2a --n-personas 12 --persona-offset 12
$PY scripts/lpm_quant_gap.py s2a --n-personas 12 --persona-offset 0  --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit
$PY scripts/lpm_quant_gap.py s2a --n-personas 12 --persona-offset 12 --fit-steps 20 --fit-max-length 96 --max-length 160 --out-tag _m193fit

# Part 2: apply results/F179_M3_S2A_ENGINE_PATCH.diff to the fork (branch
# v2.1.0-suppress-patch), then convert with the env knob in the arm64 container
# (fork python + matching .so mounted; see results/F179_M3_S2A_ENGINE_EMBED.json),
# reusing the existing CQ weights:
#   CACTUS_QWEN_EMBED_LAYERS=6 cactus convert <merged> <out> --bits 4 \
#     --weights-dir <control-weights> --local-files-only \
#     --components lm_encoder_text_chunk,decoder_prefill_chunk,decoder_embed_chunk,lm_encoder_step,decoder_media_step,decoder_step
# then: python scripts/cactus_embed_probe.py (bundle enc6b-deploy, patched .so)
```

Artifacts: `results/D203_S2A_REFERENCE_offset{0,12}[_m193fit].json`,
`results/D203_S2A_ENGINE_EMBED.json`, `results/F179_M3_S2A_ENGINE_PATCH.diff`;
engine probe `scripts/cactus_embed_probe.py`. Raw bundles (gitignored):
`outputs/r179-m1/s2a/enc6b-deploy/`.
