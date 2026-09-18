# F179 / M2 — Encoder choice (#189) + LPM slot-memory module (#190)

Milestone #13 (Epic #179 / M2). Design + implementation + smoke validation
only: **no training pass over the corpus** (that is #192, held for review).
Compatible with the M0/M1 deployment mechanism: 1024-dim reserved rows in a tied
`model.embed_tokens.weight`, IDs 151670..151935, CQ4 at bake time.

## #189 — Encoder: frozen Qwen3-0.6B early layers

**Decision: use the first 6 of 28 layers of
`outputs/merged-model-distill-v2-rationale` as the encoder, fully frozen
(`trainable_layers=0`), mean-pooled over tokens → R^{1024}.**

Why this path works and the fallbacks were not needed:
- It is a *prefix of the deployed model*, not a new model: the encoder shares
  `embed_tokens` + layers 0-5 with the NTP backbone and the Cactus bundle, so its
  incremental footprint is **0 new weights**. A BERT-base or Qwen3-Embedding-0.6B
  encoder would stack a second network on the 12 GB deployment target.
- Its output width is already the backbone hidden size (1024), so ε(x) feeds
  LPM's W_Q with no adapter layer.
- Frozen, because these same tensors back the 28-layer NTP forward LPM must stay
  compatible with; unfreezing them changes the model the soft prompts were
  trained against.

Measured tradeoff (`scripts/lpm_smoke.py --probe`), retrieval proxy: one
prompt→next-chunk pair per persona, in-batch contrastive top-1 over 40 train /
20 eval personas, 150 steps, discriminative LR (head 1e-3, unfrozen stub 1e-5):

| arm | eval top-1 | final train loss | trainable params | s/step | peak GPU |
|---|---|---|---|---|---|
| frozen stub | 0.35 | 3.03 | 1.05 M | 0.062 | 674 MB |
| stub unfrozen (last layer) | 0.35 | 2.92 | 16.78 M | 0.087 | 1994 MB |
| frozen-vs-finetuned eval repr. cosine | 1.0000 | rel. L2 shift | 0.0048 | | |

Fine-tuning buys **no accuracy** on the proxy while costing ~16x trainable
params, ~40% more step time, and ~3x peak memory; at the head-sized LR it
actually damaged the representation (cos 0.83, 58% relative L2 shift in the
first run) before the discriminative LR stabilized it. Honest caveat: the proxy's
absolute top-1 is low (it is a small 150-step probe, not a real training run), so
the decision rests primarily on the cost + coupling evidence, where the accuracy
tied.

## #190 — LPM slot-memory module

`lpm/config.py`, `lpm/memory.py`, `lpm/data.py`. PersonaMem config, with the
paper's two ambiguous numbers pinned to evidence:

**N=32 is slots *per head*, so 256 slots per persona.** Paper footnote 1:
"~65K unique parameters per user across all heads"; 8 heads x 32 slots x 256 =
65,536 exactly. Paper §6 confirms: "we average the trained slot matrices across 8
heads to obtain M_u in R^{32x256}". The 266-ID reserved pool (256 + 10 spare)
corroborates the 256-slot reading.

**Reserved-ID mapping (documented, 1 slot ↔ 1 row):**

```
slot (h, n)  ->  151670 + 32*h + n        h in [0,8), n in [0,32)
```

giving 151670..151925 (256 IDs), with 151926..151935 spare. `slot_to_reserved_id`
/ `reserved_id_map` in `lpm/config.py` are exact inverses and unit-tested. *This
deviates from the issue's illustrative "151670..151701" (32 IDs), which assumed
32 total slots; the paper's own parameter count and UMAP section fix N as
per-head.* M3/M4 bake `bake_vectors()[:, h, n]` into row `(h,n)`.

### Gap 1 resolved — the 256→1024 projection is end-to-end, inside the module

`ProjectionNetwork.w_v` (per head, `Linear(256, 1024)`) is both the value path of
the dynamic cross-attention and the source of the deployable static vectors
(`LPMModule.bake_vectors`, asserted equal to W_V(slots) in the smoke). The LPM
checkpoint therefore already contains the exact map M3 applies; a bake-time-only
projection could not be trained end-to-end and would have to be invented later.

Because `w_v`'s output is what gets written into tied embedding rows, the
magnitude constraint lives there:
`lambda_mag * mean(relu(||v||_2 - 4.0)^2)` over the projected slots (M1 #186's
observed safe zone: norm ≤ 4.0 did not leak, ~8.0 leaked on 16/16 tokens).
`lambda_mag=1e-2`, `max_vector_norm=4.0`, both configurable; the smoke reports
mean/max projected norms so later runs can watch them.

### Gap 2 resolved — first-chunk (copy-shortcut) records are dropped from the NTP loss

`LPMConfig.include_first_chunk=False` is the default and `lpm.data.filter_records`
enforces it: the 1673 `is_first_chunk` (t=0) records where `prompt_text ==
target_text` are removed, leaving the 2914 cross-chunk records where recall is
actually exercised. The paper-faithful t=0 path remains available via
`include_first_chunk=True` for a #192 ablation. `source` and `cross_chunk` stay
per-record so #192 can slice metrics.

### Entropy load-balancing (not optional)

Per head, `H(a) = -Σ a log a` over the N slot-attention weights;
hinge penalty `alpha * relu(H_target - H(a))`, `H_target = 0.4 * ln(32) = 1.386`,
`alpha = 0.1` (paper §3.1; asymmetric so it penalizes collapse, never forces
uniformity). `loss_terms()` returns `entropy_penalty` and the scaled
`mag_penalty` for #192 to add to the NTP loss.

### Smoke results (`results/F179_M2_SMOKE.json`, real corpus records)

- shapes: prompts `[B,8,1024]`, attention `[B,8,32]`, bake `[B,8,32,1024]`;
  attention rows sum to 1 (max err 1.2e-7); all finite.
- entropy 3.466 ≈ ln(32), above target → entropy penalty 0 at init.
- projected norm mean/max ≈ 1.2/1.3 < 4.0 → mag penalty 0 at init.
- gradients reach both per-persona slots and `w_v`; `bake_vectors()` == forward
  projector output (max abs diff 0.0).
- module trainable params 6.06 M (shared projection; independent of persona
  count) + 65,536 per persona for slots.

## Integration notes for #192 / M3 / M4

- Loss: NTP on `target_text` + `entropy_penalty` + `lambda_mag * mag_penalty`.
- **Persona budget is a real #192 knob.** The train corpus has 1500 personas but
  only 1437 have ≥2 chunks and 298 have ≥3. Training all 1500 means ~98 M slot
  parameters and ~1200 two-chunk personas that can only overfit; `≥3 chunks`
  (298 personas, ~19.5 M slot params, 1538 cross-chunk records) is the natural
  starting point, and #192 should report whichever it uses.
- M3 bake: for the target persona, write `bake_vectors()[0, h, n]` into
  `151670 + 32h + n`, then CQ4-convert. Check post-quantization norms and
  reserved-token suppression at #193 (M1 #187 remains unresolved).
- Alternative if 256 baked rows is judged too many for the bundle: the paper's
  §6 head-averaging gives 32 static vectors, but that collapses the per-head
  structure; not the default.

## Reproduction

```
[LOCAL_PATH] scripts/lpm_smoke.py \
    --probe --probe-steps 150 --probe-train-personas 40 --probe-eval-personas 20 \
    --out results/F179_M2_SMOKE.json
[LOCAL_PATH] -m pytest tests/test_lpm_module.py -q
```
