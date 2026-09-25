# F179 / M3 — #232 Phase 4: retrain the LPM module against the adapted Qwen3-1.7B backbone

Parent #179; follows #232 Phases 1–3 (GO — the 1.7B distillation's adaptation landed:
ladder 0.8667 vs 0.7286 base, `F179_M3_BACKBONE_ADAPT_PHASE3.md`). Scope:
**LPM training only.** No generation-legibility probe (Phase 5), no CQ4, no ladder
re-score, no further backbone SFT.

**Verdict: GO for Phase 5.** Training completed without divergence (4000/4000
steps, NTP 4.05→2.89, no NaN, `projected_norm_max` exactly 4.00 with 0
leakage-watch triggers), and the checkpoint round-trips through
`lpm.checkpoint.load_lpm_checkpoint` with the correct d_model, slot shape, baked
norms, and reserved-ID map. That is the deliberately low Phase-5 bar. **Recall is
reported but is not gated on** — see §6, where the recall proxies move *opposite*
to the Phase-3 ladder result, which is itself the cleanest possible demonstration
that recall@1 is not a legibility proxy.

## 1. Code change: pointing the encoder at the adapted local checkpoint

`scripts/lpm_train.py` gained `--backbone-path` (default `None`), an explicit
path/HF-id override that takes precedence over both `--backbone`'s built-in model
and the pre-existing `--model`:

```python
model_path = args.backbone_path or args.model or BACKBONE_MODEL[args.backbone]
```

The LPM d_model profile still comes from `--backbone`, and a hidden-size guard
rejects a mismatch loudly rather than silently training a width-inconsistent
module:

```
ValueError: backbone '...1.7b' hidden_size=2048 != LPMConfig.d_model=1024
for --backbone 0.6b; pick the matching --backbone profile or a supported width (1024, 2048)
```

`lpm/encoder.py` needed no change: `load_encoder()` already accepts a local path
via its `Path(model_path).exists()` check (verified, not assumed — the adapted
directory loads as a 2048-wide 6-layer stub, 613,182,976 frozen params).

**Backward compatibility.** No existing path was altered: `--backbone-path`
defaults to `None`, so `--backbone 0.6b` and `--backbone 1.7b` are unchanged.
Verified with a 3-step `--backbone 0.6b` smoke (d_model=1024, encoder chunk 0, the
#192 subset, 2.84 GB peak), exactly as #231 Phase 3 held the 0.6B default fixed.

## 2. Recipe (identical to #231 Phase 3, encoder weights swapped)

| setting | value |
|---|---|
| encoder/backbone | **`outputs/merged-model-distill-v2-rationale-1.7b`** (adapted; `--backbone 1.7b --backbone-path …`) |
| d_model / H / N / d_mem | 2048 / 8 / 32 / 256 |
| subset | `include_first_chunk=False`, ≥3 cross-chunk → **273 personas, 1488 records**, 1215 fit + 273 held-out probes |
| steps / batch / max_length / lr | 4000 / 2 / 192 / 1e-3 |
| regularizers | `lambda_mag=0.01`, `mag_penalty_mode=max`, `clamp_memory_norm=true` |
| grad checkpointing | on |
| eval | every 500: val-personas 40 (24 eligible; fit 40 steps) + within-persona probes 48 |
| encoder chunk | 64 (the #231 OOM fix, unchanged) |
| seed | 0 |
| module params | 28,957,184 |

The only difference from #231 Phase 3 is the encoder weights (adapted vs stock).
The fitted subset is byte-identical: 273 personas / 1488 records / 1215 fit / 273
probes, same as both #231 and #192.

## 3. Cost

| metric | #232 Phase 4 (adapted 1.7B) | #231 Phase 3 (raw 1.7B) | #192 (0.6B) |
|---|---|---|---|
| wall clock | **871.7 s** | 854.5 s | 560.6 s |
| **GPU-hours** | **0.242** | 0.237 | 0.156 |
| seconds/step | 0.218 | 0.214 | 0.140 |
| peak GPU | **8.388 GB** | 8.388 GB | 5.18 GB |
| total samples seen | 8000 | 8000 | 8000 |

Unchanged within noise from #231, as expected: the encoder architecture (6/28
layers, 2048-wide) is the same, only its weights differ. Comfortably inside one
12 GB GPU.

## 4. Training metrics (#231-comparable table)

| metric | #232 adapted 1.7B | #231 raw 1.7B |
|---|---|---|
| NTP loss (step 1 → final) | 4.049 → **2.892** | 5.226 → 2.768 |
| projected norm max (leakage-watch triggers) | **4.00, 0** | 4.00, 0 |
| projected norm mean (start → final) | 1.72 → **3.871** | 1.72 → 3.908 |
| entropy (start → final; target 1.386) | 3.466 → **1.554** (min 0.733) | 3.466 → 1.440 (min 1.047) |
| slot grad norm (min … max; final) | 0.017 … 0.999 (**0.137**) | 0.021 … 0.983 (0.156) |
| projection grad norm (min … max; final) | 0.0003 … 0.084 (**0.0014**) | 0.0015 … 0.053 (0.006) |

Sane end-to-end: monotone NTP decline with no divergence or NaN, gradients
bounded, entropy pulled to the 1.386 target, and the magnitude clamp holding
exactly at 4.0 with zero leakage-watch triggers — the same qualitative picture as
#231. The lower starting NTP (4.05 vs 5.23) is consistent with the encoder being
domain-adapted rather than stock.

## 5. Checkpoint

`outputs/lpm-persona-memory-232-p4/checkpoint.pt` (gitignored, ~115.9 MB).
Round-trip via `lpm.checkpoint.load_lpm_checkpoint` (strict load, fresh process):

- `config.d_model = 2048`, H=8, N=32, d_mem=256
- `slot_memory.slots.shape = [273, 8, 32, 256]`, 273 personas, `step = 4000`
- `bake_vectors([0]).shape = [1, 8, 32, 2048]`, norm max **4.0000009** ≤ 4.0
- reserved-ID map: 256 ids, **151670..151925**
- `summary.backbone = "1.7b"`, `summary.backbone_path = "outputs/merged-model-distill-v2-rationale-1.7b"`, `summary.d_model = 2048`

This is exactly the object Phase 5 needs: the LPM module now trained against the
adapted 1.7B encoder, the confound from #231 removed.

## 6. Recall proxies — reported, deliberately NOT over-interpreted

| eval step | val recall@1 (chance 0.042) | within-persona probe recall@1 (chance 0.021) |
|---|---|---|
| 500 | 0.1250 | 0.0417 |
| 1000 | 0.1667 | 0.0208 |
| 1500 | 0.2083 | 0.1042 |
| 2000 | 0.1667 | 0.0417 |
| 2500 | 0.2083 | 0.0417 |
| 3000 | 0.1250 | 0.0625 |
| 3500 | 0.1250 | 0.0417 |
| 4000 | 0.1250 | 0.0625 |
| **mean** | **0.1562 (3.8× chance)** | **0.0521 (2.5× chance)** |
| #231 raw-1.7B mean | 0.2761 (6.6× chance) | 0.1823 (8.8× chance) |

Both proxies are above chance, so the module learns persona-discriminative
memory, but **both are materially lower than the raw-1.7B run** (val 0.156 vs
0.276; probe 0.052 vs 0.182). Read against Phase 3, this is the key point:

- The **adapted** encoder improved the domain ladder by **+13.8 pp** over base.
- The **adapted** encoder produced a **worse** LPM recall proxy than the raw one.

Adaptation and recall move in *opposite* directions — direct, within-this-epic
evidence that recall@1 is not a proxy for generation legibility, reinforcing
#231 Phase 4 (strong recall, zero persona markers). Per the phase brief, and per
#228/#230/#231, recall is **informative but not the test**; no victory or defeat is
declared on it, and Phase 5 is not gated on it. It is also a noisy proxy (batch 2,
24 val personas).

## 7. GO / NO-GO for Phase 5

**GO.** The bar is deliberately low and both conditions hold:
1. the checkpoint round-trips correctly (d_model 2048, slot `[273,8,32,256]`,
   bake norm ≤ 4.0, reserved IDs 151670–151925, step 4000);
2. training completed without divergence (4000/4000 steps, NTP 4.05→2.89, no NaN,
   0 leakage-watch triggers, cost and peak memory matching #231).

Phase 5 = the #228/#230/#231 generation-legibility probe (same persona, same 12
prompts, same marker list) against this adapted-backbone checkpoint. Its job is
to test whether an *adapted* 1.7B backbone surfaces persona-distinctive content
where raw 1.7B (and 0.6B) did not. A negative Phase 5 cleanly rules out both width
and this adaptation recipe.

## 8. Reproduce

```bash
cd [LOCAL_PATH]
PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 \
[LOCAL_PATH] scripts/lpm_train.py \
    --backbone 1.7b \
    --backbone-path outputs/merged-model-distill-v2-rationale-1.7b \
    --steps 4000 --batch-size 2 --max-length 192 --encoder-chunk 64 \
    --eval-every 500 --val-personas 40 --val-fit-steps 40 --probe-personas 48 \
    --seed 0 \
    --out-dir outputs/lpm-persona-memory-232-p4 \
    --curve-out results/F179_M3_BACKBONE_ADAPT_PHASE4_TRAINING_CURVES.json

# 0.6B path unchanged (sanity, 3 steps):  --backbone 0.6b --eval-every 1000000
```

Artifacts: `results/F179_M3_BACKBONE_ADAPT_PHASE4.md`,
`results/F179_M3_BACKBONE_ADAPT_PHASE4_TRAINING_CURVES.json`, code diff in
`scripts/lpm_train.py`; checkpoint under `outputs/lpm-persona-memory-232-p4/`.
