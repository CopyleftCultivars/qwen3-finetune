# F179 / M2 — #192 LPM training results

Status: **complete** — dry run validated the wiring, the magnitude gap it found
was fixed, and the full run produced a loadable checkpoint. #193 (M3 bf16-vs-CQ4)
is unblocked.

## Wiring

`scripts/lpm_train.py`. Frozen `Qwen3-0.6B` backbone (gradient checkpointing on
so gradients reach the soft prompts without holding 28 layers of activations);
frozen 6-layer `Qwen3EarlyLayerEncoder`; `LPMModule.forward` → `prompts [B,8,1024]`
prepended to the tokenized `target_text` `inputs_embeds`; cross-entropy over
target positions only (8 soft positions + padding masked to −100). Only the LPM
module (per-persona slots + shared projection) is optimized; the backbone is
never updated. Loss = NTP CE + `entropy_penalty` + `lambda_mag * mag_penalty`,
each component logged separately alongside entropy, projected norms and the
slot/projection grad norms.

Subset (`include_first_chunk=False`, ≥3 cross-chunk records): **273 personas,
1488 records** → 1215 fit + 273 held-out within-persona probes. Per-persona
slots: 65536 × 273 = 17.9M; shared projection 5.8M.

## Magnitude gap found by the dry run — fixed

The dry run showed the #190 mean-hinge penalty did not hold the M1 #186 safe
zone: `projected_norm_max` reached **5.91** at `lambda_mag=0.01` and **4.34** at
`lambda_mag=0.5`. Fix (approved): `lpm/memory.py` now (a) defaults
`mag_penalty_mode="max"` (penalise the worst vector, not the mean) and (b) applies
a differentiable radial clamp `min(1, 4/||v||)` to every projected slot vector, so
the **baked rows are guaranteed ≤4.0**, not merely nudged. The same clamped
vectors feed the dynamic value path and `bake_vectors()`, so training and
deployment agree. Unit-tested (`project_to_ball` bound + gradient; max-vs-mean
penalty).

## Full run

4000 steps, batch 2, max_length 192, lr 1e-3, seed 0, `lambda_mag=0.01`,
`mag_penalty_mode=max`, `clamp_memory_norm=true`. 273 personas / 1215 fit
records. Wall clock **560.6 s = 0.156 GPU-hours**, peak GPU 5.18 GB.

| metric | value |
|---|---|
| NTP loss | 9.14 (step 1) → ~2.8–3.7 band, 2.92 final (batch-2 variance) |
| projected norm max | **4.00, 0 steps above** (0 leakage-watch triggers) |
| projected norm mean | 1.20 → 3.65 (near the cap by the end) |
| entropy | 3.47 → 1.55 (target 1.386; penalty fires on 7/41 logged steps) |
| slot grad norm | 0.031 … 0.124 throughout |
| projection grad norm | 0.0024 … 0.016 throughout |
| val TTP recall@1 (24 personas, chance 0.042) | 0.29, 0.38, 0.42, 0.29, 0.33, 0.33, 0.33, 0.33 → mean **0.339** |
| within-persona held-out probe recall@1 (48-way, chance 0.021) | 0.04, 0.10, 0.08, 0.25, 0.13, 0.21, 0.15, 0.15 → mean **0.14** |

Both recall proxies sit ~7–8× chance, so the module learns persona-discriminative
memory rather than only lowering the loss; the val (persona-disjoint,
test-time-personalization) curve is the more stable of the two.

## Checkpoint

`outputs/lpm-persona-memory-192/checkpoint.pt` (gitignored, 95 MB). Reload it in
a fresh process with `lpm.checkpoint.load_lpm_checkpoint(path)` — it applies
`LPMConfig.from_dict` (handles the `n_slots_per_head` rename and the
computed-only keys), rebuilds the module, and strict-loads the state dict, so no
hand-mapping is needed. Verified: `slots [273,8,32,256]`, `bake_vectors` norm
max 4.0. Contents: `module_state_dict`,
`config` (LPMConfig), `persona_to_slot_index`, `reserved_id_map` (slot→
151670..151925), `subset`, `step`, `summary`. Curves:
`results/F179_M2_TRAINING_CURVES.json` (per-step loss components + entropy +
norm stats + all eval records).

## What #193 must now do

- Bake one persona's `bake_vectors()[0,h,n]` into row `151670+32h+n`, CQ4-convert,
  and re-measure reserved-ID leakage. M1's ≤4.0 observation was for random-
  direction vectors at bf16; here the vectors are **learned and the mean norm is
  3.65 (many pinned at the 4.0 clamp)**, so CQ4 quantization fidelity and
  suppression need a fresh measurement, not an assumption.

## Honest caveats

- The recall proxies are weak in absolute terms and noisy (batch 2; only 24 val
  personas have ≥2 chunks, so val chance is 1/24; targets truncated at 192 of a
  ~560-token median). They validate that memory is persona-discriminative; they
  do not rank hyperparameters.
- Mean projected norm rising to 3.65 means the clamp is active for much of the
  memory — capacity may be squeezed. If #193 finds recall-limited-by-norm, the
  safe zone or a larger slot budget is the lever.
- 4000 steps is well inside the budget; longer training / larger batch is cheap
  (0.156 GPU-h here) if #193 wants a stronger checkpoint.

## Reproduction

```
PYTORCH_ALLOC_CONF=expandable_segments:True \
[LOCAL_PATH] scripts/lpm_train.py \
    --steps 4000 --batch-size 2 --max-length 192 --eval-every 500 \
    --val-personas 40 --val-fit-steps 40 --probe-personas 48 \
    --out-dir outputs/lpm-persona-memory-192 \
    --curve-out results/F179_M2_TRAINING_CURVES.json
```
