# F179 / M2 — #192 LPM training results

Status: **dry run complete; full run pending approval** (per the #192 brief:
validate wiring on a few hundred steps, then treat the multi-hour run as a
second, separately-approved step).

## Wiring

`scripts/lpm_train.py`. Frozen `Qwen3-0.6B` backbone (gradient checkpointing on
so gradients can reach the soft prompts without holding 28 layers of
activations); frozen `Qwen3EarlyLayerEncoder` (`lpm/encoder.py`, 6 layers);
`LPMModule.forward` → `prompts [B,8,1024]` prepended to the tokenized
`target_text` `inputs_embeds`; cross-entropy over target positions only (8 soft
positions and target padding masked to −100). Only the LPM module (per-persona
slots + shared projection) is optimized; the backbone is never updated.

Loss = NTP CE + `entropy_penalty` + `lambda_mag * mag_penalty`, all three logged
separately plus `entropy` and `projected_norm_mean/max` and the slot/projection
grad norms.

Subset (`include_first_chunk=False`, ≥3 cross-chunk records): **1488 records /
273 personas** (slot params 17.9M; shared projection 5.8M). `query_emb` is
precomputed once because the encoder is frozen.

## Dry-run results (2 configs, 1000 steps each, batch 2, max_length 192)

Both configs: NTP loss falls from 8.90 (step 1) into the ~1.4–3.9 band, and
gradients reach `slot_memory.slots` and `projection.w_v` at every logged step.
Wall clock ~110 s, peak GPU 5.81 GB, **0.03 GPU-hours per 1000 steps** — the
frozen backbone makes this far cheaper than the "few GPU-hours" budget.

| | `lambda_mag=0.01` (#190 default) | `lambda_mag=0.5` |
|---|---|---|
| final NTP loss | 3.07 | 2.81 |
| projected norm mean | 2.35 | 2.21 |
| projected norm max | **5.91** | **4.34** |
| steps with max > 4.0 | many (500+) | 3 (700/900/1000) |
| entropy | 3.47 → 1.30 (penalty fires when < 1.386) | 3.47 → 1.84 |
| val recall@1 (20 personas, chance 0.05) | 0.10 / 0.10 / 0.40 / 0.20 | 0.35 (500) / 0.25 (1000) |

## Headline finding — the magnitude constraint is not actually holding the M1 safe zone

The 4.0 norm cap is a **mean squared hinge** over 273×256 projected slot vectors,
so a handful of drifting vectors barely move the mean. At the #190 default
`lambda_mag=0.01`, `projected_norm_max` climbs to **5.91** by step 1000 — past
M1 #186's observed safe zone (≤4.0 no leak, ~8.0 leaked on 16/16 tokens).
Raising `lambda_mag` to 0.5 cuts the max to 4.34 but still crosses 4.0. **Flagged
loudly per the #192 brief: the constraint as designed neither bounds the max nor
guarantees the safe zone.** Note the learned direction may behave differently
from M1's random-direction probe and M1's measurement was at bf16, not CQ4 —
#193 must re-measure leakage on the actual baked/quantized vectors. Options for
the full run (owner's call):
1. keep the mean hinge and use a larger `lambda_mag` + monitor (values already
   exceed the cap, so this is only a partial mitigation);
2. add a max-based penalty (`relu(||v||-4)^2 .amax()`) or a hard clamp with a
   straight-through gradient on the deployable `bake_vectors` — the only way to
   *guarantee* ≤4.0;
3. rely on #193's post-quantization leakage measurement rather than training-time
   clamping.

## Recall proxy caveat

The val proxy is test-time personalization (freeze the trained projection, fit
each persona-disjoint val persona's slots for 30 steps, then B×B NTP-loss
recall@1). It runs at 4–8× chance but is **noisy** (12–20 personas, one held-out
probe each, targets truncated at 192 of ~560 median tokens). It is good enough to
confirm the module learns persona-discriminative memory, not good enough to rank
runs. The full run should use ≥40 val personas and report the mean over several
probe draws.

## Checkpoint format

`outputs/lpm-persona-memory-192-dryrun/checkpoint.pt` (94.8 MB, gitignored) —
`module_state_dict`, `config` (LPMConfig), `persona_to_slot_index`,
`reserved_id_map` (slot→ID), `subset`, `step`, `summary`. Curves:
`results/F179_M2_TRAINING_CURVES_DRYRUN.json` (`lambda_mag=0.5` run; per-step
loss components + entropy + projected-norm stats + eval records).

## Reproduction

```
PYTORCH_ALLOC_CONF=expandable_segments:True \
[LOCAL_PATH] scripts/lpm_train.py \
    --steps 1000 --batch-size 2 --max-length 192 --lambda-mag 0.5 \
    --eval-every 500 --val-personas 20 --val-fit-steps 30 \
    --out-dir outputs/lpm-persona-memory-192-dryrun \
    --curve-out results/F179_M2_TRAINING_CURVES_DRYRUN.json
```

Proposed full run (pending approval): 3000–5000 steps, `lambda_mag` per the
decision above, `--val-personas 40 --val-fit-steps 40`, into
`outputs/lpm-persona-memory-192/`. At the measured 0.03 GPU-h/1000 steps this is
well under an hour of wall clock.
