# F179 / M3 — #232 Phase 1: Qwen3-1.7B 4-bit base + SFT cost smoke

Parent #232 (backbone-adaptation disambiguation, child of epic #179), filed after
#231 closed with "width alone, unadapted, doesn't fix generation legibility."
Scope: **feasibility check only** — no full SFT run, no adapter merge, no ladder
scoring, no LPM work, no probe re-run (Phases 2–5).

**Verdict: GO for Phase 2.** The F7 SFT recipe runs against
`unsloth/Qwen3-1.7B-unsloth-bnb-4bit` with only the base-model argument changed,
the 20-step smoke fits VRAM with **~4.0 GiB headroom** at the F7 batch/seq-len,
and the loss curve is sane. Extrapolated full-run cost is **~45–75 min
(0.75–1.25 GPU-h)** — about **2×** the comparable 0.6B 4-epoch run, proportional
to the parameter increase. No batch / seq-len / grad-accum adjustment needed.

## 1. The actual F7 script and recipe (located)

- Driver: `autoresearch/finetune.py` → `train.py` (Unsloth `FastLanguageModel` +
  TRL `SFTTrainer`). Config: `autoresearch/configs/phase25-distill/distill-v2-rationale.yaml`.
- F7 base (from `outputs/lora-adapter-distill-v2-rationale/final/adapter_config.json`):
  `unsloth/qwen3-0.6b-unsloth-bnb-4bit`; the 1.7B equivalent
  **`unsloth/Qwen3-1.7B-unsloth-bnb-4bit` exists** on the hub (verified).
- Exact F7 args extracted from
  `outputs/lora-adapter-distill-v2-rationale/checkpoint-1612/training_args.bin`:
  `max_seq_length=2048`, `per_device_train_batch_size=4`,
  `gradient_accumulation_steps=4` (effective 16), `epochs=4`, `lr=5e-5` cosine,
  `warmup_ratio=0.05`, `weight_decay=0.01`, `optim=adamw_8bit`, bf16, gradient
  checkpointing on, `packing=false`, `seed=42`. LoRA: `r=32`, `alpha=64`,
  `dropout=0.025`, targets `{q,k,v,o,gate,up,down}_proj` (matches the issue).

**Data correction (the issue's premise is off).** The **871** is the teacher
*source* (`data/sources/teacher_distill_qwen3-8b.jsonl`). The prepared training set
the config actually trains on is **`data/train_distill.jsonl` = 6437 rows** (871
expanded), with `data/val_distill.jsonl` = 716. This is confirmed by F7's
`trainer_state.json`: `max_steps=1612 = 4 × ceil(6437/16)`. The smoke uses the
same prepared files, so it is apples-to-apples with F7.

## 2. Smoke (20 optimizer steps, same recipe, only base swapped)

`train.py --model unsloth/Qwen3-1.7B-unsloth-bnb-4bit ... --max-steps 20`
(the `--max-steps` knob was added this phase; default `-1` reproduces F7 exactly —
F7's `training_args.bin` also had `max_steps=-1`).

| metric | 1.7B | 0.6B (same 20-step smoke) |
|---|---|---|
| return code | 0 | 0 |
| train_runtime | 55.70 s | 27.31 s |
| seconds / optimizer step | **2.785** | 1.366 |
| steps / second | 0.359 | 0.732 |
| loss @ step 10 / 20 | 1.970 / 1.810 | 2.038 / 1.971 |
| train_loss | 1.890 | 2.005 |
| NaN | none | none |
| peak VRAM (system) | **8228 MiB** | — |
| idle VRAM | 1853 MiB | — |
| incremental VRAM | ~6375 MiB | — |
| headroom (of 12282 MiB) | **~4054 MiB** | — |

Loss decreases and is finite; VRAM peak is ~8.2 GB system-wide against ~10.4 GB
free, leaving **~4.0 GiB headroom** at the F7 batch 4 / seq 2048. The 1.7B is
**2.04×** the 0.6B per optimizer step (as measured over 20 steps).

Independent re-run of the 1.7B smoke: **53.84 s, 0.371 steps/s, loss 1.8891,
peak 8076 MiB** — stable against the first run (55.70 s / 0.359 / 1.8898 / 8228
MiB).

## 3. Extrapolated full-run cost

Full run = **1612 optimizer steps** (`4 × ceil(6437/16)`, matching F7).

| basis | 1.7B full | 0.6B full |
|---|---|---|
| 20-step measured average (incl. first-step compile) | 4489 s = **74.8 min = 1.25 GPU-h** | 2202 s = 36.7 min |
| steady-state correction | ~44 min | ~21.6 min (recorded 4-epoch 0.6B run, journal `p5accept-ep4`) |

The 20-step averages include Unsloth's first-step compile, which inflates
per-step cost (more so for the 0.6B run, where the recorded full 4-epoch run was
21.6 min vs the 36.7-min naive extrapolation, ~0.80 s/step steady-state).
Applying the same correction, the realistic 1.7B full run is **~45 min
(~0.75 GPU-h)**; **75 min (1.25 GPU-h) is a conservative upper bound.** Either
way it is **~2×** the 0.6B run, consistent with the ~2.8× parameter increase and
the measured 2.04× per-step ratio — **not unexpectedly large**.

## 4. GO / NO-GO

**GO for Phase 2 (the full SFT run).** Meets the bar: fits VRAM with reasonable
headroom, loss curve sane, full-run cost proportional to the 0.6B reference. No
change to batch (4), seq-len (2048), or grad-accum (4) is required. The one code
change is the `--max-steps` smoke knob on `train.py` (default preserves F7).

## 5. Reproduce

```bash
cd [LOCAL_PATH]
# start VRAM sampler, then:
[LOCAL_PATH] train.py \
  --model unsloth/Qwen3-1.7B-unsloth-bnb-4bit \
  --train-data data/train_distill.jsonl --val-data data/val_distill.jsonl \
  --output-dir /tmp/lora17smoke \
  --epochs 4 --max-steps 20 --lr 5e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --weight-decay 0.01 --batch-size 4 --grad-accum 4 --lora-r 32 --lora-alpha 64 \
  --lora-dropout 0.025 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max-seq-len 2048 --chat-template-variant hardcoded-v1 --train-order shuffled --seed 42
```

Artifacts: `results/F179_M3_BACKBONE_ADAPT_PHASE1.json`; code change
`--max-steps` in `train.py` (default `-1` = F7 behavior). Smoke checkpoints under
`/tmp` (not committed); no adapter merged.
