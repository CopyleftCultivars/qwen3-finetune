# F179 / M3 — #232 Phase 2: full Qwen3-1.7B SFT run + adapter merge

Parent #232 (backbone-adaptation disambiguation, child of epic #179), follows
Phase 1 (GO — recipe runs on `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`, ~4.0 GiB
headroom, extrapolated 0.75–1.25 GPU-h). Scope: **full training + merge only.**
No ladder scoring (Phase 3), no LPM retraining (Phase 4), no probe re-run
(Phase 5), no quantization.

**Verdict: GO for Phase 3.** The full 4-epoch run completed cleanly, the loss
curve is sane end-to-end, the adapter merged into a loadable 16-bit
`merged-model-distill-v2-rationale-1.7b`, and that model generates coherent
non-degenerate text. The actual accept/revert question (does the adaptation help)
is Phase 3's ladder score — not pre-judged here.

## 1. Full run

Same command as Phase 1's reproduce block, `--max-steps -1` (epoch-driven, = F7's
own `max_steps=-1`) and a real output dir. Recipe exactly F7:
`unsloth/Qwen3-1.7B-unsloth-bnb-4bit`, `data/train_distill.jsonl` (6437) /
`data/val_distill.jsonl` (716), seq 2048, batch 4 × grad-accum 4 (eff 16),
4 epochs, lr 5e-5 cosine, warmup 0.05, wd 0.01, adamw_8bit, bf16, gradient
checkpointing, packing off, seed 42; LoRA r32 / alpha64 / dropout0.025 on
`{q,k,v,o,gate,up,down}_proj`.

| metric | value |
|---|---|
| return code | **0** |
| completed optimizer steps | **1612** (4 × ceil(6437/16), matches F7) |
| wall clock | **4282 s = 71.4 min** |
| **GPU-hours** | **1.188** |
| seconds / step | 2.66 |
| peak VRAM (system) | **8881 MiB** (Phase-1 smoke: 8228/8076) |
| adapter | `outputs/lora-adapter-distill-v2-rationale-1.7b` |

## 2. Cost vs the Phase-1 bracket

Phase 1 extrapolated **44–75 min (0.75–1.25 GPU-h)**. Actual: **71.4 min
(1.188 GPU-h)** — **the upper end of the bracket**, close to the conservative
1.25 GPU-h bound. The lower "steady-state corrected" estimate (~44 min) was too
optimistic: per-step stayed ~2.66 s for the whole run, i.e. the 20-step smoke's
average (2.79 s/step) was already representative and there was no compile
amortization gain at scale. Still ~2× the 0.6B 4-epoch run, proportional to the
parameter increase, and comfortably within a single 12 GB GPU (peak 8.88 GB).

## 3. Loss curve (sane end-to-end)

From `checkpoint-1612/trainer_state.json` (161 logged points, every 10 steps):

| | value |
|---|---|
| first / last | 2.193 / **0.8292** |
| min / max | 0.7364 / 2.3779 |
| NaN / inf | **none** |
| quartile-mean loss (by training progress) | 1.298 → 0.991 → 0.883 → **0.823** |

Steady monotonic decline, no divergence. For reference the 0.6B F7 run's own
final train loss was 1.030 — the 1.7B run reaches 0.829, but absolute loss is not
comparable across capacities and is not a quality signal here.

## 4. Merge (mirrors F8)

`merge_adapter.py` (Unsloth `save_pretrained_merged`, `merged_16bit`), exactly the
F8 step, with the 1.7B base:

```
--base Qwen/Qwen3-1.7B --adapter outputs/lora-adapter-distill-v2-rationale-1.7b \
--output outputs/merged-model-distill-v2-rationale-1.7b --dtype float16 --max-seq-len 2048
```

Result: 1 safetensors file, **3282 MB** (3,441,185,608 B); `config.json` is
`Qwen3ForCausalLM`, `hidden_size=2048`, 28 layers, `vocab_size=151936`,
`tie_word_embeddings=true`.

## 5. Load + generation sanity (not the ladder)

Greedy (`do_sample=False`), `max_new_tokens=32`, chat template, fp16 on GPU, 4
prompts:

| prompt | continuation |
|---|---|
| The best fertilizer for tomatoes is | "I don't know, but perhaps I can offer more specific guidance if you provide additional details. However, based on the information provided in the \"WHAT IS NAT…" |
| Explain how to make compost at home. | "I don't know, but perhaps I can offer more guidance if you provide additional details. However, I can provide a general explanation on how to make compost at…" |
| What is crop rotation? | "Crop rotation is a farming practice where different types of crops are grown in a sequence on the same piece of land over successive seasons. This method is used to optimize…" |
| One fact about soil: | "Soil is a living, dynamic ecosystem that supports a vast array of life, from microscopic bacteria and fungi to large animals and plants." |

Coherent, fluent, non-degenerate. Some prompts open with an "I don't know,
but…" hedge inherited from the F6 MCQ-rationale target style — noted, not a
quality judgement (the ladder is Phase 3).

## 6. GO / NO-GO for Phase 3

**GO.** Run completed without error; loss sane; merged model loads and generates
coherent text. Phase 3 (210-item ladder scoring of the merged 1.7B-distilled model
against the F8 accept/revert gate) is the next step and the real quality bar.

## 7. Reproduce

```bash
cd [LOCAL_PATH]
# full run
[LOCAL_PATH] train.py \
  --model unsloth/Qwen3-1.7B-unsloth-bnb-4bit \
  --train-data data/train_distill.jsonl --val-data data/val_distill.jsonl \
  --output-dir outputs/lora-adapter-distill-v2-rationale-1.7b \
  --epochs 4 --max-steps -1 --lr 5e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --weight-decay 0.01 --batch-size 4 --grad-accum 4 --lora-r 32 --lora-alpha 64 \
  --lora-dropout 0.025 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max-seq-len 2048 --chat-template-variant hardcoded-v1 --train-order shuffled --seed 42
# merge
[LOCAL_PATH] merge_adapter.py \
  --base Qwen/Qwen3-1.7B --adapter outputs/lora-adapter-distill-v2-rationale-1.7b \
  --output outputs/merged-model-distill-v2-rationale-1.7b --dtype float16 --max-seq-len 2048
```

Artifacts: `results/F179_M3_BACKBONE_ADAPT_PHASE2.md`, `..._PHASE2.json`. Adapter
(`outputs/lora-adapter-distill-v2-rationale-1.7b`) and merged model
(`outputs/merged-model-distill-v2-rationale-1.7b`) are under gitignored `outputs/`.
