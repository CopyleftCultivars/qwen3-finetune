# G6 (#153): 8B LoRA SFT on the G5 trajectory set

Date: 2026-09-11/12. Single lever = dataset: LoRA hyperparameters reused
verbatim from the v5clean recipe (`autoresearch/configs/v5-clean-baseline.yaml`)
except where hardware forced an adaptation (noted below); the trajectory set
(G5, `results/G5_RLM_TRAJECTORIES.md`) is the only thing that changed.

## Data prep

`rlm_slice/build_g6_sft_dataset.py` converts G5's 131 quality-filtered
trajectories (`results/g5_trajectories_qwen3-8b_filtered.jsonl`) into
`train.py`'s `messages`-format SFT rows. Each row is built from the
trajectory's *last* RLM iteration only: the `rlm` package's LocalREPL
re-sends the full running transcript as that iteration's `prompt` (system
prompt + question + every prior Turn-N marker and assistant response), so
`prompt + [{"role": "assistant", "content": response}]` is already the
complete, clean conversation exactly as the model saw it at generation
time -- no reconstruction, no risk of drifting from the real inference-time
format. No loss masking (matches `train.py`'s existing convention for plain
instruction/output pairs; not a new convention invented for this data).

131/131 rows converted, 0 skipped. Split 90/10 (seed 42): 117 train / 14 val
→ `data/g6_rlm_sft_train.jsonl`, `data/g6_rlm_sft_val.jsonl`.

Token lengths (Qwen3 tokenizer, ChatML-rendered): min 887, median 1599, mean
1536, max 2280 — `--max-seq-len 2560` covers every example with margin, no
truncation.

## Training

`train.py --model unsloth/qwen3-8b-unsloth-bnb-4bit` (the same 4-bit
checkpoint used for G2/G3/G5 inference), QLoRA:

| Param | Value | Source |
| --- | --- | --- |
| lora_r / lora_alpha / lora_dropout | 32 / 64 / 0.025 | v5clean recipe, unchanged |
| target_modules | q/k/v/o_proj, gate/up/down_proj (7) | v5clean recipe, unchanged |
| lr / scheduler / warmup | 5e-5 / cosine / 0.05 | v5clean recipe, unchanged |
| weight_decay | 0.01 | v5clean recipe, unchanged |
| epochs | 4 | v5clean recipe, unchanged |
| max_seq_len | 2560 | v5clean used 2048; raised to fit this dataset's longer multi-turn rows (max 2280 tokens) |
| batch_size / grad_accum | **1 / 16** | v5clean used 4/4 (same effective batch 16); **forced down** — batch_size=4 at seq_len 2560 for an 8B model OOM'd this 12GB card mid-backward-pass on the first attempt. Same total batch size (16), smaller per-step memory footprint. |
| seed | 42 | v5clean recipe, unchanged |

Effective batch size held at 16 throughout, matching the recipe; only the
per-device/accumulation split changed, which is memory-shape not a training
lever.

Result: 32 steps (4 epochs × 117 examples ÷ effective batch 16), ~11.5
minutes wall time. Loss: 1.199 (epoch 1.27) → 0.774 (epoch 2.55) → 0.652
(epoch 3.82) — steady monotonic decrease, no divergence, no early-stopping
trigger.

## Merge

`merge_adapter.py --base Qwen/Qwen3-8B --force-peft` (the CPU/PEFT path,
not the default Unsloth GPU path — the Unsloth path loads the full fp16 8B
base directly onto the GPU, which needs ~16GB and does not fit this 12GB
card; PEFT's `device_map="cpu"` path merges on the CPU instead, using system
RAM (45GB available) rather than VRAM). Output: `outputs/merged-model-g6-rlm-8b`,
4 safetensors files, 15,623 MB total.

## Accept criterion (#153)

- Adapter merged: `outputs/merged-model-g6-rlm-8b` (not committed to git —
  `outputs/` is gitignored repo-wide, consistent with every prior champion
  checkpoint in this repo).
- Artifact produced: adapter at `outputs/lora-adapter-g6-rlm-8b/final`,
  merged model at `outputs/merged-model-g6-rlm-8b`.
- Config recorded: this document, plus `data/g6_rlm_sft_train.jsonl` /
  `data/g6_rlm_sft_val.jsonl` (committed) and
  `rlm_slice/build_g6_sft_dataset.py` (the exact, reproducible data-prep
  step).

Unblocks G7 (#154): evaluate the fine-tuned 8B and record accept/reject
against the same rlm_slice arms (A/B/R/C) and G2 baseline used for G3/G4,
to see whether the fine-tune actually improved RLM adaptability over the
prompted-only result or just overfit 117 examples.
