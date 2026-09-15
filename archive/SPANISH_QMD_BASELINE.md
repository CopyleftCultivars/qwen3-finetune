# Spanish QMD baseline track

The Spanish training package is now built from the audited QMD-generated MCQ
pool. There were 228 raw items; 14 were excluded for benchmark text overlap
(12 against the English anchor and 2 caught after Spanish transfer audit),
leaving 214 source items.

The canonical Spanish workspace contains:

- `../spanish-benchmark-workspace/data/training/spanish_qmd_mcq_train.jsonl` —
  768 records, A/B/C/D = 192 each;
- `../spanish-benchmark-workspace/data/training/spanish_qmd_mcq_validation.jsonl` —
  88 records, A/B/C/D = 22 each;
- `../spanish-benchmark-workspace/data/training/spanish_qmd_freeform_train.jsonl` —
  192 derived open-answer records;
- `../spanish-benchmark-workspace/data/training/spanish_qmd_freeform_validation.jsonl` —
  22 derived open-answer records;
- `../spanish-benchmark-workspace/manifests/source_registry.jsonl` — source-level
  provenance for all 214 QMD-derived IDs.

The SFT projections are combined with the strictly filtered English v5clean
base in:

- `data/train_spanish_qmd_baseline_v1.jsonl` — 5,356 records;
- `data/val_spanish_qmd_baseline_v1.jsonl` — 599 records;
- `results/spanish_qmd_baseline_dataset_manifest.json` — checksums and counts.

Both candidate splits pass the English-anchor contamination audit and the
Spanish transfer contamination audit. The Spanish transfer benchmark and its
sealed split are evaluation-only and are not in either candidate dataset.

## Training status

The full one-pass baseline completed outside the sandbox on an RTX 4070 Ti:

- 1,340 steps / 4 epochs;
- final training loss: `0.9747786436507951`;
- adapter: `outputs/lora-adapter-spanish-qmd-baseline-v1/final`;
- merged model: `outputs/merged-model-spanish-qmd-baseline-v1`.

CUDA-matched evaluation rejected this candidate as a regression:

- Spanish transfer: option-set gold log-probability `-1.53714` vs v5clean
  `-0.98704`; argmax accuracy `47.97%` vs `71.24%`;
- English v5clean anchor: option-set gold log-probability `-1.43950` vs
  `-0.77167`; argmax accuracy `57.83%` vs `73.67%`.

The candidate is therefore a completed baseline data point, not a promoted
autoresearch model. Comparison details are in
`results/spanish_qmd_candidate_comparison.json`.

## Serial debug diagnostics

To separate Spanish data composition from training pressure, three one-epoch
overlays were trained serially from the untouched v5clean merged model
`outputs/merged-model-p6-lr5e5-r2`. All runs used the v5 settings: learning rate
`5e-5`, cosine schedule, warmup ratio `0.05`, weight decay `0.01`, batch size
`4`, gradient accumulation `4`, LoRA `r=32/alpha=64/dropout=0.025`, maximum
sequence length `2048`, and seed `42`. No sealed benchmark file was read.

The baseline is scored on the 321-row Spanish transfer development split and
the 1,603-row English v5clean anchor:

| Variant | Spanish option-set LL / accuracy | English option-set LL / accuracy | Result |
| --- | --- | --- | --- |
| v5clean baseline | `-1.02648` / `71.65%` | `-0.77167` / `73.67%` | anchor |
| English-only continuation control | `-1.23236` / `65.11%` | `-1.01341` / `66.19%` | attribution control |
| MCQ-only, 768 Spanish records | `-1.18795` / `65.11%` | `-1.03967` / `65.44%` | rejected |
| Free-form-only, 192 Spanish records | `-1.06137` / `65.73%` | `-0.91454` / `68.68%` | rejected |
| Mixed 10%, 95 Spanish records | `-1.24329` / `62.93%` | `-0.98418` / `66.75%` | rejected |

The English-only control also regresses, so continuing one full epoch from the
v5clean merged checkpoint is itself too aggressive. Spanish data still adds a
composition effect—the MCQ-only and mixed variants are worse than the control in
different ways—but the control means Spanish is not the sole cause. Free-form-
only is the least damaging Spanish diagnostic by Spanish option-set
log-likelihood, but it still fails both bilingual gates. This does not establish
that Spanish data is useless. The next controlled work should inspect formatting
and truncation, then reduce optimization pressure (dose, learning rate, or
replay ratio) before changing the benchmark or removing the Spanish track.

The reproducible dataset manifest, contamination audits, model scores, and
gate calculations are in:

- `data/spanish_debug/manifest.json`;
- `results/spanish_debug_english_contamination_audit.json`;
- `results/spanish_debug_spanish_contamination_audit.json`;
- `results/spanish_debug_comparison.json`.

The format audit in `results/spanish_training_format_audit.json` passes the
Spanish canonical message shape, language labels, answer surfaces, and source
split disjointness. Spanish MCQ records peak at 413 tokens and Spanish
free-form records at 354 tokens under the exact hardcoded-v1 template, so none
are truncated at `max_seq_len=2048`. The seven truncated records observed in
the combined overlays are inherited English-base examples, not Spanish rows;
they should be addressed in the from-scratch data audit but do not explain the
Spanish overlay regression.

The initial sandbox-only smoke had failed because it could not see CUDA and
Unsloth refused to run without an accelerator. This is historical; the complete
training and debug runs above were executed on the host RTX 4070 Ti.

```text
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

The exact command to resume on a GPU host is:

```bash
conda run -n qwen3-train python train.py \
  --model outputs/merged-model-p6-lr5e5-r2 \
  --train-data data/train_spanish_qmd_baseline_v1.jsonl \
  --val-data data/val_spanish_qmd_baseline_v1.jsonl \
  --output-dir outputs/lora-adapter-spanish-qmd-baseline-v1 \
  --epochs 4 --lr 5e-5 --lr-scheduler cosine --warmup-ratio 0.05 \
  --weight-decay 0.01 --batch-size 4 --grad-accum 4 \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.025 \
  --max-seq-len 2048 --seed 42
```

The four-epoch command above is retained for reproducibility. Its resulting
candidate was scored and rejected; it is not the autoresearch champion.
