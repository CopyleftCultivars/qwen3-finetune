# Spanish transfer evaluation

The Spanish artifact is an evaluation-only English-to-Spanish transfer
diagnostic. It is not a Spanish training source and is not merged into the
native Spanish benchmark namespace.

## Pinned artifacts

- Full transfer set: `../spanish-benchmark-workspace/benchmarks/transfer/en_to_es/english_v5clean_benchmark_es.jsonl`
- Development view: `../spanish-benchmark-workspace/benchmarks/transfer/en_to_es/dev.jsonl`
- Sealed evaluation view: `../spanish-benchmark-workspace/benchmarks/transfer/en_to_es/test.sealed.jsonl`
- Translation manifest: `../spanish-benchmark-workspace/data/provenance/translations/en_to_es_english_v5clean_manifest.json`
- Split manifest: `../spanish-benchmark-workspace/data/provenance/translations/en_to_es_english_v5clean_split_manifest.json`

The translated MCQs must never enter `data/train*.jsonl`, `data/val*.jsonl`,
`eval_freeform/`, or any future training/validation/test pool. The preparation
script rejects the transfer path directly, and the broad preflight audit checks
the current JSONL pools for exact normalized and high token-overlap matches.

## Required preflight

Run this before building or evaluating a candidate that changes the data
pipeline:

```bash
python scripts/check_spanish_transfer_contamination.py
```

The audit must report `passed: true`. A nonzero exit is a hard stop.

## Scoring

Use the same deterministic MCQ log-likelihood scorer as the English anchor:

```bash
conda run -n qwen3-diagnostics python autoresearch/score_mcq_ll.py \
  --model outputs/<candidate-model> \
  --dataset ../spanish-benchmark-workspace/benchmarks/transfer/en_to_es/english_v5clean_benchmark_es.jsonl \
  --provenance-manifest ../spanish-benchmark-workspace/data/provenance/translations/en_to_es_english_v5clean_manifest.json \
  --suite-name es-transfer-full \
  --output results/es_transfer_<candidate>.json \
  --diagnostics-dir ../benchmark-eval-workspace/diagnostics
```

Run the same command against
`../benchmark-eval-workspace/benchmark_combined.jsonl` for the English v5clean
regression anchor. Report the Spanish transfer result and English result
separately; the cross-language difference is descriptive only.

The bilingual acceptance report requires Spanish baseline/improvement and an
English no-regression comparison for the same candidate. The current baseline
is recorded in `results/spanish_transfer_acceptance_baseline.json`.

## Spanish training track status

The Spanish training track is now separate from this transfer diagnostic. It is
derived from the audited QMD-generated source pool, not from the translated
benchmark above:

- canonical MCQ and free-form data live in `../spanish-benchmark-workspace/data/training/`;
- the strict English-only base and Spanish overlays are recorded in
  `results/spanish_qmd_baseline_dataset_manifest.json` and
  `data/spanish_debug/manifest.json`;
- English-anchor and Spanish-transfer contamination audits are required and
  currently pass;
- the Spanish transfer benchmark remains evaluation-only and no sealed test
  file has been read by the training or debug pipeline.

The four-epoch Spanish-QMD candidate and three one-epoch debug overlays all
regressed against the v5clean bilingual gates. Their results are recorded in
`results/spanish_qmd_candidate_comparison.json` and
`results/spanish_debug_comparison.json`; v5clean remains the champion.
