# G2 (#149): prompted Qwen3-8B baseline — 210-item ladder and val_v3

Date: 2026-09-11. Model: `qwen3:8b` via Ollama (Docker, GPU passthrough,
`ollama/ollama` image — see #148). No RLM, no fine-tuning: this is the plain
prompted baseline Epic G's later phases (G3+) will compare RLM scaffolding
and adaptability fine-tuning against.

## 210-item ladder (natural_fertilizers_benchmark_v1_graded.jsonl)

Protocol: `scripts/score_ladder_openai.py`, importing Epic F's exact
`SYSTEM_PROMPT`/`PROMPT_TEMPLATE`/`extract_answer` from `evaluate.py`.
Greedy (temperature 0, seed 42), reasoning off (`reasoning_effort=none`).

| Model | Correct | Accuracy | Wilson 95% |
| --- | --- | --- | --- |
| qwen3:8b (Ollama, this run) | 185/210 | 0.8810 | [0.8302, 0.9181] |
| qwen3-8b (existing reference, `quality/irt/responses/`) | — | 0.8952 | — |

Close to the existing reference point but not identical (-1.4pp) — expected
given a different serving stack (Ollama's own quantization of the published
`qwen3:8b` tag vs whatever produced the reference number) and this being a
genuinely independent run, not a repeat of the same harness.

Full per-item results: `results/ladder210_ollama_qwen3-8b-g2-baseline.jsonl`,
summary: `results/ladder210_ollama_qwen3-8b-g2-baseline_summary.json`.

## val_v3 MCQ gold-token log-likelihood

Protocol: `autoresearch/score_mcq_ll.py` (the same primary-objective MCQ
scorer used for every local 0.6B champion in `autoresearch/program.md`),
`--model unsloth/qwen3-8b-unsloth-bnb-4bit --no-full-precision` (4-bit NF4 —
full bf16 needs ~16GB, this card has 12GB, matches F5's precedent for this
exact model).

- 489 rows in `data/val_v3.jsonl`, 113 are MCQ-formatted (376 skipped as
  non-MCQ instruction/output pairs).
- `mean_gold_logprob` (full sequence): -19.37
- `mean_option_set_gold_logprob` (restricted to the 4 letter tokens): -2.26
- `argmax_accuracy` (does the highest-logprob letter match gold): **0.4867**

Full result: `results/mcq_gold_ll_qwen3-8b-g2-baseline.json`.

## Note: the two accuracy numbers are not directly comparable

Ladder generation accuracy (0.881) and val_v3 argmax log-likelihood accuracy
(0.487) measure different things and should not be read as contradictory:
the ladder score is post-generation, letting the model produce its own
completion (which can draw on positional/format cues, and this scorer
requests reasoning off); the val_v3 score is the raw teacher-forced
probability mass on the gold letter token immediately after the prompt, with
no generation step. A large gap here is a property of the metric, not
evidence the model is unreliable — this is the same reason
`autoresearch/program.md`'s champion promotions always report a primary MCQ
LL score alongside (not instead of) generation-based ladder numbers.

## Accept criterion

Both scores recorded with Wilson 95% CI (ladder) and full metric detail
(val_v3 LL). G2 complete; unblocks G4 (#151, phase-1 gate) once G3's RLM arms
(#150) are run.
