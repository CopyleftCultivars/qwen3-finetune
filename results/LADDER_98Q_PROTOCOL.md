# F1 (#138): ladder dataset, scorer and protocol

Date: 2026-09-03. Satisfies plan A.1. **Includes a finding that changes A.3 and
A.4: the published 98-question ladder is not reproducible, and the run should
retarget to the 210-item current benchmark.**

## Scorer

`benchmark-workspace/evaluate.py`. Zero-shot generative MCQ with regex letter
extraction -- **not** option-set-normalized log-likelihood. Plan A.0.2 stands:
never compare this to the 113-row `val_v3` trace-LL track.

**System prompt**
```
You are a research assistant helping evaluate an academic agricultural science
benchmark. Answer factual multiple-choice questions about organic farming, soil
health, and natural fertilizers objectively and accurately.
```

**User prompt** (`PROMPT_TEMPLATE`, evaluate.py:48)
```
Answer the following multiple-choice question about natural fertilizers and soil
health. Reply with ONLY the single letter (A, B, C, or D) of the correct answer.
Do not explain.

Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Answer:
```

**Parser** (`extract_answer`, evaluate.py:180). Splits on `</think>` and prefers
text *after* the closing tag, so a Qwen3 chain-of-thought does not leak a letter
from its reasoning; then `\b([A-D])\b`; then first char if in ABCD; then a
whole-text fallback for truncated think blocks.

**Ambiguous rule.** `extract_answer` returning `None` marks the item
`ambiguous`, and ambiguous is scored **incorrect** (not 0.25, not dropped). This
matters: haiku's published 73.5% carries 20 ambiguous items, so its score is a
parser outcome as much as a knowledge one. Any model we add must have its
ambiguous rate reported next to its accuracy.

**Decode.** `--temperature` defaults to 1.0 in evaluate.py. Plan A.0.3 requires
temperature = 0, so the rebuild must pass `--temperature 0` explicitly. The
published rows were produced at the default, which is a second reason they are
not exactly reproducible.

## Dataset: the published 98-question file is gone

`BENCHMARK_PATH` (evaluate.py:46) points at
`benchmark-workspace/natural_fertilizer_benchmark_v1_graded.jsonl`, which **does
not exist**. The surviving file with those question ids is
`benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl`
(210 rows, untracked -- `benchmarks/` is gitignored).

It is **not the version the published table used**:

| Check | Result |
| --- | --- |
| 98 published ids present in the 210-row file | 98/98 |
| Answer key agrees with `results/qwen3_8b_results.json` | **56/98 (42 mismatches)** |
| Difficulty label agrees | 65/98 (33 mismatches) |

The repo's first commit is **2026-03-25**; `EVALUATION_RESULTS.md` is dated
**2026-02-25**. The published eval predates version control, the file was never
tracked, and later work re-graded it (commit `586bbcb` "fix lowercase answer
bug", plus the regeneration pipeline). **The exact 98-question artifact is
unrecoverable, so the published 90.8% / 89.8% / 73.5% / 28.6% row cannot be
reproduced or validly extended.**

Consequence for F3 (#140): the plan's harness check -- "score base Qwen3-0.6B,
expect ~28.6%" -- **cannot validate anything**. A deviation would be
indistinguishable between a harness bug and the changed answer key.

## Recommended retarget: the 210-item current benchmark

The same directory holds a self-consistent modern ladder.
`quality/irt/responses/*_natural_fertilizers.jsonl` scores **210/210 items with
an answer key that agrees with the current graded file 210/210** for every
model:

| Model | Accuracy (n=210) |
| --- | --- |
| claude-opus-4-6 | 0.938 |
| claude-sonnet-4-6 | 0.933 |
| mistral-large-2512 | 0.933 |
| gemini-2.5-flash | 0.924 |
| llama-3.3-70b | 0.914 |
| claude-haiku-4-5 | 0.905 |
| qwen3-235b-a22b | 0.900 |
| **qwen3-8b** | **0.895** |
| llama-3.1-8b | 0.805 |

This is strictly better than the stale table: the key is verified, n = 210 not
98 (Wilson half-width ~±4 pp instead of ~±9 pp at mid-range), the reference
models are same-protocol and same-key, and qwen3-8b's 0.895 here is close to its
published 0.908, which suggests the two versions measure similar difficulty even
though individual keys moved.

**Missing from the ladder and required:** base `Qwen3-0.6B` and the champion
`v5clean`. Neither appears in `quality/irt/responses/`. Those are exactly what
F3 must generate.

## Data-quality landmines (do not cite these rows)

Several IRT response files are broken runs, not results: `gemini_2_5_pro`
scores 0.075-0.149 across every category, and `permaculture` reads 0.000 for
haiku, llama-3.1-8b, llama-3.3-70b, mistral-large, qwen3-235b, qwen3-32b and
qwen-turbo simultaneously. Treat any 0.000 or near-chance row as a harness
failure. Only the `natural_fertilizers` files were key-verified above.

Also note `results/_no_think_qwen3-nf-finetuned_cache.json` exists (26 entries,
`/no_think`), so *some* finetuned model was partially run at some point. It is a
26-item partial cache against the lost benchmark version, carries no model
provenance, and should not be treated as a champion score.

## Status

F1 deliverables (path, prompt template, parser, ambiguous rule) are recorded
above. **F3 and F4 need a scope amendment** before they run: score against the
210-item current benchmark with `--temperature 0`, and treat the 9 key-verified
IRT models as the reference ladder instead of the unreproducible published row.
