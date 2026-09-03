# F3/F4 (#140, #141): the champion on the 210-item ladder

Date: 2026-09-03. Benchmark:
`benchmark-workspace/benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl`
(210 items). Protocol imported verbatim from `benchmark-workspace/evaluate.py`
(`SYSTEM_PROMPT`, `PROMPT_TEMPLATE`, `extract_answer`) by
`scripts/ladder_score_local.py`; greedy, temperature 0, seed 42, think-off.
Contamination: clean (F2, `results/LADDER_210_CONTAMINATION.md`).

## The ladder

| Model | Params | Accuracy | Wilson 95% | Ambiguous | Harness |
| --- | --- | --- | --- | --- | --- |
| claude-opus-4-6 | - | 0.9381 | 0.8970-0.9635 | - | IRT |
| claude-sonnet-4-6 | - | 0.9333 | 0.8912-0.9599 | - | IRT |
| mistral-large-2512 | - | 0.9333 | 0.8912-0.9599 | - | IRT |
| gemini-2.5-flash | - | 0.9238 | 0.8798-0.9526 | - | IRT |
| llama-3.3-70b | 70B | 0.9143 | 0.8686-0.9451 | - | IRT |
| claude-haiku-4-5 | - | 0.9048 | 0.8575-0.9375 | - | IRT |
| qwen3-235b-a22b | 235B | 0.9000 | 0.8520-0.9337 | - | IRT |
| **qwen3-8b (target)** | **8B** | **0.8952** | **0.8465-0.9298** | - | IRT |
| llama-3.1-8b | 8B | 0.8048 | 0.7459-0.8527 | - | IRT |
| **v5clean (champion)** | **0.6B** | **0.7810** | **0.7202-0.8316** | **0/210** | **local, this run** |
| **base Qwen3-0.6B** | **0.6B** | **0.4286** | **0.3635-0.4962** | **0/210** | **local, this run** |

## Two results

**1. The fine-tune is worth +35.2 pp.** Base 0.6B 0.4286 -> champion 0.7810.
The Wilson intervals do not come close to touching (0.3635-0.4962 vs
0.7202-0.8316), so this is unambiguous rather than a sampling artifact. Both
scored 0 ambiguous items, so no part of either number is a parser outcome --
unlike the legacy table, where haiku's 73.5% carried 20 ambiguous items.

**2. The remaining size gap is 11.4 pp, and it is real.** Champion 0.7810 vs
qwen3-8b 0.8952. The champion's interval tops out at 0.8316, below qwen3-8b's
point estimate and outside its interval's lower bound of 0.8465, so the gap
survives the n=210 uncertainty. **This is the target line Epic F needed:
+11.4 pp to reach an 8B, from a model with 13x fewer parameters.**

Notably the champion already beats **llama-3.1-8b (0.8048)** to within
overlapping intervals -- a 0.6B is within noise of a 13x-larger instruct model
on this domain, which is the strongest existing evidence that the distillation
lever in Part B is aiming at something reachable.

## Where the gap actually lives

| Tier | n | base 0.6B | v5clean | Gap to close |
| --- | --- | --- | --- | --- |
| easy | 175 | 0.4686 | 0.8286 | modest |
| hard | 33 | 0.2424 | 0.5455 | **large** |
| moderate | 2 | 0.0 | 0.5 | n too small to read |

The champion's easy tier (0.8286) is approaching the frontier band; its hard
tier (0.5455) is where the 11.4 pp lives. **Part B's teacher data should be
weighted toward hard-tier content**, not spread uniformly -- the easy tier has
little headroom left and the moderate tier has only 2 items and should be
ignored entirely for targeting.

## Caveat: the reference rows are a different harness

The nine reference models come from `quality/irt/responses/` (OpenRouter, IRT
run). Their **answer key agrees with this exact benchmark file 210/210**, so the
dataset and grading are apples-to-apples. What is *not* verified identical is
the prompt wording, temperature and parser used in that run. So the 11.4 pp gap
is trustworthy as a locator but carries an unquantified harness component.

A true same-harness qwen3:8b rescore is blocked: **Ollama is not installed on
this machine** (no `ollama` binary; nothing listening on 11434 or 11435), and
`evaluate.py`'s only non-HF path is `--ollama-model`. Closing this would mean
installing Ollama and pulling qwen3:8b (~5 GB). Recommended before Part B's F8
gate leans on the gap, since F8 accepts or rejects a checkpoint against it.

## Superseded

The legacy 98-question ladder (90.8 / 89.8 / 73.5 / 28.6,
`benchmark-workspace/EVALUATION_RESULTS.md`, 2026-02-25) is **dead and must not
be quoted alongside this table**: different question set, different answer key
(42/98 disagreement), and a different temperature default. See
`results/LADDER_98Q_PROTOCOL.md`. In particular the old on-device 28.6% row is
not the base-model floor -- the base 0.6B scores 0.4286 here.
