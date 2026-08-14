# Phase 23 results: GRPO baseline (Tier 4, RL second-stage)

**Decision: PAUSE (`gate.py --budget-key compute_budget_grpo`).** Primary delta is
not significant (`-0.0036`, `-0.018 SE`, nowhere near the `+2 SE` gate) — this
would be a clean REJECT on the merit of the effect size alone, but the gate
returns PAUSE because two guardrails (`rubric`, `grader_anchor`) were
deliberately not run for this pilot (see "What wasn't run" below). Do not read
this as v5clean remaining champion by default the way a real REJECT would mean
that; it means the promotion question wasn't fully evaluated.

## Two real bugs found and fixed mid-run

The first execution of this phase (200 GRPO steps, ~53 min) produced *exactly
identical* `tool_use_correctness` for baseline and candidate
(`0.26666666666666666`) — the reward component GRPO weights heaviest (0.4).
Root cause: TRL's `GRPOTrainer` renders prompts via
`apply_chat_template({"prompt": prompt}, tokenizer)` with no `tools=`
argument, and the dataset's `prompt` column never carried one either, so the
`<tools>...</tools>` schema block Qwen3's chat template conditionally renders
never appeared in any prompt, training or eval — the model was guessing tool
names/arguments from vague system-prompt text alone, identically in both
conditions.

Investigating that surfaced a second issue: the real completions call ~37
distinct tool names, not the ~9 the dataset card documents. About half are
legitimate plant-genomics/chemistry lookups the card just didn't enumerate;
the rest (`terminal`, `web_search`, `memory`, `todo`, `tool_search`, etc.) are
agent-harness scaffolding leaked from the generating agent's own research
process. 22.4% of non-agy rows touched one of these.

Both are fixed in `autoresearch/grpo_dataset.py`
(`TOOL_SCHEMAS`/`DOMAIN_TOOL_NAMES`, `bake_tools_into_system_message`,
`require_domain_tools_only` filtering) and verified: the baked `<tools>` block
is byte-identical to the tokenizer's own `tools=` rendering; Stage 0 reward
validation still passes (agy gap now `+0.028`, was `+0.034`). This phase's
numbers below are from the **rerun with the fix applied**; the pre-fix run's
outputs were deleted and never entered the historical record.

## What was measured

| | v5clean baseline | phase23-grpo-baseline candidate | delta |
|---|---|---|---|
| GRPO primary (mean `combined_reward`, 60 held-out prompts) | 0.6796 | 0.6761 | -0.0036 |
| `tool_use_correctness` | 0.5333 | 0.5333 | **0.0000 (exact)** |
| `content_quality_proxy` | 0.9544 | 0.9425 | -0.0119 |
| `citation_presence` | 0.2000 | 0.2000 | 0.0000 |
| `ethics_redflags` | 1.0000 | 1.0000 | 0.0000 |
| MCQ argmax accuracy (forgetting guard, `data/val_v3.jsonl`) | 0.7257 | 0.7434 | +0.0177 (improved) |
| MCQ option-set gold-LL (forgetting guard) | -0.7752 | -0.7195 | +0.0557 (improved) |
| Letter-distribution bias (forgetting guard) | clean | clean | no reintroduction |
| Compute (training-stage wall clock) | -- | 66.70 min | under the 90 min `compute_budget_grpo` |

**`tool_use_correctness` didn't move at all, even after the fix** -- not a
small effect, an *exact* tie (8/15 = 0.5333... both sides, same 60 prompts).
This forced a follow-up question: did training actually do anything? A direct
LoRA weight diff (`adapter_model.safetensors`, champion vs. phase23 adapter,
392 matching tensors) answered it: the weights **did** change -- 0.31%
overall relative L2 change, up to 3.7% on the single most-affected tensor
(`layers.2.mlp.down_proj.lora_B`) -- so this is genuine but small: **the run
was under-trained, not broken.**

## Why it was likely under-trained

- `--num-generations 8` with `--batch-size 8` means exactly **one unique
  prompt per training step** (the group of 8 completions is 8 samples of the
  same prompt). 200 steps therefore touched at most 200 distinct prompts out
  of a ~3800-row clean training pool.
- `--lr 2e-6` is conservative even by GRPO standards (TRL's own `GRPOConfig`
  default is `1e-6`; SFT's settled `lr` is `5e-5`, 25x higher).
- 32% of logged steps had `reward_std == 0` (all 8 generations for that step's
  prompt scored identically) -- those steps contribute no policy gradient at
  all, so the *effective* number of informative updates is meaningfully below
  200.
- The first ~20-30 steps showed severe transient instability (`grad_norm` up
  to ~860k, `kl` up to ~25.6k) before self-stabilizing to healthy values
  (`loss` <5, `grad_norm` <100, `kl` <20) for the remaining ~170 steps --
  plausibly the model's first exposure to the `<tools>` block (never seen in
  SFT) combined with sampling temperature 1.0 producing a few very
  low-reference-probability completions early on. `max_grad_norm=1.0`
  clipping kept it from diverging (no NaNs, clean exit), but those early
  steps' *useful* contribution to the final policy was likely small.

## What wasn't run (why this is PAUSE, not a completed REJECT)

- **Freeform rubric guard** (LLM-judged quality, `run_freeform_evaluation.py`)
  -- not run this pass; would need an OpenRouter judge call budget.
- **Grader-anchor drift guard** -- not run for the same reason.
- Both guardrails are guardrails, not the primary signal -- their absence
  doesn't change the PAUSE-worthy conclusion that the primary effect is not
  significant, but `gate.py` correctly refuses to call this a clean REJECT
  without them.
- **Reward-component weights remain untuned defaults** (`grpo_reward_weights.json`,
  still marked PILOT).
- **Eval scale**: 60 of 429 held-out prompts (14%), not the full pool -- a
  pragmatic cap for a first pilot, not a statistically powered read.

## Recommendation for a next iteration

Not "GRPO doesn't work here" -- the infrastructure is now verified correct
end-to-end (schema fix, domain filter, weight updates all confirmed real) and
MCQ competency did not regress. The likely lever to move first: **increase
prompt diversity per step** (raise `--batch-size` well above `--num-generations`,
e.g. 32/8, so each step trains on 4 distinct prompts instead of 1) and/or
**raise `--lr`** and **`--max-steps`** materially. Re-run the completion-length
and reward_std checks (`autoresearch/grpo_stage1_calibration.json`'s method)
at the new settings before committing to a full budgeted phase run.
