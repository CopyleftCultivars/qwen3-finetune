#!/usr/bin/env python3
"""
grpo_dataset.py - Load CopyleftCultivars/Agriculture-Agent-RL-Training-Data rollout
prompts for GRPO training, excluding sources with known reward-inflation bias.

Usage:
  from autoresearch.grpo_dataset import load_rollout_prompts
  ds = load_rollout_prompts()  # training-host only, requires `datasets`

Dataset shape (HF): {prompt, completion, reward, task, info}, ~6.19k rows, single
`train` split. GRPOTrainer generates its own completions online during training —
only `prompt` (+ `info`/`task` passthrough as reward_fn kwargs) is used as the
rollout seed set. The dataset's own `completion`/`reward` columns are the
*original* export's recorded rollout and are dropped here; they are not replayed
as training pairs.

Known bias (confirmed via full-dataset scan, 2026-08-13, n=6191):
  generator_agent  mean_reward  frac_reward==1.0
  agy              0.999        0.996   <- self-judging inflation, EXCLUDED
  hermes           0.819        0.541
  hermes_cli       0.698        0.285
`agy`-sourced rows are identified via info.generation_meta.generator_agent == "agy"
(not a top-level "source" field). Excluded by default everywhere this dataset is
touched — both as rollout seeds (so the model doesn't preferentially rehearse
agy's prompt distribution) and, separately, autoresearch/grpo_reward.py's reward
functions are written from scratch against info.reward_components' *semantics*
rather than calibrated to agy's inflated numbers. Do not re-include without a new,
falsified lever entry in levers.md's Tier 4 (RL second-stage) section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

DEFAULT_DATASET_ID = "CopyleftCultivars/Agriculture-Agent-RL-Training-Data"
EXCLUDED_GENERATOR_AGENTS = frozenset({"agy"})


def generator_agent_of(row: dict) -> str:
    return ((row.get("info") or {}).get("generation_meta") or {}).get("generator_agent", "?")


def iter_local_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_excluded_sources(
    rows: Iterable[dict],
    exclude_sources: frozenset[str] = EXCLUDED_GENERATOR_AGENTS,
) -> Iterator[dict]:
    for row in rows:
        if generator_agent_of(row) not in exclude_sources:
            yield row


def source_counts(rows: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        src = generator_agent_of(row)
        counts[src] = counts.get(src, 0) + 1
    return counts


def load_rollout_prompts(
    dataset_id: str = DEFAULT_DATASET_ID,
    local_path: Optional[Path] = None,
    exclude_sources: frozenset[str] = EXCLUDED_GENERATOR_AGENTS,
    split: str = "train",
):
    """Return a `datasets.Dataset` of rollout seed prompts, agy-excluded.

    Requires the `datasets` package (training-host only — not installed in the
    dev sandbox this file was authored in; autoresearch/grpo_reward.py and this
    module's non-`datasets` helpers above are dependency-light on purpose so
    Stage 0 validation can run without it).
    """
    import datasets  # deferred: training-host-only dependency

    if local_path is not None:
        ds = datasets.Dataset.from_list(list(iter_local_jsonl(local_path)))
    else:
        ds = datasets.load_dataset(dataset_id, split=split)

    ds = ds.filter(lambda row: generator_agent_of(row) not in exclude_sources)
    ds = ds.remove_columns([c for c in ("completion", "reward") if c in ds.column_names])
    return ds
