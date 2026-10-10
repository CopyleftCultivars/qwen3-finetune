#!/usr/bin/env python3
"""
validate_grpo_reward.py - Stage 0 falsification check for autoresearch/grpo_reward.py.

Runs combined_reward() (and each component) over real dataset rows, grouped by
info.generation_meta.generator_agent, and checks that the reimplemented reward
does NOT reproduce the `agy` source's self-judging inflation (99.6% reward==1.0
in the dataset's own recorded rewards, vs 54.1%/28.5% for hermes/hermes_cli).

If our reward_fn also scores agy rows suspiciously higher than the other two
sources, that's a signal the reward function is exploitable the same way the
original judge was -- fix the reward function before ever wiring it into
train_grpo.py, not after a GRPO run reward-hacks it live.

Usage:
  conda activate qwen3-finetune  # or any env with stdlib only, no extra deps needed
  python scripts/validate_grpo_reward.py [--sample-per-agent N] [--cache PATH]

Network: fetches rows from the HF datasets-server /rows API directly (stdlib
urllib only) -- this script intentionally has zero dependency on `datasets`/
`torch`/`trl` so it can run in a plain dev sandbox before any training-host work.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.grpo_reward import (  # noqa: E402
    combined_reward,
    tool_use_reward,
    content_quality_proxy_reward,
    citation_reward,
    ethics_redflag_reward,
)

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "CopyleftCultivars/Agriculture-Agent-RL-Training-Data"
DEFAULT_CACHE = Path(os.environ.get("GRPO_CACHE_DIR", ".cache/grpo_reward"))
RESULTS_PATH = Path(__file__).resolve().parents[1] / "autoresearch" / "grpo_stage0_reward_validation.json"


def _get(url: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise


def fetch_all_rows(page_size: int = 100, total: int = 6191) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while offset < total:
        data = _get(f"{HF_ROWS_API}?dataset={DATASET_ID}&config=default&split=train"
                     f"&offset={offset}&length={page_size}")
        batch = [r["row"] for r in data["rows"]]
        rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break
        time.sleep(0.5)
    return rows


def load_rows(cache: Path) -> list[dict]:
    if cache.exists():
        with open(cache, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    rows = fetch_all_rows()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def generator_agent_of(row: dict) -> str:
    return ((row.get("info") or {}).get("generation_meta") or {}).get("generator_agent", "?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    print(f"Loading dataset rows (cache: {args.cache}) ...", file=sys.stderr)
    rows = load_rows(args.cache)
    print(f"Loaded {len(rows)} rows.", file=sys.stderr)

    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[generator_agent_of(row)].append(row)

    report = {"n_rows": len(rows), "by_generator_agent": {}}

    for agent, agent_rows in sorted(by_agent.items()):
        completions = [r["completion"] for r in agent_rows]
        tasks = [r.get("task") for r in agent_rows]
        infos = [r.get("info") for r in agent_rows]
        prompts = [r.get("prompt") for r in agent_rows]

        tool = tool_use_reward(prompts, completions, task=tasks, info=infos)
        content = content_quality_proxy_reward(prompts, completions)
        citation = citation_reward(prompts, completions, task=tasks, info=infos)
        ethics = ethics_redflag_reward(prompts, completions)
        combined = combined_reward(prompts, completions, task=tasks, info=infos)

        original_reward = [r.get("reward") for r in agent_rows if r.get("reward") is not None]

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        report["by_generator_agent"][agent] = {
            "n": len(agent_rows),
            "original_dataset_reward_mean": round(mean(original_reward), 4),
            "reimplemented_combined_reward_mean": round(mean(combined), 4),
            "component_means": {
                "tool_use_correctness": round(mean(tool), 4),
                "content_quality_proxy": round(mean(content), 4),
                "citation_presence": round(mean(citation), 4),
                "ethics_redflags": round(mean(ethics), 4),
            },
        }

    print(json.dumps(report, indent=2))

    agy = report["by_generator_agent"].get("agy")
    others = [v["reimplemented_combined_reward_mean"] for k, v in report["by_generator_agent"].items() if k != "agy"]
    verdict = "N/A - no agy rows found"
    if agy and others:
        agy_mean = agy["reimplemented_combined_reward_mean"]
        others_mean = sum(others) / len(others)
        gap = agy_mean - others_mean
        if gap > 0.05:
            verdict = (f"FAIL: reimplemented reward still favors agy (agy={agy_mean:.3f} vs "
                       f"others_mean={others_mean:.3f}, gap={gap:+.3f}) -- do not wire into train_grpo.py yet")
        else:
            verdict = (f"PASS: reimplemented reward does not reproduce agy inflation "
                       f"(agy={agy_mean:.3f} vs others_mean={others_mean:.3f}, gap={gap:+.3f})")
    report["verdict"] = verdict
    print(f"\n{verdict}", file=sys.stderr)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
