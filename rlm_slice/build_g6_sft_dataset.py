"""G6 (#153) data prep: G5 trajectories -> train.py's `messages` SFT format.

Each G5 trajectory's last RLM iteration already carries the FULL conversation
up to that turn in its `prompt` field (system + question + every prior
Turn-N marker and assistant response), because the rlm package's LocalREPL
re-sends the whole running transcript each iteration. So the complete,
clean trajectory is simply that last iteration's `prompt` plus its own
`response` appended as the final assistant turn -- no reconstruction from
scratch needed, and no risk of drifting from what the model actually saw
at inference time.

No loss masking: matches this repo's own train.py convention (single
instruction/output pairs are trained on end-to-end too, no response-only
collator), so multi-turn rows are treated the same way, not a new
convention invented for this dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_messages(row: dict) -> list[dict[str, str]] | None:
    traj = row.get("trajectory")
    if not traj or not traj.get("iterations"):
        return None
    last = traj["iterations"][-1]
    prompt = last.get("prompt")
    response = last.get("response")
    if not isinstance(prompt, list) or not response:
        return None
    messages = list(prompt) + [{"role": "assistant", "content": response}]
    for turn in messages:
        if not isinstance(turn, dict) or turn.get("role") not in {"system", "user", "assistant"}:
            return None
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            return None
    if messages[-1]["role"] != "assistant":
        return None
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/g5_trajectories_qwen3-8b_filtered.jsonl")
    parser.add_argument("--train-out", default="data/g6_rlm_sft_train.jsonl")
    parser.add_argument("--val-out", default="data/g6_rlm_sft_val.jsonl")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.input).open(encoding="utf-8")]
    examples = []
    skipped = 0
    for row in rows:
        messages = build_messages(row)
        if messages is None:
            skipped += 1
            continue
        examples.append({"messages": messages, "question_id": row.get("question_id")})

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split_idx = int(len(examples) * args.split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    for path, rows_out in ((args.train_out, train_examples), (args.val_out, val_examples)):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for ex in rows_out:
                handle.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"input rows: {len(rows)}, built: {len(examples)}, skipped: {skipped}")
    print(f"train: {len(train_examples)} -> {args.train_out}")
    print(f"val: {len(val_examples)} -> {args.val_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
