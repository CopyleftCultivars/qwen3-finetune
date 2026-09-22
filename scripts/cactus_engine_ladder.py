#!/usr/bin/env python3
"""#207 engine-transport 210-item ladder runner.

Drives the benchmark's own protocol (SYSTEM_PROMPT / PROMPT_TEMPLATE /
extract_answer imported from benchmark-workspace/evaluate.py -- not duplicated)
through the real Cactus ARM engine via cactus_complete, and diffs two runs
item-by-item. The container side ("run") is dependency-free (cactus + stdlib);
"prepare"/"compare" are host-side and reuse the frozen protocol.

Subcommands:
  prepare  -- benchmark JSONL -> prompts JSONL (question_id, gold, system, user)
  run      -- prompts JSONL -> answers JSONL via cactus_complete (greedy);
              supports --offset/--stride + --jsonl for sharded parallel runs
  compare  -- prompts + two answers runs -> correctness + per-item diff verdict

Reproduce (see results/F179_M3_S5S6_RESULTS.md):
  python scripts/cactus_engine_ladder.py prepare --out /tmp/ladder_prompts.jsonl
  # per engine/bundle condition, sharded across containers (see docs)
  python scripts/cactus_engine_ladder.py run --prompts /tmp/ladder_prompts.jsonl \
      --out /tmp/answers.jsonl --offset 0 --stride 8
  python scripts/cactus_engine_ladder.py compare --prompts /tmp/ladder_prompts.jsonl \
      --a /tmp/run_unpatched.jsonl --b /tmp/run_patched.jsonl --out /tmp/diff.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

BENCH_WS = Path("[LOCAL_PATH]")
DEFAULT_BENCH = BENCH_WS / "benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[1]

# The engine is only needed by "run"; keep the import lazy so prepare/compare
# work on the host without the arm64 cactus binding.
_cactus = None


def _load_cactus():
    global _cactus
    if _cactus is None:
        from cactus.bindings import cactus as c

        _cactus = c
    return _cactus


def import_protocol():
    """Pull the frozen prompt + parser out of evaluate.py (single source of truth)."""
    sys.path.insert(0, str(BENCH_WS))
    try:
        import evaluate as ev  # noqa: E402
    except ModuleNotFoundError:
        # evaluate.py imports tqdm only for its CLI progress bar; stub it so a
        # minimal host python can still borrow the protocol strings/parser.
        import types

        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda x=None, *a, **k: x
        sys.modules.setdefault("tqdm", stub)
        import evaluate as ev  # noqa: E402

    return ev.SYSTEM_PROMPT, ev.PROMPT_TEMPLATE, ev.extract_answer


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
# prepare (host)
# --------------------------------------------------------------------------- #

def cmd_prepare(args: argparse.Namespace) -> None:
    system_prompt, prompt_template, _ = import_protocol()
    rows = read_jsonl(args.benchmark)
    if args.limit:
        rows = rows[: args.limit]
    out: list[dict] = []
    for row in rows:
        user = prompt_template.format(
            question=row["question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"]
        )
        out.append(
            {
                "question_id": row["question_id"],
                "gold": str(row["answer"]).strip().upper(),
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "system": system_prompt,
                "user": user,
            }
        )
    write_jsonl(args.out, out)
    print(f"wrote {args.out} ({len(out)} prompts)")


# --------------------------------------------------------------------------- #
# run (container)
# --------------------------------------------------------------------------- #

def cmd_run(args: argparse.Namespace) -> None:
    c = _load_cactus()
    prompts = read_jsonl(args.prompts)
    model = c.cactus_init(args.bundle)
    if not model:
        raise SystemExit(f"cactus_init returned null for {args.bundle}")
    out_path = Path(args.out)
    jsonl = Path(args.jsonl) if args.jsonl else out_path
    results: list[dict] = []
    t0 = time.time()
    for index, row in enumerate(prompts):
        if args.offset is not None and index % args.stride != args.offset:
            continue
        gen_ids: list[int] = []

        def cb(text, token_id):
            gen_ids.append(int(token_id))

        t = time.time()
        res = c.cactus_complete(
            model,
            [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
            ],
            options={"temperature": 0.0, "max_tokens": args.max_tokens},
            callback=cb,
        )
        dt = time.time() - t
        raw = ""
        if isinstance(res, dict):
            raw = res.get("response") or res.get("completion") or ""
        rec = {
            "question_id": row["question_id"],
            "index": index,
            "raw_response": raw,
            "gen_ids": gen_ids,
            "n_tokens": len(gen_ids),
            "seconds": round(dt, 2),
        }
        results.append(rec)
        write_jsonl(jsonl, [rec], append=True)
        if args.verbose and len(results) % 10 == 0:
            print(f"  [{args.label}] {len(results)} done ({time.time()-t0:.0f}s)", flush=True)
    write_jsonl(out_path, results)
    c.cactus_destroy(model)
    avg = statistics.mean(r["seconds"] for r in results) if results else 0.0
    print(f"[{args.label}] {len(results)} items, {time.time()-t0:.0f}s total, {avg:.1f}s/item -> {out_path}")


# --------------------------------------------------------------------------- #
# compare (host)
# --------------------------------------------------------------------------- #

def cmd_compare(args: argparse.Namespace) -> None:
    _, _, extract_answer = import_protocol()
    prompts = {r["question_id"]: r for r in read_jsonl(args.prompts)}

    def load_answers(paths: list[Path]) -> dict[str, dict]:
        merged: dict[str, dict] = {}
        for path in paths:
            for row in read_jsonl(path):
                merged[row["question_id"]] = row
        return merged

    a = load_answers([args.a])
    b = load_answers([args.b]) if args.b else {}

    per_item: list[dict] = []
    for qid, row in prompts.items():
        gold = row["gold"]
        ra = a.get(qid, {})
        rb = b.get(qid, {})
        pa = extract_answer(ra.get("raw_response", ""))
        pb = extract_answer(rb.get("raw_response", "")) if b else None
        rec = {
            "question_id": qid,
            "gold": gold,
            "a_answer": pa,
            "a_correct": pa == gold,
            "b_answer": pb if b else None,
            "b_correct": (pb == gold) if b else None,
            "same_answer": (pa == pb) if b else None,
        }
        per_item.append(rec)

    n = len(per_item)
    ca = sum(r["a_correct"] for r in per_item)
    summary: dict = {
        "n": n,
        "a_label": args.a_label,
        "a_correct": ca,
        "a_accuracy": round(ca / n, 4) if n else 0.0,
        "a_wilson95": [round(x, 4) for x in wilson(ca, n)],
        "a_ambiguous": sum(1 for r in per_item if r["a_answer"] is None),
    }
    if b:
        cb_ = sum(r["b_correct"] for r in per_item)
        diffs = [r for r in per_item if not r["same_answer"]]
        summary.update(
            {
                "b_label": args.b_label,
                "b_correct": cb_,
                "b_accuracy": round(cb_ / n, 4) if n else 0.0,
                "b_wilson95": [round(x, 4) for x in wilson(cb_, n)],
                "b_ambiguous": sum(1 for r in per_item if r["b_answer"] is None),
                "n_differ": len(diffs),
                "differ_question_ids": [r["question_id"] for r in diffs],
                "correct_delta": cb_ - ca,
            }
        )
    out = {"summary": summary, "per_item": per_item}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    if args.out:
        print(f"wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--jsonl", type=Path, default=None, help="Incremental per-item output (crash resilience).")
    r.add_argument("--label", default="run")
    r.add_argument("--max-tokens", type=int, default=8)
    r.add_argument("--offset", type=int, default=None)
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    cp = sub.add_parser("compare")
    cp.add_argument("--prompts", type=Path, required=True)
    cp.add_argument("--a", type=Path, required=True)
    cp.add_argument("--b", type=Path, default=None)
    cp.add_argument("--a-label", default="a")
    cp.add_argument("--b-label", default="b")
    cp.add_argument("--out", type=Path, default=None)
    cp.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
