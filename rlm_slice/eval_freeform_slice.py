"""Free-form dev track for the RLM slice (L7 / #130).

LOCAL_RLM_SLICE_PLAN.md sec.3.4.3: "Also run the free-form dev set
(eval_freeform/dev.jsonl) with the rubric, because the deploy target is
free-form question/answer, not MCQ." This is the track that matters for the
product; the MCQ track is instrumentation.

Two outputs per arm:

1. **Candidates** in exactly the row shape `eval_freeform/score_rubric.py`
   consumes (`--input`), generated with the same convention
   `autoresearch/generate_freeform_candidates.py` uses -- one user turn,
   `enable_thinking=False`, `prompt_mode=chat_assistant_reference_v1`, greedy --
   so a graded run is comparable to the existing codexv5 baseline. The baseline
   arm reproduces that script's prompt exactly; the retrieval arm prepends
   snippets and records that it did.

2. **Reference log-likelihood**, computed locally. `score_refll` measures
   P(reference_answer | prompt); conditioning the prompt on retrieved snippets
   and re-scoring the *same* reference answer asks directly whether retrieval
   makes the correct answer more likely. That needs no grader, so it is the one
   quantitative free-form signal available without an OpenRouter key.

   Caveat, and it is the scorer config's own: "If score_refll improves while
   score_rubric falls, investigate reference-surface learning before accepting
   the run." Reference-LL is a proxy, not the rubric. It cannot close #130 by
   itself.

Never reads eval_freeform/test.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from rlm_slice.guards import DEFAULT_SEED, check_not_sealed
from rlm_slice.model_host import DEFAULT_MODEL, ModelHost
from rlm_slice.retrieval import DEFAULT_K, retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_MODE = "chat_assistant_reference_v1"


def select_subset(rows: list[dict[str, Any]], items_per_type: int | None) -> list[dict[str, Any]]:
    """First N rows per type in file order -- the same deterministic rule the
    existing candidate generator uses. None means every row."""
    if not items_per_type:
        return rows
    counts: dict[str, int] = {}
    selected = []
    for row in rows:
        key = str(row.get("type", "unknown"))
        if counts.get(key, 0) < items_per_type:
            selected.append(row)
            counts[key] = counts.get(key, 0) + 1
    return selected


def build_prompt(host: ModelHost, question: str, context: str = "") -> str:
    content = (
        f"Reference material:\n{context}\n\nUsing the reference material where "
        f"relevant, answer:\n{question}"
        if context
        else question
    )
    return host.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def reference_logprob(host: ModelHost, prompt: str, reference: str) -> tuple[float, int]:
    """Mean per-token log P(reference | prompt), and the reference token count.

    Length-normalized so items of different reference lengths are comparable,
    and so the paired arm delta is not dominated by long references.
    """
    import torch

    prompt_ids = host.tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = host.tokenizer(prompt + reference, return_tensors="pt").input_ids.to(host.device)
    n_prompt = prompt_ids.shape[1]
    n_ref = full_ids.shape[1] - n_prompt
    if n_ref <= 0:
        return 0.0, 0
    with torch.no_grad():
        logits = host.model(full_ids).logits
    logprobs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = full_ids[0, 1:]
    token_lp = logprobs[torch.arange(targets.shape[0]), targets][n_prompt - 1 :]
    return float(token_lp.mean()), int(n_ref)


def run_arm(
    host: ModelHost,
    arm: str,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    k: int,
    out_path: Path,
) -> list[dict[str, Any]]:
    results = []
    with open(out_path, "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            context = ""
            retrieval_meta: dict[str, Any] = {}
            if arm == "R":
                result = retrieve(row["question"], k=k)
                context = result.as_context()
                retrieval_meta = result.as_dict()

            prompt = build_prompt(host, row["question"], context)
            answer, _n = host.generate_raw(prompt, max_new_tokens=max_new_tokens)
            ref_lp, ref_tokens = reference_logprob(host, prompt, row["reference_answer"])

            record = {
                "id": row["id"],
                "type": row.get("type"),
                "source_doc": row.get("source_doc"),
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "rubric": row["rubric"],
                "candidate_answer": answer.strip(),
                "generation_model": host.model_path,
                "prompt_mode": PROMPT_MODE,
                "enable_thinking": False,
                "slice_arm": arm,
                "mean_reference_logprob": round(ref_lp, 6),
                "reference_tokens": ref_tokens,
                "answer_chars": len(answer.strip()),
                "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
                **retrieval_meta,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            results.append(record)
            if index % 10 == 0:
                print(f"  [{arm}] {index}/{len(rows)}", flush=True)
    return results


def summarize(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "arm": arm,
        "n_items": len(rows),
        "mean_reference_logprob": round(statistics.mean(r["mean_reference_logprob"] for r in rows), 6),
        "mean_answer_chars": round(statistics.mean(r["answer_chars"] for r in rows), 1),
        "empty_answers": sum(1 for r in rows if not r["candidate_answer"]),
        "mean_wall_ms": round(statistics.mean(r["wall_ms"] for r in rows), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-input", default="eval_freeform/dev.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arms", default="base,R")
    parser.add_argument("--items-per-type", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-prefix", default="results/freeform_slice")
    args = parser.parse_args(argv)

    check_not_sealed(args.dev_input)  # sec.2.2
    sys.path.insert(0, str(REPO_ROOT))
    from eval_freeform.guards import assert_not_sealed_path  # noqa: E402

    assert_not_sealed_path(args.dev_input)  # the free-form track's own guard too

    rows = [json.loads(line) for line in open(args.dev_input, encoding="utf-8")]
    rows = select_subset(rows, args.items_per_type)
    print(f"{len(rows)} free-form dev items", flush=True)

    host = ModelHost(model_path=args.model, device=args.device, seed=args.seed)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    per_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        print(f"=== arm {arm} ===", flush=True)
        results = run_arm(host, arm, rows, args.max_new_tokens, args.k,
                          Path(f"{out_prefix}_{arm}.jsonl"))
        per_arm[arm] = results
        summary = summarize(arm, results)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    payload: dict[str, Any] = {"model": args.model, "n_items": len(rows), "arms": summaries}
    if "base" in per_arm and "R" in per_arm:
        base = {r["id"]: r["mean_reference_logprob"] for r in per_arm["base"]}
        deltas = [r["mean_reference_logprob"] - base[r["id"]] for r in per_arm["R"] if r["id"] in base]
        payload["refll_delta_R_vs_base"] = {
            "mean": round(statistics.mean(deltas), 6),
            "improved": sum(1 for d in deltas if d > 0),
            "n": len(deltas),
        }
        print(json.dumps(payload["refll_delta_R_vs_base"], indent=2), flush=True)

    Path(f"{out_prefix}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_prefix}_summary.json", flush=True)
    print("\nNOTE: rubric grading (score_rubric.py) still required to close #130; "
          "OPENROUTER_API_KEY is unset.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
