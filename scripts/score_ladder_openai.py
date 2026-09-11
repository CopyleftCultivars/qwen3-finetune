"""Score the 210-item ladder through an OpenAI-compatible chat endpoint.

This is the OpenAI-compatible transport adapter for Epic G1. The benchmark
prompt, system prompt, and answer parser are imported from the Epic F
benchmark workspace. Ollama is the default server because it is already the
project's working inference path; vLLM remains supported with ``--backend``.

Example::

    python scripts/score_ladder_openai.py \
        --base-url http://127.0.0.1:11434/v1 \
        --model qwen3:8b \
        --limit 1 \
        --out-prefix results/ladder210_ollama_smoke

The endpoint must implement ``POST /v1/chat/completions``. Requests are
greedy (temperature 0), use seed 42 by default, and request no reasoning by
default. Ollama receives ``reasoning_effort=none``; vLLM receives its native
``chat_template_kwargs`` control.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_WORKSPACE = REPO_ROOT.parent / "benchmark-workspace"
DEFAULT_BENCHMARK = (
    BENCHMARK_WORKSPACE
    / "benchmarks/natural_fertilizers/natural_fertilizers_benchmark_v1_graded.jsonl"
)
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_SECONDS = 300.0
RETRY_DELAYS_SECONDS = (0.0, 2.0, 5.0, 15.0)


def import_benchmark_protocol(
    benchmark_workspace: Path = BENCHMARK_WORKSPACE,
) -> tuple[str, str, Any]:
    """Import Epic F's prompt and parser without copying their definitions."""
    workspace = str(benchmark_workspace.resolve())
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    import evaluate as benchmark_evaluate  # noqa: E402

    return (
        benchmark_evaluate.SYSTEM_PROMPT,
        benchmark_evaluate.PROMPT_TEMPLATE,
        benchmark_evaluate.extract_answer,
    )


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson 95% interval used by the Epic F ladder report."""
    if total == 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def build_payload(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    seed: int,
    enable_thinking: bool,
    backend: str = "ollama",
) -> dict[str, Any]:
    """Build a deterministic Ollama or vLLM chat-completions request."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if backend == "ollama":
        payload["reasoning_effort"] = "high" if enable_thinking else "none"
    elif backend == "vllm":
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return payload


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, Any]:
    """POST JSON and return a decoded object, preserving useful error text."""
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"endpoint returned HTTP {error.code}: {detail[:500]}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"endpoint request failed: {error}") from error
    if not isinstance(decoded, dict):
        raise RuntimeError("endpoint returned JSON that is not an object")
    return decoded


def _post_with_retries(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, Any]:
    """Retry transient endpoint failures with bounded, visible backoff."""
    last_error: RuntimeError | None = None
    for attempt, delay in enumerate(RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            return _post_json(url, payload, timeout_seconds, api_key)
        except RuntimeError as error:
            last_error = error
            if attempt == len(RETRY_DELAYS_SECONDS) - 1:
                break
    assert last_error is not None
    raise last_error


def _response_text(response: dict[str, Any]) -> str:
    """Extract assistant text from a chat response, including Qwen reasoning."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"chat response has no choices[0].message: {response}") from error
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning
    raise RuntimeError(f"chat response has no assistant text: {response}")


def query_chat_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    api_key: str | None,
) -> tuple[str, dict[str, Any]]:
    """Query one chat completion and return its text plus the raw response."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    response = _post_with_retries(endpoint, payload, timeout_seconds, api_key)
    return _response_text(response), response


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load ladder rows, optionally truncating for a smoke test."""
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit is not None else rows


def score_rows(
    rows: list[dict[str, Any]],
    system_prompt: str,
    prompt_template: str,
    extract_answer: Any,
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Score rows and stream per-item records to a resume-friendly JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            user_prompt = prompt_template.format(
                question=row["question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"]
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            payload = build_payload(
                args.model,
                messages,
                args.max_new_tokens,
                args.seed,
                args.enable_thinking,
                args.backend,
            )
            started = time.perf_counter()
            raw, response = query_chat_completion(
                args.base_url, payload, args.timeout, args.api_key
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            predicted = extract_answer(raw)
            gold = str(row["answer"]).strip().upper()
            record = {
                "question_id": row["question_id"],
                "correct_answer": gold,
                "model_answer": predicted,
                "raw_response": raw[:4000],
                "correct": predicted is not None and predicted == gold,
                "ambiguous": predicted is None,
                "difficulty": row.get("difficulty"),
                "topic": row.get("topic"),
                "latency_ms": round(elapsed_ms, 1),
                "server_model": response.get("model"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(record)
            print(f"  [{args.label}] {index}/{len(rows)}", flush=True)
    return results


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Build the ladder summary with the same Wilson interval as Epic F."""
    total = len(results)
    correct = sum(bool(row["correct"]) for row in results)
    ambiguous = sum(bool(row["ambiguous"]) for row in results)
    low, high = wilson_interval(correct, total)
    return {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "benchmark": args.benchmark,
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "wilson95": [round(low, 4), round(high, 4)],
        "ambiguous": ambiguous,
        "ambiguous_rate": round(ambiguous / total, 4) if total else 0.0,
        "enable_thinking": args.enable_thinking,
        "temperature": 0.0,
        "seed": args.seed,
        "mean_latency_ms": round(
            statistics.mean(row["latency_ms"] for row in results), 1
        )
        if results
        else 0.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the G1 transport and scoring options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("ollama", "vllm"), default="ollama")
    parser.add_argument("--label", default="qwen3-8b-ollama")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--out-prefix", default="results/ladder210_ollama")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the OpenAI-compatible ladder scorer."""
    args = parse_args(argv)
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise SystemExit(f"benchmark not found: {benchmark_path}")
    system_prompt, prompt_template, extract_answer = import_benchmark_protocol()
    rows = load_rows(benchmark_path, args.limit)
    print(f"{len(rows)} ladder items from {benchmark_path.name}", flush=True)
    output_path = Path(f"{args.out_prefix}_{args.label}.jsonl")
    results = score_rows(rows, system_prompt, prompt_template, extract_answer, args, output_path)
    summary = summarize(results, args)
    summary_path = Path(f"{args.out_prefix}_{args.label}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
