# Epic G, G1: Qwen3-8B serving and ladder transport

G1 uses the existing Ollama serving path. vLLM is optional; it is not a
benchmark prerequisite. The scorer imports `SYSTEM_PROMPT`, `PROMPT_TEMPLATE`,
and `extract_answer` from the Epic F benchmark workspace, so prompt wording and
grading stay aligned with the existing 210-item ladder results.

## Start the server

Ollama can remain running as usual. If it is not already running:

```bash
ollama serve
ollama pull qwen3:8b
```

Ollama exposes an OpenAI-compatible endpoint at
`http://127.0.0.1:11434/v1`. Its native benchmark endpoint is also supported
by the existing Epic F evaluator.

## Run the G1 smoke test

```bash
python scripts/score_ladder_openai.py \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3:8b \
  --limit 1 \
  --out-prefix results/ladder210_ollama_smoke
```

The scorer uses temperature 0, top-p 1, seed 42, and Ollama's
`reasoning_effort=none` by default. Remove `--limit 1` for the full 210-item
G2 baseline. Per-item JSONL and a summary with Wilson 95% confidence interval
are written under `results/`. For the exact native Epic F route, use:

```bash
python [LOCAL_PATH] \
  --ollama-model qwen3:8b \
  --ollama-url http://127.0.0.1:11434 \
  --benchmark [LOCAL_PATH]
```

For a vLLM endpoint, pass `--backend vllm`, use its `/v1` URL, and set the
served model name accordingly.

The transport has a network-boundary integration test in
`tests/test_openai_ladder_scorer.py`; it does not require vLLM, OpenAI's Python
package, or model weights.
