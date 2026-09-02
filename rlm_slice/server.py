"""OpenAI-compatible HTTP surface over the in-process champion (L1 / #124).

LOCAL_RLM_SLICE_PLAN.md sec.3.1 wants the merged champion reachable through an
OpenAI-compatible endpoint so the `rlm` package can talk to it with
backend="openai" (Arm B, sec.3.2.2). This serves exactly that, backed by the
*same* `ModelHost` singleton the arms and the Pass-2 scorer use -- see
`model_host.py` for why this is a wrapper rather than vLLM.

Only the subset Arm B actually needs is implemented: /v1/models,
/v1/chat/completions, /v1/completions, plus /health and /stats. Streaming is
not implemented; the slice is a batch evaluation, and a streaming path would
add a second decode implementation to keep deterministic for no benefit.

Start:
  conda run -n qwen3-train python -m rlm_slice.server --port 8000

Guards: every request is forced to greedy (sec.2.5). A request carrying a
non-zero temperature or a top_p < 1 is rejected with HTTP 400 rather than
silently downgraded, so a mis-configured arm fails loudly instead of quietly
producing non-comparable numbers.
"""

from __future__ import annotations

import argparse
import time
import uuid

from flask import Flask, jsonify, request

from rlm_slice.guards import DEFAULT_SEED, assert_greedy
from rlm_slice.model_host import DEFAULT_MODEL, get_host

app = Flask(__name__)
_MAX_NEW_TOKENS = 512


def _greedy_or_400(payload: dict):
    """Enforce sec.2.5 at the HTTP boundary. Returns an error response or None."""
    try:
        assert_greedy(payload.get("temperature", 0), payload.get("top_p"))
    except ValueError as exc:
        return jsonify({"error": {"message": str(exc), "type": "invalid_request_error"}}), 400
    return None


@app.get("/health")
def health():
    host = get_host()
    return jsonify({"status": "ok", "model": host.model_path, "device": host.device})


@app.get("/stats")
def stats():
    """Decode-call and latency telemetry (sec.3.4.4), for arms driven over HTTP."""
    return jsonify(get_host().stats.as_dict())


@app.post("/stats/reset")
def stats_reset():
    get_host().reset_stats()
    return jsonify({"status": "reset"})


@app.get("/v1/models")
def list_models():
    host = get_host()
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": host.model_path,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
def chat_completions():
    payload = request.get_json(force=True) or {}
    error = _greedy_or_400(payload)
    if error:
        return error

    messages = payload.get("messages") or []
    if not messages:
        return jsonify({"error": {"message": "messages is required", "type": "invalid_request_error"}}), 400

    host = get_host()
    max_new = min(int(payload.get("max_tokens") or 256), _MAX_NEW_TOKENS)
    # Qwen3 think-on/think-off is a chat-template flag, not a sampling knob, so
    # it rides as an extra_body field rather than an OpenAI-standard one.
    enable_thinking = bool(payload.get("enable_thinking", True))
    text, n_new = host.generate_chat(messages, max_new_tokens=max_new, enable_thinking=enable_thinking)

    return jsonify(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": host.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length" if n_new >= max_new else "stop",
                }
            ],
            "usage": {"completion_tokens": n_new, "prompt_tokens": 0, "total_tokens": n_new},
        }
    )


@app.post("/v1/completions")
def completions():
    payload = request.get_json(force=True) or {}
    error = _greedy_or_400(payload)
    if error:
        return error

    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        return jsonify({"error": {"message": "prompt must be a string", "type": "invalid_request_error"}}), 400

    host = get_host()
    max_new = min(int(payload.get("max_tokens") or 256), _MAX_NEW_TOKENS)
    text, n_new = host.generate_raw(prompt, max_new_tokens=max_new)

    return jsonify(
        {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": host.model_path,
            "choices": [{"index": 0, "text": text, "finish_reason": "length" if n_new >= max_new else "stop"}],
            "usage": {"completion_tokens": n_new, "prompt_tokens": 0, "total_tokens": n_new},
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    get_host(model_path=args.model, device=args.device, seed=args.seed)
    # threaded=False: one model, one CUDA stream, and a shared greedy seed --
    # concurrent requests would interleave torch.manual_seed calls and break
    # the determinism guarantee in sec.2.5.
    app.run(host=args.host, port=args.port, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
