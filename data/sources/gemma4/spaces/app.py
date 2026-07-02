"""
HuggingFace Spaces Gradio chatbot — Natural Farming / KNF Assistant.

Inference backend: llama-cpp-python (CPU, GGUF q4_k_m)
Runtime: free CPU tier on HuggingFace Spaces

Required Spaces secrets (Settings → Variables):
  HF_TOKEN  — read token if the GGUF repo is private (optional)
"""

import json
import os
import gradio as gr
from llama_cpp import Llama
import knf_tools
import rag

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
GGUF_REPO = os.getenv("GGUF_REPO", "CopyleftCultivars/gemma4-natural-farming-gguf")
GGUF_FILE = os.getenv("GGUF_FILE", "gemma4.Q4_K_M.gguf")

print(f"Loading model from {GGUF_REPO} / {GGUF_FILE} ...")
llm = Llama.from_pretrained(
    repo_id=GGUF_REPO,
    filename=GGUF_FILE,
    n_ctx=4096,
    n_threads=4,
    embedding=True,   # enables create_embedding() for RAG on same instance
    verbose=False,
)
print("Model ready.", flush=True)

print(f"RAG corpus exists: {rag.CORPUS_PATH.exists()} ({rag.CORPUS_PATH})", flush=True)
print(f"RAG embeddings exists: {rag.EMBEDDINGS_PATH.exists()} ({rag.EMBEDDINGS_PATH})", flush=True)

try:
    rag.init(llm)
    print("RAG init complete.", flush=True)
except Exception as e:
    print(f"RAG init FAILED: {e}", flush=True)

# ---------------------------------------------------------------------------
# Full master prompt — used at inference (full prompt is safe here; no
# seq-length pressure since we are not training weights).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert assistant specializing in Korean Natural Farming (KNF) \
and regenerative agriculture. You help farmers, gardeners, and homesteaders \
apply natural farming principles using locally available materials.

Core principles you embody:
- Work with nature's cycles, not against them
- Use indigenous microorganisms (IMO) and fermented inputs
- Build living soil through biological activity
- Minimize external inputs; maximize on-farm resources
- Respect the philosophy of Masanobu Fukuoka and Han-Kyu Cho

You answer questions about:
- KNF inputs: IMO, FPJ, OHN, LAB, WCA, FFJ, BRV, and more
- Preparation methods, application rates, and timing
- Soil health, composting, and fermentation
- Integrating KNF with permaculture and regenerative practices
- Troubleshooting crop and animal health naturally

Always be practical, encouraging, and rooted in natural farming philosophy.\
"""

# ---------------------------------------------------------------------------
# Chat function
# ---------------------------------------------------------------------------
MAX_TOKENS = 512
N_CTX = 4096


def _count_tokens(text: str) -> int:
    """Token count for a single text string."""
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def _messages_token_count(messages: list) -> int:
    """Sum token counts of all message contents."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, str):
            total += _count_tokens(content)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments", "") or ""
                total += _count_tokens(args)
    return total


def _build_messages(history: list, message: str, rag_context: str = "") -> list:
    system_content = SYSTEM_PROMPT
    if rag_context:
        system_content = system_content + "\n\n" + rag_context
    messages = [{"role": "system", "content": system_content}]

    for entry in history:
        if isinstance(entry, dict):
            messages.append(entry)
        elif isinstance(entry, (list, tuple)):
            user_msg = entry[0] or ""
            bot_msg = entry[1] if len(entry) > 1 else ""
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})
    return messages


def _trim_history(history: list, message: str, rag_context: str) -> list:
    """Drop oldest history pairs until total tokens fit in N_CTX - MAX_TOKENS."""
    budget = N_CTX - MAX_TOKENS - 200  # safety margin for formatting overhead

    trimmed = list(history)
    messages = _build_messages(trimmed, message, rag_context)
    total = _messages_token_count(messages)

    while total > budget and len(trimmed) >= 2:
        trimmed = trimmed[2:]  # drop oldest user/assistant pair
        messages = _build_messages(trimmed, message, rag_context)
        total = _messages_token_count(messages)

    return trimmed


def _maybe_trim_loop(messages: list) -> list:
    """Trim tool responses from middle of loop if context overflows."""
    budget = N_CTX - MAX_TOKENS - 200
    total = _messages_token_count(messages)
    if total <= budget:
        return messages

    # Drop oldest non-system, non-last-pair messages
    # Keep: system (index 0), last 2 user/assistant exchanges, current user msg
    keep_last = 4  # last 4 interleaved (user/assistant/tool)
    if len(messages) > 1 + keep_last:
        # preserve system + last N messages
        return [messages[0]] + messages[-(keep_last):]
    return messages


def chat(message: str, history: list) -> str:
    # ── Step 1: inject RAG context into system prompt (once, before any tool call) ──
    rag_context = rag.retrieve(message)
    trimmed_history = _trim_history(history, message, rag_context)
    messages = _build_messages(trimmed_history, message, rag_context)

    full_response = ""
    continuation_prompt = "continue"
    max_continuations = 3
    max_tool_rounds = 6  # safety limit on tool-call loops

    for _tool_round in range(max_tool_rounds):
        # Safety trim before each API call (tool results can bloat messages)
        messages = _maybe_trim_loop(messages)

        response = llm.create_chat_completion(
            messages=messages,
            tools=knf_tools.KNF_TOOLS,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<end_of_turn>", "<eos>"],
        )

        choice = response["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        msg = choice["message"]

        # ── Step 2: tool call loop (no RAG injection here) ──
        if finish_reason == "tool_calls" and msg.get("tool_calls"):
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": msg["tool_calls"]})
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                result_str = knf_tools.dispatch(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": result_str,
                })
            continue  # let the model process tool results

        # ── Step 3: final text response with continuation support ──
        text = (msg.get("content") or "").strip()
        full_response += text

        if finish_reason != "length":
            break

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": continuation_prompt})
        for _cont in range(max_continuations):
            messages = _maybe_trim_loop(messages)
            resp2 = llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<end_of_turn>", "<eos>"],
            )
            c2 = resp2["choices"][0]
            t2 = (c2["message"].get("content") or "").strip()
            full_response += " " + t2
            if c2.get("finish_reason") != "length":
                break
            messages.append({"role": "assistant", "content": t2})
            messages.append({"role": "user", "content": continuation_prompt})
        break

    return full_response


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
examples = [
    "What is OHN and how do I make it?",
    "How do I prepare IMO4 for soil application?",
    "What is the difference between FPJ and FFJ?",
    "How do I use LAB serum for livestock health?",
    "What inputs work best for fruiting vegetables?",
]

demo = gr.ChatInterface(
    fn=chat,
    title="🌱 Natural Farming Assistant (KNF)",
    description=(
        "Ask anything about Korean Natural Farming — inputs, preparations, "
        "application rates, soil health, and natural farming philosophy."
    ),
    examples=examples,
    cache_examples=False,

)

if __name__ == "__main__":
    demo.launch()
