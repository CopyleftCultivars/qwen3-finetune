"""Arm B: the real `rlm` LocalREPL at depth 1 (L4 / #127).

LOCAL_RLM_SLICE_PLAN.md sec.3.2.2. This uses the actual Recursive Language
Models package (Alex Zhang, MIT) vendored at
`../persistent-memory-workspace/rlm` -- distributed as `rlms`, which is why it
does not appear on PyPI under the name `rlm`.

Wiring notes
------------
* The package is added to sys.path rather than pip-installed. Its metadata pins
  requires-python >= 3.11 and the `qwen3-train` env is 3.10.20, but the code
  imports and runs clean on 3.10; a path import gets Arm B running without
  either rebuilding the env that produced the champion or relaxing a pin in a
  repo we do not own.
* `backend="openai"` + `base_url` points at `rlm_slice/server.py`, so the RLM
  controller drives the *same* in-process champion the Pass-2 scorer uses. The
  server rejects non-greedy sampling, so `sampling_args` here must stay at
  temperature=0 (sec.2.5).
* `max_depth=1` per the plan: one REPL level, no recursive sub-calls.
* `environment="local"` is the plan's LocalREPL. It executes model-authored
  code in-process, which is acceptable here only because the author is our own
  local 0.6B model working on our own MCQ prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

RLM_PACKAGE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "persistent-memory-workspace" / "rlm"
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# The stock RLM_SYSTEM_PROMPT is written for long-context haystack tasks: it
# describes `context` as "potentially very long information" and tells the model
# to probe it before answering. A 4-option MCQ gives it nothing to probe, and the
# 0.6B champion responded to the mismatch literally -- "I am not able to interact
# with the context" for six straight iterations, emitting zero ```repl blocks.
#
# This replacement keeps the RLM machinery (REPL, iterate-until-answer, custom
# tools) and drops the long-context framing. Notes on its shape:
#   * `{custom_tools_section}` is required: build_rlm_system_prompt() calls
#     .format() on this string, so the placeholder must exist and any literal
#     brace must be doubled.
#   * The fence must be exactly ```repl -- find_code_blocks() matches
#     r"```repl\s*\n(.*?)\n```" and nothing else.
#   * One worked example, because a 0.6B model copies format far more reliably
#     than it follows a format description.
#   * orchestrator=False is passed alongside this, dropping ORCHESTRATOR_ADDENDUM
#     -- that addendum pushes work into llm_query sub-calls, which is the
#     multi-agent behaviour we do not want in the minimal slice (sec.3.2.2).
MCQ_SYSTEM_PROMPT = """You are answering a multiple-choice question with the help of a Python REPL.

The variable `context` holds the question and its four options as a string.
The REPL persists across turns. Only `print(...)` output is shown back to you.
{custom_tools_section}

To run code, write exactly one block per turn using the ```repl fence:

```repl
print(context)
```

Work in short turns:
1. Print the question and read it carefully.
2. Use the REPL to note what each option claims and rule options out.
3. When you are confident, submit inside a ```repl block:

```repl
answer["content"] = "<your reasoning>. The best option is <LETTER>."
answer["ready"] = True
```

Submit your reasoning AND the single best option letter. Always use the ```repl
fence when you want code to run -- text outside a fence is never executed."""


def ensure_rlm_importable(package_path: Path | str = RLM_PACKAGE_PATH):
    """Put the vendored `rlm` package on sys.path and return the RLM class."""
    resolved = Path(package_path).resolve()
    if not (resolved / "rlm" / "__init__.py").exists():
        raise FileNotFoundError(
            f"rlm package not found at {resolved}. Arm B (sec.3.2.2) needs the "
            "Recursive Language Models repo; see rlm_slice/README.md."
        )
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from rlm import RLM  # noqa: E402

    return RLM


def agronomy_tools() -> dict[str, Any]:
    """Helper functions exposed inside the REPL (sec.3.2.2 "helper functions").

    `retrieve` is deliberately absent: retrieval is I13 (#118) and must be
    measured as its own arm, not folded into Arm B (sec.3.3.5).
    """

    def count_words(text: str) -> int:
        """Word count of a string."""
        return len(str(text).split())

    def compare_numbers(a: float, b: float) -> str:
        """Compare two numbers; returns 'a>b', 'a<b' or 'a==b'."""
        return "a>b" if a > b else ("a<b" if a < b else "a==b")

    return {"count_words": count_words, "compare_numbers": compare_numbers}


def build_rlm(
    model_name: str,
    base_url: str = DEFAULT_BASE_URL,
    max_iterations: int = 6,
    max_errors: int = 3,
    custom_tools: dict[str, Any] | None = None,
    verbose: bool = False,
    orchestrator: bool = False,
    system_prompt: str | None = MCQ_SYSTEM_PROMPT,
):
    """Construct the depth-1 RLM controller pointed at the local server."""
    RLM = ensure_rlm_importable()
    return RLM(
        backend="openai",
        backend_kwargs={
            "model_name": model_name,
            "base_url": base_url,
            # The local server ignores auth; the OpenAI client requires a value.
            "api_key": "local-rlm-slice",
        },
        environment="local",
        max_depth=1,
        max_iterations=max_iterations,
        max_errors=max_errors,
        custom_tools=agronomy_tools() if custom_tools is None else custom_tools,
        # sec.2.5: greedy, or the arm is not comparable to the v5clean baseline.
        sampling_args={"temperature": 0},
        # orchestrator=False drops ORCHESTRATOR_ADDENDUM, which instructs the
        # model to decompose work into llm_query sub-calls. The minimal slice
        # wants the REPL, not a multi-agent tree (sec.3.2.2).
        orchestrator=orchestrator,
        custom_system_prompt=system_prompt,
        # log_dir=None keeps the trajectory in memory only. Without a logger the
        # RLM returns metadata=None, and Arm B's REPL-usage telemetry below
        # would have nothing to read.
        logger=_in_memory_logger(),
        verbose=verbose,
    )


def _in_memory_logger():
    ensure_rlm_importable()
    from rlm.logger import RLMLogger

    return RLMLogger(log_dir=None)


def repl_telemetry(result: Any) -> dict[str, Any]:
    """Summarize what the RLM actually did, for the per-item record.

    This exists because Arm B's first run looked healthy -- non-empty response,
    no errors, notes=[] -- while emitting zero code blocks across every
    iteration. Without `repl_blocks` and `subcalls` on each row, I12 (#117)
    would report a degenerate Arm B as a working one.
    """
    metadata = getattr(result, "metadata", None) or {}
    iterations = metadata.get("iterations", []) or []
    blocks = 0
    subcalls = 0
    for iteration in iterations:
        code_blocks = iteration.get("code_blocks") or []
        blocks += len(code_blocks)
        for block in code_blocks:
            code = block.get("code", "") if isinstance(block, dict) else str(block)
            subcalls += code.count("llm_query") + code.count("rlm_query")
    return {
        "rlm_iterations": len(iterations),
        "repl_blocks": blocks,
        "repl_subcalls": subcalls,
        "submitted_answer": any(it.get("final_answer") for it in iterations),
    }


def reset_server_stats(base_url: str = DEFAULT_BASE_URL) -> None:
    """Zero the server-side decode counters before an Arm B item."""
    import requests

    root = base_url.rstrip("/").removesuffix("/v1")
    requests.post(f"{root}/stats/reset", timeout=10)


def fetch_server_stats(base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Read back the decode calls Arm B spent inside the server process.

    Arm B is the one arm whose generation happens out-of-process, so without
    this its per-item decode count and latency (sec.3.4.4) would read as zero
    and the sec.3.5.3 on-device latency estimate would be silently wrong -- in
    the direction that makes Arm B look free.
    """
    import requests

    root = base_url.rstrip("/").removesuffix("/v1")
    return requests.get(f"{root}/stats", timeout=10).json()
