#!/usr/bin/env python3
"""
Generate MCQs via QMD corpus retrieval + OpenRouter API.

For each topic in the category's prompt file, queries the QMD collection
for the most relevant chunks across the entire corpus, then calls an
OpenRouter model to generate MCQs grounded in that cross-document context.

This replaces sequential per-document chunking with corpus-wide semantic
retrieval — producing higher-quality, topically coherent questions that
draw on evidence from multiple source documents.

Requires:
  - qmd CLI in PATH (run `bash index_corpus.sh` first to build collections)
  - OPENROUTER_API_KEY env var (or --openrouter-key flag)
  - openai Python package (pip install openai)

Usage:
    conda run -n nf-benchmark python generate_mcqs_rag.py \\
        --category "Permaculture" \\
        --collection Permaculture \\
        --output-dir mcq_outputs/permaculture \\
        --prompt-file prompts/prompt_for_permaculture.txt \\
        --prefix PC

    # Dry run (first 3 topics only):
    conda run -n nf-benchmark python generate_mcqs_rag.py ... --dry-run

    # Different model:
    ... --model anthropic/claude-opus-4.6 --fallback-model xiaomi/mimo-v2-flash
"""

import argparse
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Quality pipeline modules (same repo)
sys.path.insert(0, str(Path(__file__).parent / "quality"))
from prompt_injector import inject_feedback, load_feedback_context  # noqa: E402

# Load .env from the workspace root (same directory as this script)
load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "xiaomi/mimo-v2.5-pro"
FALLBACK_MODEL = "xiaomi/mimo-v2.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_QMD_RESULTS = 20     # chunks retrieved per topic query
DEFAULT_QPT = 15             # questions per topic
QMD_TIMEOUT = 45             # seconds for qmd subprocess
API_TIMEOUT = 180            # seconds for OpenRouter API call
RETRY_DELAYS = [3, 10, 30]   # seconds before each retry
INTER_TOPIC_DELAY = 2.0      # seconds between topics


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(prefix: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(__file__).parent / "logs" / f"generate_mcqs_rag_{prefix}_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("generate_mcqs_rag")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

def parse_topics_from_prompt(prompt_file: Path) -> list[str]:
    """Extract topic slugs from the TOPIC CLASSIFICATION section of the prompt file."""
    text = prompt_file.read_text(encoding="utf-8")
    match = re.search(
        r"TOPIC CLASSIFICATION[^\n]*\n.*?:\s*\n(.*?)(?=\n\s*\d+\.|OUTPUT FORMAT|\Z)",
        text, re.DOTALL
    )
    if not match:
        raise ValueError(f"Could not find TOPIC CLASSIFICATION section in {prompt_file}")
    block = match.group(1)
    topics = re.findall(r"^\s+-\s+(\S+)", block, re.MULTILINE)
    if not topics:
        raise ValueError(f"No topics found in TOPIC CLASSIFICATION block of {prompt_file}")
    return topics


def extract_system_prompt(prompt_file: Path) -> str:
    """Extract the system prompt (text before <doc>) from the prompt file."""
    text = prompt_file.read_text(encoding="utf-8")
    if text.startswith("System:"):
        text = text[len("System:"):].lstrip("\n")
    doc_start = text.find("<doc>")
    if doc_start > 0:
        return text[:doc_start].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# QMD retrieval
# ---------------------------------------------------------------------------

_SNIPPET_HEADER = re.compile(r"^@@ -(\d+),(\d+) @@[^\n]*\n\n?", re.MULTILINE)
_SNIPPET_LINE_RE = re.compile(r"^@@ -(\d+),(\d+) @@")


def _clean_snippet(snippet: str) -> str:
    """Strip the @@ -line,count @@ header from a QMD snippet."""
    return _SNIPPET_HEADER.sub("", snippet, count=1).strip()


def _parse_snippet_location(snippet: str) -> tuple[int, int] | None:
    """Extract (line_number, context_lines) from a QMD snippet header.
    Returns None if the header is missing."""
    m = _SNIPPET_LINE_RE.search(snippet)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# conda environments (e.g. nf-benchmark) bundle their own Node.js, which shadows nvm's
# node on PATH when qmd's shebang script runs under `conda run`. qmd's better-sqlite3
# native module is compiled against a specific node ABI (nvm's v22.1.0) — loading it
# under a different node version crashes with a NODE_MODULE_VERSION mismatch. Force nvm's
# node bin dir to the front of PATH for every qmd subprocess call, regardless of which
# conda env is active.
_NVM_NODE_BIN = shutil.which("node") or "node"


def _qmd_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = _NVM_NODE_BIN + os.pathsep + env.get("PATH", "")
    return env


def qmd_expand_snippets(
    snippets: list[dict],
    collection: str,
    context_lines: int,
    logger: logging.Logger,
) -> list[dict]:
    """For each snippet result, use `qmd get` to fetch expanded context around
    the match point. Returns enriched results with fuller text.

    Strategy:
    - Parse the @@ -line,count @@ header to find where the match occurred
    - Fetch a window of `context_lines` before and after that point
    - Deduplicate by file (keep the best-expanded section per file)
    """
    # Group snippets by file, keeping the highest-scored one per file
    best_by_file: dict[str, dict] = {}
    for item in snippets:
        f = item.get("file", "")
        if f not in best_by_file or item.get("score", 0) > best_by_file[f].get("score", 0):
            best_by_file[f] = item

    results = []
    for file_path, item in best_by_file.items():
        snippet = item.get("snippet", item.get("text", ""))
        loc = _parse_snippet_location(snippet)

        if loc:
            match_line, _ = loc
            # Fetch a window around the match point
            start = max(1, match_line - context_lines)
            total_lines = context_lines * 2 + 10  # extra buffer
            cmd = [
                "qmd", "get", f"{file_path}:{start}",
                "-c", collection,
                "-l", str(total_lines),
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=QMD_TIMEOUT, env=_qmd_env()
                )
                if result.returncode == 0 and result.stdout.strip():
                    text = result.stdout.strip()
                    # Skip if the expanded text is too short (likely front matter)
                    if len(text) > 200:
                        results.append({
                            "file": item.get("file", ""),
                            "title": item.get("title", ""),
                            "text": text,
                            "score": item.get("score", 0),
                        })
                        logger.debug("  expanded %s around line %d (%d chars)",
                                     Path(file_path).name, match_line, len(text))
                        continue
            except subprocess.TimeoutExpired:
                logger.debug("  qmd get timed out for %s", file_path)

        # Fallback: use the original snippet if expansion failed
        text = _clean_snippet(snippet)
        if text:
            results.append({
                "file": item.get("file", ""),
                "title": item.get("title", ""),
                "text": text,
                "score": item.get("score", 0),
            })

    return results


def qmd_query(
    query: str,
    collection: str,
    n_results: int,
    logger: logging.Logger,
    expand_context: int = 0,
) -> list[dict]:
    """Run `qmd query --json --no-rerank -c <collection>` and return results.

    Returns a list of dicts with keys: file, title, text (cleaned snippet).
    Uses --no-rerank to skip the LLM reranking step (faster; RRF scores only).

    If expand_context > 0, uses `qmd get` to fetch expanded sections around
    each match point instead of returning raw snippets.
    """
    cmd = [
        "qmd", "query", query,
        "-c", collection,
        "-n", str(n_results),
        "--json",
        "--no-rerank",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=QMD_TIMEOUT, env=_qmd_env()
        )
        if result.returncode != 0:
            logger.warning("qmd query failed (exit %d): %s", result.returncode, result.stderr.strip()[:300])
            return []
        raw = result.stdout.strip()
        if not raw:
            logger.warning("qmd query returned empty output for: %s", query)
            return []

        data = json.loads(raw)
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        if not isinstance(data, list):
            logger.warning("Unexpected qmd JSON type: %s", type(data))
            return []

        if expand_context > 0:
            return qmd_expand_snippets(data, collection, expand_context, logger)

        results = []
        for item in data:
            snippet = item.get("snippet", item.get("text", ""))
            text = _clean_snippet(snippet)
            if not text:
                continue
            results.append({
                "file": item.get("file", ""),
                "title": item.get("title", ""),
                "text": text,
                "score": item.get("score", 0),
            })
        return results

    except subprocess.TimeoutExpired:
        logger.warning("qmd query timed out (%ds) for: %s", QMD_TIMEOUT, query)
        return []
    except json.JSONDecodeError as e:
        logger.warning("qmd JSON parse error for '%s': %s", query, e)
        return []


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_rag_user_prompt(
    category: str,
    topic: str,
    chunks: list[dict],
    questions_per_topic: int,
) -> str:
    """Assemble user prompt from retrieved corpus chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = Path(chunk["file"]).name if chunk["file"] else "unknown"
        title = chunk["title"] or source
        context_parts.append(f"[Excerpt {i} — {title}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return (
        f"The following excerpts were retrieved from the **{category}** corpus "
        f"as the most relevant passages for the topic: **{topic}**.\n\n"
        f"<corpus_excerpts>\n{context}\n</corpus_excerpts>\n\n"
        f"Using ONLY information present in these excerpts, generate exactly "
        f"{questions_per_topic} MCQs about the topic **{topic}**. "
        f"Use `corpus:{category}/{topic}` as the value for the `source_document` field. "
        f"The `source_passage` for each question must be copied verbatim from one of the "
        f"excerpts above. Follow all question requirements and output format from your system instructions."
    )


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def call_openrouter(
    user_prompt: str,
    system_prompt: str,
    model: str,
    client: OpenAI,
    logger: logging.Logger,
) -> str:
    """Call OpenRouter with retries. Returns raw text."""
    last_error = ""
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            logger.debug("Retry %d — waiting %ds", attempt, delay)
            time.sleep(delay)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=8192,
                timeout=API_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            last_error = f"empty response from {model}"
            logger.warning("Empty response (attempt %d)", attempt + 1)
        except Exception as e:
            last_error = str(e)
            if attempt == len(RETRY_DELAYS):
                raise RuntimeError(
                    f"OpenRouter API failed after {attempt + 1} attempts: {last_error}"
                ) from e
            logger.warning("API error (attempt %d): %s", attempt + 1, e)
    raise RuntimeError(f"OpenRouter API failed: {last_error}")


def call_with_fallback(
    user_prompt: str,
    system_prompt: str,
    primary_model: str,
    fallback_model: str,
    client: OpenAI,
    logger: logging.Logger,
) -> tuple[str, str]:
    """Try primary model, fall back on failure. Returns (text, model_used)."""
    try:
        text = call_openrouter(user_prompt, system_prompt, primary_model, client, logger)
        return text, primary_model
    except RuntimeError as e:
        logger.warning("Primary model failed (%s), trying fallback: %s", primary_model, e)
        text = call_openrouter(user_prompt, system_prompt, fallback_model, client, logger)
        return text, fallback_model


# ---------------------------------------------------------------------------
# Response parsing + normalization
# ---------------------------------------------------------------------------

def extract_json(raw: str):
    """Strip markdown fences and reasoning tags, then parse JSON.
    Returns a list or dict depending on what the model emitted."""
    text = raw.strip()
    # Strip <think>...</think> blocks from reasoning models (e.g. MiMo-V2-Pro)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract JSON from within ```json...``` or ```...``` code blocks,
    # ignoring any prose/markdown before or after the block
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # No code fences — try to find a bare JSON object/array in the text
        m2 = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if m2:
            text = m2.group(1)
    return json.loads(text)


_LETTERS = ["A", "B", "C", "D"]

# Detects question stems that reference source material directly — these should
# never appear in a self-contained benchmark question.
_SOURCE_REF_RE = re.compile(
    r"""
    (?:
        # Quoted title in source-reference context ONLY — requires a preceding
        # source-reference word so domain concepts like 'Catch and Store Energy'
        # or 'Sleeping Beauty Paradox' are not flagged when used as subject matter
        \b(?:according\s+to|from|in|per|as\s+stated\s+in|as\s+noted\s+in|as\s+described\s+in|stated\s+in|noted\s+in|described\s+in)\s+
        ['\u2018\u2019\u201c\u201d\"]
        (?-i:[A-Z])[^'\u2018\u2019\u201c\u201d\"]{3,}
        ['\u2018\u2019\u201c\u201d\"]
    ) |
    (?:
        # "excerpt(s)" always signals a source document reference
        \bexcerpts?\b
    ) |
    (?:
        # "both/the two [verb]" comparing multiple source documents
        \b(?:both|the\s+two)\s+
        (?:emphasize|highlight|discuss|describe|mention|note|state|suggest|address|explore|cover)\b
    ) |
    (?:
        # "Which book/manual/guide/document discusses/covers/recommends..."
        # — questions asking test-takers to identify the source document
        \bwhich\s+(?:book|manual|guide|handbook|textbook|document|text|publication|resource|training\s+manual)\b
    ) |
    (?:
        # "mentioned/discussed/found/listed/outlined in the manual/guide/book/..."
        \b(?:mentioned|discussed|stated|described|found|covered|outlined|presented|listed|noted|referenced)\s+
        in\s+the\s+
        (?:manual|guide|handbook|textbook|book|document|text|publication|training\s+manual|resource|excerpt|passage)\b
    ) |
    (?:
        # "in the manual/guide/handbook/training manual" used as a source reference
        \bin\s+the\s+(?:manual|guide|handbook|training\s+manual)\b
    ) |
    (?:
        # "according to the manual/guide/book/..." (non-passage variant)
        \baccording\s+to\s+the\s+
        (?:manual|guide|handbook|textbook|book|document|text|publication|training\s+manual|resource|excerpt|passage)\b
    ) |
    (?:
        # "according to the [domain] literature/research/documentation/sources" —
        # generic source-genre references, not just named documents. Found in
        # pilot testing (2026-07-10): the LLM judge's 5-dimension rubric does not
        # independently check for self-containment, so this regex is the only
        # safety net for this failure mode — worth keeping broad.
        \baccording\s+to\s+the\s+
        (?:\w+\s+){0,2}
        (?:literature|research|documentation|sources?)\b
    ) |
    (?:
        # Markdown-italicized book title, e.g. *Permaculture Two*,
        # *Permaculture: A Designers' Manual*. Found in pilot testing
        # (2026-07-10) — this pattern correlates almost perfectly with a title
        # reference in generated MCQs; a domain question essentially never
        # needs markdown italics in its stem. Unlike the quoted-title clause
        # above, this one does NOT require a preceding trigger word, since
        # "*Title*" alone (any position in the sentence) is the tell.
        \*(?-i:[A-Z])[^*]{3,}\*
    ) |
    (?:
        # "table of contents" / "sidebar" / "chapter [on/titled/called]" —
        # always signal a source-document-structure reference, never
        # legitimate domain content on their own.
        \btable\s+of\s+contents\b
        | \bsidebar\b
        | \bchapter\s+(?:on|titled|called|about)\b
    ) |
    (?:
        # Bare "the book/manual/guide 'Title'" — a source-reference noun
        # immediately before a quoted title, without one of the prepositions
        # ("according to", "from", etc.) the first clause requires.
        \bthe\s+(?:book|manual|guide|handbook|textbook)\s+
        ['‘’“”\"]
        (?-i:[A-Z])[^'‘’“”\"]{3,}
        ['‘’“”\"]
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_question(q: dict) -> dict:
    """Normalize a question dict to the exact canonical schema fields.

    Uses a whitelist — only the five canonical fields survive. This handles
    all MiMo-V2-Pro quirks (extra fields, wrong key names, list options,
    full-text correct_answer, answer/correct_answer aliases) without needing
    to enumerate every possible variant.

    Canonical output fields: question, options, correct_answer, topic, source_passage
    """
    # Alternate field name seen from xiaomi/mimo-v2.5-pro (2026-07-10): "question_text"
    # instead of "question". Normalize early so every downstream lookup of q["question"]
    # (including the empty-stem guard below) sees it.
    if "question" not in q and "question_text" in q:
        q["question"] = q["question_text"]

    # Unify options/choices → {A: str, B: str, C: str, D: str}
    opts = q.get("options") or q.get("choices") or {}
    if isinstance(opts, list):
        # Resolve integer correct_answer as 0-based index NOW, before list→dict conversion.
        # Also seen as "correct_answer_index" (2026-07-10) rather than "correct_answer".
        ca_int = q.get("correct_answer")
        if not isinstance(ca_int, int):
            ca_int = q.get("correct_answer_index")
        if isinstance(ca_int, int) and 0 <= ca_int < 4:
            q["correct_answer"] = _LETTERS[ca_int]
        opts = {_LETTERS[i]: str(v) for i, v in enumerate(opts) if i < 4}
    elif isinstance(opts, dict) and opts:
        # Remap lowercase (a→A), numeric ("1"→A), or option_x (option_a→A) keys to standard letters
        _LOWER_MAP = {"a": "A", "b": "B", "c": "C", "d": "D"}
        _NUM_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
        _OPTION_MAP = {"option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D"}
        keys = list(opts.keys())
        if all(k in _OPTION_MAP for k in keys):
            opts = {_OPTION_MAP[k]: str(v) for k, v in opts.items()}
        elif all(k in _LOWER_MAP for k in keys):
            opts = {_LOWER_MAP[k]: str(v) for k, v in opts.items()}
        elif all(k in _NUM_MAP for k in keys):
            opts = {_NUM_MAP[k]: str(v) for k, v in opts.items()}

    # If no standard options dict found, try to build one from option_a/option_b/option_c/option_d flat keys
    if not opts or all(k not in opts for k in _LETTERS):
        alt = {}
        for letter, key in [("A", "option_a"), ("B", "option_b"), ("C", "option_c"), ("D", "option_d")]:
            if key in q:
                alt[letter] = str(q.pop(key))
        if len(alt) == 4:
            opts = alt

    # Last resort: model placed A/B/C/D as direct top-level keys in the question dict
    if not opts or all(k not in opts for k in _LETTERS):
        alt = {letter: str(q[letter]) for letter in _LETTERS if letter in q}
        if len(alt) == 4:
            opts = alt

    # Final fallback: model used full-text correct_answer + distractors list
    # (no options/choices dict at all).  Rebuild A/B/C/D from the two fields,
    # assigning the correct answer to a deterministic letter position.
    if not opts or all(k not in opts for k in _LETTERS):
        distractors = q.get("distractors", [])
        ca_text = str(q.get("correct_answer") or q.get("answer") or "").strip()
        if ca_text and isinstance(distractors, list) and len(distractors) >= 3:
            pos = hash(ca_text) % 4
            dist_idx = 0
            opts = {}
            for i, letter in enumerate(_LETTERS):
                if i == pos:
                    opts[letter] = ca_text
                else:
                    opts[letter] = str(distractors[dist_idx])
                    dist_idx += 1
            # Overwrite so the letter-resolution pass finds a direct match
            q["correct_answer"] = _LETTERS[pos]

    # Fallback: model used distractor_a/distractor_b/distractor_c flat keys
    if not opts or all(k not in opts for k in _LETTERS):
        ca_text = str(q.get("correct_answer") or q.get("answer") or "").strip()
        dist_a = str(q.get("distractor_a", "")).strip()
        dist_b = str(q.get("distractor_b", "")).strip()
        dist_c = str(q.get("distractor_c", "")).strip()
        if ca_text and dist_a and dist_b and dist_c:
            pos = hash(ca_text) % 4
            distractors_flat = [dist_a, dist_b, dist_c]
            dist_idx = 0
            opts = {}
            for i, letter in enumerate(_LETTERS):
                if i == pos:
                    opts[letter] = ca_text
                else:
                    opts[letter] = distractors_flat[dist_idx]
                    dist_idx += 1
            q["correct_answer"] = _LETTERS[pos]

    # Guard: reject if options are missing any of A/B/C/D
    missing = [k for k in _LETTERS if k not in opts]
    if missing:
        raise ValueError(
            f"Options missing required keys {missing} (got keys: {list(opts.keys())})"
        )

    # Guard: reject if any option value is empty or whitespace-only
    empty_opts = [k for k in _LETTERS if not str(opts.get(k, "")).strip()]
    if empty_opts:
        raise ValueError(f"Options have empty values for keys: {empty_opts}")

    # Guard: reject if question stem is empty or whitespace-only
    question_text = str(q.get("question", "")).strip()
    if not question_text:
        raise ValueError("Question stem is empty or missing")

    # Guard: reject if question references source material (book titles, excerpts, etc.)
    m = _SOURCE_REF_RE.search(question_text)
    if m:
        raise ValueError(
            f"Question references source material ({m.group()!r}) — "
            "questions must be self-contained and not name source documents"
        )

    # Resolve correct_answer: prefer explicit letter, fall back to 'answer' alias,
    # then match full-text answer against option values.
    # Use explicit None check so integer 0 is not silently dropped as falsy.
    ca_raw = q.get("correct_answer")
    if ca_raw is None:
        ca_raw = q.get("answer")
    ca = str(ca_raw).strip().upper() if ca_raw is not None else ""
    # Handle option_a/option_b style answers
    _OPT_CA_MAP = {"option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D"}
    if ca in _OPT_CA_MAP:
        ca = _OPT_CA_MAP[ca]
    # Handle numeric answer (1→A, 2→B, 3→C, 4→D)
    _NUM_CA_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
    if ca in _NUM_CA_MAP:
        ca = _NUM_CA_MAP[ca]
    # Handle "C) 20 percent" / "C. text" / "C: text" — letter prefix + option text,
    # not just the bare letter (seen 2026-07-11).
    if ca not in _LETTERS:
        m_prefix = re.match(r"^([ABCD])[).:]\s", ca)
        if m_prefix:
            ca = m_prefix.group(1)
    if ca not in _LETTERS:
        if ca:
            # Full-text answer — find first option whose value starts with it
            ca_lower = ca.lower()
            ca = next(
                (letter for letter, val in opts.items()
                 if str(val).lower().startswith(ca_lower[:40])),
                "",
            )
        if not ca:
            raise ValueError(
                f"Cannot resolve correct_answer to A/B/C/D (got: {q.get('correct_answer')!r})"
            )

    # Return only the canonical fields — drops source_document, explanation,
    # answer, choices, question_id, and any other extras the model adds
    return {
        "question": question_text,
        "options": opts,
        "correct_answer": ca,
        "topic": str(q.get("topic", "")),
        "source_passage": str(q.get("source_passage") or q.get("source_text") or ""),
    }


def parse_response(raw: str, source_label: str) -> dict:
    """Parse and normalize the model's JSON response into canonical MCQ format.

    Accepts both the schema-compliant {"questions": [...]} object AND the
    bare array [...] that MiMo-V2-Pro sometimes returns.
    """
    data = extract_json(raw)

    # Handle single question object (model returned one question instead of a list)
    if isinstance(data, dict) and "question" in data and "options" in data:
        data = {"questions": [data], "source_document": source_label}

    # Handle top-level array (model skipped the wrapper object)
    if isinstance(data, list):
        data = {"questions": data, "source_document": source_label}

    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Response missing 'questions' list")
    if not data["questions"]:
        raise ValueError("Response contains zero questions")

    # Validate each question independently — one bad question (e.g. a
    # source-material reference) should not discard an entire topic's worth of
    # otherwise-good questions. Original behavior raised on the first bad item,
    # which was wasteful for batch generation.
    normalized: list[dict] = []
    skipped: list[str] = []
    for q in data["questions"]:
        try:
            normalized.append(normalize_question(q))
        except (ValueError, KeyError, TypeError) as e:
            skipped.append(str(e))
    if skipped:
        logging.getLogger(__name__).warning(
            "  Skipped %d/%d questions in this batch: %s",
            len(skipped), len(data["questions"]), "; ".join(skipped[:3])
        )
    if not normalized:
        raise ValueError(f"All {len(data['questions'])} questions in batch failed validation")

    data["questions"] = normalized
    data.setdefault("source_document", source_label)
    return data


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def list_pending_topics(topics: list[str], output_dir: Path) -> list[str]:
    """Return topics that have no completed .json in output_dir."""
    return [t for t in topics if not (output_dir / f"{t}.json").exists()]


# ---------------------------------------------------------------------------
# Per-topic processing
# ---------------------------------------------------------------------------

def process_topic(
    topic: str,
    category: str,
    collection: str,
    system_prompt: str,
    output_dir: Path,
    primary_model: str,
    fallback_model: str,
    client: OpenAI,
    questions_per_topic: int,
    n_results: int,
    logger: logging.Logger,
) -> dict:
    """Query QMD for topic → build prompt → call OpenRouter → save JSON."""
    query = topic.replace("_", " ")
    chunks = qmd_query(query, collection, n_results, logger, expand_context=60)

    if not chunks:
        msg = f"No QMD chunks retrieved for topic: {topic}"
        logger.warning("  %s", msg)
        (output_dir / f"{topic}.error.txt").write_text(msg, encoding="utf-8")
        return {"topic": topic, "questions_generated": 0, "model_used": primary_model, "error": msg}

    logger.debug("  %s: %d chunks retrieved", topic, len(chunks))

    user_prompt = build_rag_user_prompt(category, topic, chunks, questions_per_topic)
    source_label = f"corpus:{category}/{topic}"

    try:
        raw, model_used = call_with_fallback(
            user_prompt, system_prompt, primary_model, fallback_model, client, logger
        )
    except RuntimeError as e:
        logger.error("  Both models failed for topic %s: %s", topic, e)
        (output_dir / f"{topic}.error.txt").write_text(str(e), encoding="utf-8")
        return {"topic": topic, "questions_generated": 0, "model_used": primary_model, "error": str(e)}

    try:
        data = parse_response(raw, source_label)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("  Parse error for topic %s: %s", topic, e)
        error_path = output_dir / f"{topic}.error.txt"
        error_path.write_text(f"PARSE ERROR: {e}\n\nRAW RESPONSE:\n{raw}", encoding="utf-8")
        return {"topic": topic, "questions_generated": 0, "model_used": model_used, "error": str(e)}

    questions = data["questions"]
    for q in questions:
        if not q.get("topic"):
            q["topic"] = topic

    out = {
        "source_document": source_label,
        "category": category,
        "topic": topic,
        "generation_timestamp": datetime.now().isoformat(),
        "questions": questions,
    }
    out_path = output_dir / f"{topic}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    q_count = len(data["questions"])
    logger.info("  %s → %d questions (%s, %d chunks)", topic, q_count, model_used, len(chunks))
    return {"topic": topic, "questions_generated": q_count, "model_used": model_used, "error": None}


# ---------------------------------------------------------------------------
# Regeneration helpers
# ---------------------------------------------------------------------------

# Similarity threshold above which a new question is considered a duplicate
_DEDUP_THRESHOLD_ACCEPTED = 0.75
_DEDUP_THRESHOLD_REJECTED = 0.65


def _category_slug(category: str) -> str:
    """'Natural Fertilizers' → 'natural_fertilizers'"""
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _load_existing_stems(category: str, workspace: Path) -> tuple[list[str], list[str]]:
    """
    Return (accepted_stems, rejected_stems) — lowercased question text for
    all surviving items and all judge-rejected items in this category.
    """
    accepted: list[str] = []
    rejected: list[str] = []

    slug = _category_slug(category)

    # Accepted: surviving items from final JSONL
    final_path = workspace / "quality" / "final" / f"{slug}_final.jsonl"
    if final_path.exists():
        for line in final_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("final_status") in ("keep", "review"):
                    q = item.get("question", "")
                    if q:
                        accepted.append(q.lower())

    # Rejected: items from judge cache with verdict = reject
    cache_path = workspace / "quality" / "scores" / f"{slug}_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for rec in cache.values():
            if rec.get("quality_scores", {}).get("verdict") == "reject":
                q = rec.get("question", "")
                if q:
                    rejected.append(q.lower())

    return accepted, rejected


def _is_duplicate(
    question_text: str,
    accepted_stems: list[str],
    rejected_stems: list[str],
) -> tuple[bool, str]:
    """
    Return (is_dup, reason) for a candidate question.
    Uses SequenceMatcher ratio for fuzzy matching.
    """
    q = question_text.lower()
    for stem in accepted_stems:
        ratio = difflib.SequenceMatcher(None, q, stem).ratio()
        if ratio >= _DEDUP_THRESHOLD_ACCEPTED:
            return True, f"too similar to accepted item (ratio={ratio:.2f})"
    for stem in rejected_stems:
        ratio = difflib.SequenceMatcher(None, q, stem).ratio()
        if ratio >= _DEDUP_THRESHOLD_REJECTED:
            return True, f"too similar to rejected item (ratio={ratio:.2f})"
    return False, ""


def regenerate_main(
    args: argparse.Namespace,
    logger: logging.Logger,
    client: OpenAI,
) -> int:
    """
    Regeneration mode: targeted gap-filling driven by feedback_context.json.

    - Topics: only those in the feedback context (i.e., below MIN_ITEMS_PER_TOPIC)
    - Items per topic: feedback entry's items_needed
    - Prompt: base prompt + per-topic <feedback> block injected by prompt_injector
    - Output: regenerated_mcqs/{slug}_round{N}/{topic}.json
    - Dedup: new questions checked against existing accepted + rejected stems
    """
    workspace = Path(__file__).parent
    slug = _category_slug(args.category)

    # Load feedback context — fall back to regeneration_needed.json if missing
    feedback_context = load_feedback_context(slug)
    if not feedback_context:
        regen_report_path = workspace / "quality" / "final" / "regeneration_needed.json"
        if regen_report_path.exists():
            report = json.loads(regen_report_path.read_text(encoding="utf-8"))
            for entry in report.get("topics", []):
                if entry.get("category") == args.category:
                    feedback_context[entry["topic"]] = {
                        "items_needed": entry["items_needed"],
                        "weakest_dimensions": [],
                        "common_rejection_reasons": [],
                    }
        if feedback_context:
            logger.warning(
                "No feedback context for '%s' — falling back to regeneration_needed.json "
                "(no quality guidance injected). Run feedback_aggregator.py to enable it.",
                slug,
            )
        else:
            logger.error(
                "No feedback context and no matching topics in regeneration_needed.json "
                "for '%s'. Run quality/feedback_aggregator.py --category %s first.",
                slug, slug,
            )
            return 1

    logger.info("Feedback topics: %d", len(feedback_context))

    # Output directory
    round_num = args.round
    out_dir = args.output_dir or (
        workspace / "regenerated_mcqs" / f"{slug}_round{round_num}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base system prompt
    system_prompt_base = extract_system_prompt(args.prompt_file)

    # Load existing stems for dedup
    accepted_stems, rejected_stems = _load_existing_stems(args.category, workspace)
    logger.info(
        "Dedup pool: %d accepted, %d rejected stems loaded",
        len(accepted_stems), len(rejected_stems),
    )

    stats: dict = {
        "total_generated": 0,
        "total_accepted": 0,
        "total_deduped": 0,
        "errors": 0,
        "topics": {},
    }

    topics = sorted(feedback_context.keys())
    if args.dry_run:
        topics = topics[:3]
        logger.info("DRY RUN — processing first %d topics.", len(topics))

    for i, topic in enumerate(topics, 1):
        entry = feedback_context[topic]
        items_needed = entry.get("items_needed", args.questions_per_topic)

        logger.info(
            "[%d/%d] %s — needs %d items",
            i, len(topics), topic, items_needed,
        )

        if args.dry_run:
            logger.info("  [dry-run] would generate %d items for '%s'", items_needed, topic)
            continue

        # Check skip (output already exists)
        out_path = out_dir / f"{topic}.json"
        if out_path.exists() and not args.dry_run:
            logger.info("  [skip] output exists: %s", out_path.name)
            continue

        # Inject topic-specific feedback into system prompt
        system_prompt = inject_feedback(system_prompt_base, topic, feedback_context)

        result = process_topic(
            topic=topic,
            category=args.category,
            collection=args.collection,
            system_prompt=system_prompt,
            output_dir=out_dir,
            primary_model=args.model,
            fallback_model=args.fallback_model,
            client=client,
            questions_per_topic=items_needed,
            n_results=args.n_results,
            logger=logger,
        )

        if result["error"]:
            stats["errors"] += 1
            stats["topics"][topic] = {"generated": 0, "accepted": 0, "needed": items_needed, "error": result["error"]}
            continue

        # Load generated questions and run dedup filter
        topic_path = out_dir / f"{topic}.json"
        if not topic_path.exists():
            stats["topics"][topic] = {
                "generated": 0, "accepted": 0, "needed": items_needed,
                "error": "output file missing",
            }
            stats["errors"] += 1
            continue

        topic_data = json.loads(topic_path.read_text(encoding="utf-8"))
        questions = topic_data.get("questions", [])
        accepted_questions = []
        n_deduped = 0

        for q in questions:
            is_dup, reason = _is_duplicate(
                q.get("question", ""), accepted_stems, rejected_stems
            )
            if is_dup:
                logger.debug("  [dedup] %s: %s", q.get("question_id", "?"), reason)
                n_deduped += 1
            else:
                accepted_questions.append(q)
                # Add to pool so later items in same batch can't duplicate each other
                accepted_stems.append(q.get("question", "").lower())

        # Rewrite output with dedup-filtered questions
        topic_data["questions"] = accepted_questions
        topic_data["dedup_removed"] = n_deduped
        topic_path.write_text(
            json.dumps(topic_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        n_gen = len(questions)
        n_acc = len(accepted_questions)
        stats["total_generated"] += n_gen
        stats["total_accepted"] += n_acc
        stats["total_deduped"] += n_deduped
        stats["topics"][topic] = {"generated": n_gen, "accepted": n_acc, "needed": items_needed}
        logger.info(
            "  generated=%d  accepted=%d  deduped=%d  needed=%d",
            n_gen, n_acc, n_deduped, items_needed,
        )

        if args.delay and i < len(topics):
            time.sleep(args.delay)

    # Compute derived stats
    acceptance_rate = (
        stats["total_accepted"] / stats["total_generated"]
        if stats["total_generated"] else 0.0
    )
    topics_filled = [
        t for t, td in stats["topics"].items()
        if td["accepted"] >= td["needed"]
    ]
    topics_remaining = [
        t for t, td in stats["topics"].items()
        if td["accepted"] < td["needed"]
    ]

    # Write round stats
    stats_path = out_dir / "round_stats.json"
    stats_path.write_text(json.dumps({
        "round": round_num,
        "category": args.category,
        "model": args.model,
        "acceptance_rate": round(acceptance_rate, 4),
        "topics_filled": sorted(topics_filled),
        "topics_remaining": sorted(topics_remaining),
        **stats,
    }, indent=2), encoding="utf-8")

    logger.info("=== Regeneration Round %d Summary ===", round_num)
    logger.info("Topics processed:   %d", len(topics))
    logger.info("Total generated:    %d", stats["total_generated"])
    logger.info("Total accepted:     %d", stats["total_accepted"])
    logger.info("Total deduped:      %d", stats["total_deduped"])
    logger.info("Acceptance rate:    %.1f%%", acceptance_rate * 100)
    logger.info("Topics filled:      %d", len(topics_filled))
    logger.info("Topics remaining:   %d", len(topics_remaining))
    logger.info("Errors:             %d", stats["errors"])
    logger.info("Stats: %s", stats_path)

    # Return non-zero only on complete failure (nothing generated at all).
    # Topic-level parse errors are normal when the model returns bad JSON for
    # some topics; the successfully generated questions should still be merged.
    if stats["total_generated"] == 0 and stats["errors"] > 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate MCQs via QMD corpus retrieval + OpenRouter API"
    )
    parser.add_argument("--category", required=True,
                        help="Human-readable category name (e.g. 'Permaculture')")
    parser.add_argument("--collection", required=True,
                        help="QMD collection name (e.g. 'Permaculture')")
    parser.add_argument("--output-dir", required=False, default=None, type=Path,
                        help="Directory to write per-topic MCQ JSONs (required in generate mode)")
    parser.add_argument("--prompt-file", required=True, type=Path,
                        help="Category-specific prompt file (for topics + system prompt)")
    parser.add_argument("--prefix", required=True,
                        help="2-3 letter question ID prefix, e.g. PC")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Primary OpenRouter model (default: {DEFAULT_MODEL})")
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL,
                        help=f"Fallback model (default: {FALLBACK_MODEL})")
    parser.add_argument("--openrouter-key", default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--n-results", type=int, default=DEFAULT_QMD_RESULTS,
                        help=f"QMD chunks per topic query (default: {DEFAULT_QMD_RESULTS})")
    parser.add_argument("--questions-per-topic", type=int, default=DEFAULT_QPT,
                        help=f"MCQs to generate per topic (default: {DEFAULT_QPT})")
    parser.add_argument("--delay", type=float, default=INTER_TOPIC_DELAY,
                        help=f"Seconds between topics (default: {INTER_TOPIC_DELAY})")
    parser.add_argument("--topics", default=None, metavar="T1,T2,...",
                        help="Comma-separated topic names to generate (default: all from prompt)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 3 topics only")
    parser.add_argument(
        "--mode",
        choices=["generate", "regenerate"],
        default="generate",
        help="'generate' = normal full-category run; 'regenerate' = feedback-driven gap-filling",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=2,
        metavar="N",
        help="Regeneration round number (used in output directory name, default: 2)",
    )
    args = parser.parse_args()

    logger = setup_logging(args.prefix)
    logger.info("=== RAG MCQ Generation: %s [%s] ===", args.category, args.prefix)
    logger.info("Model: %s | Fallback: %s", args.model, args.fallback_model)

    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error(
            "OpenRouter API key required. "
            "Set OPENROUTER_API_KEY env var or pass --openrouter-key."
        )
        return 1

    if not shutil.which("qmd"):
        logger.error(
            "qmd CLI not found in PATH. "
            "Install with: npm install -g @tobi/qmd"
        )
        return 1

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    if not args.prompt_file.exists():
        logger.error("Prompt file not found: %s", args.prompt_file)
        return 1

    if args.mode == "regenerate":
        return regenerate_main(args, logger, client)

    if not args.output_dir:
        logger.error("--output-dir is required in generate mode")
        return 1

    topics = parse_topics_from_prompt(args.prompt_file)
    if args.topics:
        filter_set = {t.strip() for t in args.topics.split(",")}
        topics = [t for t in topics if t in filter_set]
        logger.info("Topic filter applied: %d topic(s): %s", len(topics), ", ".join(topics))
    system_prompt = extract_system_prompt(args.prompt_file)
    logger.info("Topics (%d): %s", len(topics), ", ".join(topics))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = list_pending_topics(topics, args.output_dir)
    skipped = len(topics) - len(pending)
    if skipped:
        logger.info("Skipping %d already-processed topics.", skipped)
    logger.info("%d topics to process.", len(pending))

    if args.dry_run:
        pending = pending[:3]
        logger.info("DRY RUN — processing first %d topics only.", len(pending))

    stats: dict = {"total_questions": 0, "errors": 0, "model_counts": {}}

    for i, topic in enumerate(pending, 1):
        logger.info("[%d/%d] %s", i, len(pending), topic)
        result = process_topic(
            topic, args.category, args.collection, system_prompt,
            args.output_dir, args.model, args.fallback_model,
            client, args.questions_per_topic, args.n_results, logger,
        )
        if result["error"]:
            stats["errors"] += 1
        else:
            stats["total_questions"] += result["questions_generated"]
            m = result["model_used"]
            stats["model_counts"][m] = stats["model_counts"].get(m, 0) + 1

        if args.delay and i < len(pending):
            time.sleep(args.delay)

    logger.info("=== Summary ===")
    logger.info("Topics processed: %d", len(pending))
    logger.info("Total questions generated: %d", stats["total_questions"])
    logger.info("Errors: %d", stats["errors"])
    for model, count in stats["model_counts"].items():
        logger.info("  %s: %d topics", model, count)

    return 1 if stats["errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
