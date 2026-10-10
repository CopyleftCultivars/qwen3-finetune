#!/usr/bin/env python3
"""Generate MCQ training pairs with CoT rationale from QMD topic collections.

Walks qmd/markdown_corpus-compatible directories such as $QMD_DIR (1854 subtopic collections), filters to
NF-relevant documents (natural farming, agroecology, permaculture, soil science,
composting, bioaccumulators, etc.), and calls Claude to generate 3-5 MCQ questions
per collection with genuine rationale in the <think>...</think> block.

Cannabis-only collections (no NF agricultural content) are skipped.

Uses `claude -p` subprocess so no API key or .env needed — Claude Code OAuth
credentials are used automatically.

NOTE: This script is Claude-based. It does not use the local `qmd` binary for MCQ
generation and should be skipped when external LLM generation is not approved.

*** BENCHMARK OVERLAP RISK (see MCQ_DIAGNOSTIC_PLAN.md, 2026-07-09) ***
benchmark-eval-workspace's original MCQ generator (generate_mcqs_rag.py) retrieves from the
same qtmd_workspace/markdown_corpus this script walks. Do not run this script for v3+ training
data without first deduping its output against benchmark_combined.jsonl question text — exact
match at minimum, ideally near-duplicate/embedding similarity too. Per the redo plan, this
source stream is deferred entirely until a clean (non-QMD) finetune is validated as solid
(Phase 6).

Shutdown: Ctrl+C kills all in-flight claude subprocesses cleanly; no orphaned
processes remain after exit.

Usage:
    python scripts/generate_mcq_from_qmd.py \\
        --qmd-dir $QMD_DIR \\
        --output data/sources/qmd_mcq_generated.jsonl \\
        [--model haiku] \\
        [--max-workers 4] \\
        [--api-concurrency 1] \\
        [--max-collections 0] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

_NF_KEYWORDS = frozenset([
    "soil", "compost", "fertilizer", "fertiliser", "nutrient", "nitrogen",
    "phosphorus", "potassium", "organic", "farming", "agriculture", "crop",
    "plant", "root", "microbe", "microorganism", "fungus", "fungi", "bacteria",
    "earthworm", "humus", "mulch", "agroecology", "permaculture", "biochar",
    "biodynamic", "regenerative", "agroforestry", "cover crop", "rotation",
    "bioaccumulator", "ferment", "fermentation", "knf", "natural farming",
    "indigenous", "foliar", "drip", "irrigation", "pH", "mineral", "trace",
    "calcium", "magnesium", "sulfur", "iron", "zinc", "boron", "manganese",
    "photosynthesis", "chlorophyll", "germination", "seed", "harvest", "yield",
    "pest", "disease", "weed", "insect", "predator", "pollinator", "bee",
    "watershed", "erosion", "water retention", "mycorrhiza", "rhizosphere",
])

_CANNABIS_ONLY_KEYWORDS = frozenset([
    "cannabis", "marijuana", "thc", "cbd", "cannabinoid", "hemp cultivation",
    "trichome", "sativa", "indica", "ruderalis",
])

_SYSTEM = (
    "You are an expert in natural farming, agroecology, soil science, permaculture, "
    "and sustainable agriculture. Your task is to create high-quality multiple-choice "
    "questions based on provided text."
)

_MCQ_PROMPT = """\
Based on the following agricultural/natural farming text, generate {n} multiple-choice \
questions. Each question must:
1. Test a specific factual or conceptual point from the text
2. Have exactly 4 options (A, B, C, D) — one correct, three plausible distractors
3. Include a <think> block with a 1-3 sentence rationale explaining why the correct \
answer is right and why the others are wrong

Return ONLY a JSON array. Each element must have this exact schema:
{{
  "question": "...",
  "A": "...",
  "B": "...",
  "C": "...",
  "D": "...",
  "answer": "A|B|C|D",
  "rationale": "..."
}}

Text:
{text}

JSON array:"""

_OPTION_LETTERS = ("A", "B", "C", "D")
_RETRY_DELAYS = [5, 30, 90]
_RATE_LIMIT_KEYWORDS = frozenset(["rate limit", "too many", "overloaded", "429", "529"])

_active_procs: set[subprocess.Popen] = set()
_proc_lock = threading.Lock()


def _register_shutdown() -> None:
    """Register SIGINT/SIGTERM handlers that kill all active claude subprocesses."""
    def _handler(signum: int, frame: object) -> None:
        log.warning(
            "Signal %d received — terminating %d active claude subprocess(es)",
            signum, len(_active_procs),
        )
        with _proc_lock:
            for proc in list(_active_procs):
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _call_claude(
    prompt: str,
    model: str,
    api_sem: threading.Semaphore,
    timeout: int = 120,
) -> str:
    """Call claude -p in a tracked subprocess and return the result text."""
    full_prompt = f"{_SYSTEM}\n\n{prompt}"
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            log.info("Retry %d — waiting %ds before next attempt", attempt, delay)
            time.sleep(delay)

        with api_sem:
            proc = subprocess.Popen(
                ["claude", "-p", "--model", model, "--tools", "",
                 "--output-format", "json", full_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            with _proc_lock:
                _active_procs.add(proc)

            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                with _proc_lock:
                    _active_procs.discard(proc)
                last_exc = TimeoutError(f"claude -p timed out after {timeout}s")
                log.warning("Timeout on attempt %d", attempt + 1)
                continue
            else:
                with _proc_lock:
                    _active_procs.discard(proc)

            if proc.returncode != 0:
                err_text = (stderr or stdout).strip()
                last_exc = subprocess.CalledProcessError(
                    proc.returncode, "claude", stdout, stderr
                )
                if any(kw in err_text.lower() for kw in _RATE_LIMIT_KEYWORDS):
                    log.warning(
                        "Rate limit on attempt %d: %s", attempt + 1, err_text[:120]
                    )
                else:
                    log.warning(
                        "claude -p error (attempt %d, rc=%d): %s",
                        attempt + 1, proc.returncode, err_text[:120],
                    )
                continue

            try:
                return json.loads(stdout)["result"].strip()
            except (json.JSONDecodeError, KeyError) as exc:
                last_exc = exc
                log.warning(
                    "Parse error on attempt %d: %s | stdout: %s",
                    attempt + 1, exc, stdout[:120],
                )
                continue

    raise last_exc


def _read_collection_text(collection_dir: Path, max_chars: int = 3000) -> str | None:
    """Read markdown content from a QMD collection directory."""
    md_files = sorted(collection_dir.glob("*.md"))
    if not md_files:
        md_files = sorted(collection_dir.glob("**/*.md"))
    if not md_files:
        return None

    parts: list[str] = []
    total = 0
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
        except OSError:
            continue

    if not parts:
        return None
    combined = "\n\n".join(parts)
    return combined[:max_chars]


def _is_nf_relevant(text: str) -> bool:
    """Return True if text contains NF agricultural content."""
    lower = text.lower()
    nf_hits = sum(1 for kw in _NF_KEYWORDS if kw in lower)
    cannabis_hits = sum(1 for kw in _CANNABIS_ONLY_KEYWORDS if kw in lower)
    return nf_hits >= 3 or (nf_hits > 0 and nf_hits > cannabis_hits)


def _generate_mcq(text: str, n: int, model: str, api_sem: threading.Semaphore) -> list[dict]:
    prompt = _MCQ_PROMPT.format(n=n, text=text)
    raw = _call_claude(prompt, model, api_sem)
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON array in response: {raw[:200]}")
    items = json.loads(json_match.group())
    if not isinstance(items, list):
        raise ValueError("Expected JSON array")
    return items


def _validate_and_convert(item: dict, source_collection: str) -> dict | None:
    """Convert raw MCQ dict to instruction/output training pair."""
    required = {"question", "A", "B", "C", "D", "answer", "rationale"}
    if not required.issubset(item.keys()):
        return None
    letter = str(item["answer"]).strip().upper()
    if letter not in _OPTION_LETTERS:
        return None
    question = str(item["question"]).strip()
    opts = {ltr: str(item[ltr]).strip() for ltr in _OPTION_LETTERS}
    rationale = str(item["rationale"]).strip()
    if not question or not rationale or any(not v for v in opts.values()):
        return None

    options_block = "\n".join(f"{ltr}. {opts[ltr]}" for ltr in _OPTION_LETTERS)
    instruction = f"{question}\n\n{options_block}"
    output = f"<think>\n{rationale}\n</think>\n{letter}"

    return {
        "instruction": instruction,
        "output": output,
        "source_collection": source_collection,
    }


def _load_done_collections(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                col = rec.get("source_collection")
                if col:
                    done.add(col)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line in resume file: %s", exc)
    return done


def process_collection(
    collection_dir: Path,
    model: str,
    n_per_collection: int,
    api_sem: threading.Semaphore,
) -> list[dict]:
    """Process one QMD collection; return list of training pairs (may be empty)."""
    text = _read_collection_text(collection_dir)
    if not text:
        return []
    if not _is_nf_relevant(text):
        return []
    items = _generate_mcq(text, n_per_collection, model, api_sem)
    pairs: list[dict] = []
    for item in items:
        converted = _validate_and_convert(item, collection_dir.name)
        if converted:
            pairs.append(converted)
    return pairs


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qmd-dir",
                    default=os.environ.get("QMD_DIR", "."),
                    help="QMD markdown_corpus directory")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--model", default="haiku",
                    help="Claude model alias for claude -p (default: haiku)")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Thread pool size (default: 4)")
    ap.add_argument("--api-concurrency", type=int, default=1,
                    help="Max concurrent claude -p API calls (default: 1, sequential)")
    ap.add_argument("--n-per-collection", type=int, default=4,
                    help="MCQ questions to generate per collection (default: 4)")
    ap.add_argument("--max-collections", type=int, default=0,
                    help="Max collections to process (0 = all, default: 0)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip collections already written to output")
    args = ap.parse_args()

    _register_shutdown()

    qmd_path = Path(args.qmd_dir)
    if not qmd_path.is_dir():
        log.error("QMD directory not found: %s", qmd_path)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collections = sorted(d for d in qmd_path.iterdir() if d.is_dir())
    if args.max_collections > 0:
        collections = collections[: args.max_collections]
    log.info("Found %d collections in %s", len(collections), qmd_path)

    done_collections: set[str] = set()
    if args.resume:
        done_collections = _load_done_collections(output_path)
        log.info("Resuming: %d collections already done", len(done_collections))

    api_sem = threading.Semaphore(args.api_concurrency)
    write_lock = threading.Lock()
    stats = {"processed": 0, "skipped_irrelevant": 0, "skipped_resume": 0,
             "pairs_written": 0, "errors": 0}

    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode, encoding="utf-8")  # noqa: WPS515

    def process(col_dir: Path) -> None:
        if col_dir.name in done_collections:
            with write_lock:
                stats["skipped_resume"] += 1
            return
        try:
            pairs = process_collection(col_dir, args.model, args.n_per_collection, api_sem)
            with write_lock:
                stats["processed"] += 1
                if not pairs:
                    stats["skipped_irrelevant"] += 1
                else:
                    for pair in pairs:
                        out_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    stats["pairs_written"] += len(pairs)
                    if stats["processed"] % 20 == 0:
                        out_fh.flush()
                        log.info(
                            "Progress: %d processed, %d pairs, %d irrelevant, %d errors",
                            stats["processed"], stats["pairs_written"],
                            stats["skipped_irrelevant"], stats["errors"],
                        )
        except Exception as exc:
            log.warning("Error on collection %s: %s", col_dir.name, exc)
            with write_lock:
                stats["errors"] += 1

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, c): c for c in collections}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                log.error("Unexpected error for %s: %s", futures[fut].name, exc)

    out_fh.flush()
    out_fh.close()

    log.info(
        "Done. Collections processed: %d | Pairs written: %d | "
        "Irrelevant skipped: %d | Errors: %d → %s",
        stats["processed"], stats["pairs_written"],
        stats["skipped_irrelevant"], stats["errors"], output_path,
    )
    print(f"\nCollections processed: {stats['processed']}")
    print(f"MCQ pairs written:     {stats['pairs_written']}")
    print(f"Irrelevant skipped:    {stats['skipped_irrelevant']}")
    print(f"Errors:                {stats['errors']}")
    print(f"Output:                {output_path}")


if __name__ == "__main__":
    main()
