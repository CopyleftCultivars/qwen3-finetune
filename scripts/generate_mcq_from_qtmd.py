#!/usr/bin/env python3
"""Generate MCQ training pairs with CoT rationale from QTMD topic collections.

Walks qtmd_workspace/markdown_corpus/ (1854 subtopic collections), filters to
NF-relevant documents (natural farming, agroecology, permaculture, soil science,
composting, bioaccumulators, etc.), and calls Claude to generate 3-5 MCQ questions
per collection with genuine rationale in the <think>...</think> block.

Cannabis-only collections (no NF agricultural content) are skipped.

Requires ANTHROPIC_API_KEY in environment or .env.

Usage:
    python scripts/generate_mcq_from_qtmd.py \
        --qtmd-dir [LOCAL_PATH] \
        --output data/sources/qtmd_mcq_generated.jsonl \
        [--model claude-haiku-4-5-20251001] \
        [--max-workers 4] \
        [--max-collections 0] \
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
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


def _get_client():
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")
    return anthropic.Anthropic(api_key=api_key)


def _read_collection_text(collection_dir: Path, max_chars: int = 3000) -> str | None:
    """Read markdown content from a QTMD collection directory."""
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
            # Strip image references and HTML tags
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
    # Relevant if ≥3 NF keywords, or NF keywords outnumber cannabis keywords
    return nf_hits >= 3 or (nf_hits > 0 and nf_hits > cannabis_hits)


def _generate_mcq(text: str, n: int, client, model: str) -> list[dict]:
    prompt = _MCQ_PROMPT.format(n=n, text=text)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Extract JSON array — model may wrap in markdown code fences
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
    opts = {l: str(item[l]).strip() for l in _OPTION_LETTERS}
    rationale = str(item["rationale"]).strip()
    if not question or not rationale or any(not v for v in opts.values()):
        return None

    options_block = "\n".join(f"{l}. {opts[l]}" for l in _OPTION_LETTERS)
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
    client,
    model: str,
    n_per_collection: int,
) -> list[dict]:
    """Process one QTMD collection; return list of training pairs (may be empty)."""
    text = _read_collection_text(collection_dir)
    if not text:
        return []
    if not _is_nf_relevant(text):
        return []
    items = _generate_mcq(text, n_per_collection, client, model)
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
    ap.add_argument("--qtmd-dir",
                    default="[LOCAL_PATH]",
                    help="QTMD markdown_corpus directory")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Anthropic model for MCQ generation")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Parallel API calls (default: 4)")
    ap.add_argument("--n-per-collection", type=int, default=4,
                    help="MCQ questions to generate per collection (default: 4)")
    ap.add_argument("--max-collections", type=int, default=0,
                    help="Max collections to process (0 = all, default: 0)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip collections already written to output")
    args = ap.parse_args()

    qtmd_path = Path(args.qtmd_dir)
    if not qtmd_path.is_dir():
        log.error("QTMD directory not found: %s", qtmd_path)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client()

    collections = sorted(d for d in qtmd_path.iterdir() if d.is_dir())
    if args.max_collections > 0:
        collections = collections[: args.max_collections]
    log.info("Found %d collections in %s", len(collections), qtmd_path)

    done_collections: set[str] = set()
    if args.resume:
        done_collections = _load_done_collections(output_path)
        log.info("Resuming: %d collections already done", len(done_collections))

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
            pairs = process_collection(col_dir, client, args.model,
                                       args.n_per_collection)
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
