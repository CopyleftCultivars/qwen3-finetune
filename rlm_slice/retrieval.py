"""Controller-owned retrieval over the QMD collections (L10 / #133).

LOCAL_RLM_SLICE_PLAN.md sec.3.3.2/3.3.3, built as the *fixed pipeline* sec.3.3
recommends: the controller calls `retrieve`, the model does not. #127 established
why that is not merely a preference -- v5clean never emits a tool call on its
own, but authors a well-formed query when the call site is prefilled.

Configuration follows rlm_slice/CORPUS_DECISION.md (#132):
  * `qmd search` (BM25, no LLM) as the default mode, ~1.2 s/query.
  * Over-fetch n = 4k, then collapse by QMD docid. The same source document is
    indexed under several collections, so an undeduplicated top-5 averages ~2
    distinct documents; dedupe tripled the gold-vs-distractor margin for +70 ms.
  * `vsearch` fallback for the ~15% of questions BM25 misses entirely.
  * `qmd query` is never used: its LLM expansion would put a larger model inside
    an arm whose premise is a 0.6B on a phone.

Results are cached on disk keyed by (mode, k, query). Retrieval is already
deterministic, so the cache is for rerun cost, not correctness -- but it also
means a re-scored sweep sees byte-identical snippets, which sec.2.5 wants.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_K = 5
DEFAULT_MODE = "search"
FALLBACK_MODE = "vsearch"
OVERFETCH_FACTOR = 4
QMD_TIMEOUT_S = 120
MAX_SNIPPET_CHARS = 600

# chainlink #171: a retrieval miss must reach the model as an explicit signal,
# not silence. Cross-project finding (apart-incident-hackathon-workspace,
# chainlink #93/#99 in that workspace): a Qwen3 agent that gets nothing back
# from a failed tool call has no way to distinguish "nothing relevant exists"
# from "the tool wasn't tried", and cannot adapt. A bare rejection there was
# replaced with an informative one revealing what the agent could actually do
# next; the same principle applies here even though this call site is
# single-shot (no retry loop) -- the model should still know retrieval
# happened and came up empty, rather than silently reasoning as if it had
# never been offered reference material at all.
NO_HITS_MESSAGE = (
    "No reference material was found for this query. Answer from your own "
    "knowledge; do not claim to have consulted reference material."
)

CACHE_DIR = Path(__file__).resolve().parent / ".retrieval_cache"

# QMD prefixes each snippet with a unified-diff style locator, e.g.
# "@@ -12,2 @@ (11 before, 0 after)". Useful to a human reading search output,
# pure noise inside a 0.6B's limited context window.
_LOCATOR = re.compile(r"@@[^@]*@@\s*\([^)]*\)\s*:?\s*")


@dataclass
class Snippet:
    docid: str
    file: str
    title: str
    text: str
    score: float

    @property
    def collection(self) -> str:
        match = re.match(r"qmd://([^/]+)/", self.file or "")
        return match.group(1) if match else ""


@dataclass
class RetrievalResult:
    query: str
    snippets: list[Snippet]
    mode_used: str
    latency_ms: float
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)

    def as_context(self, max_chars: int = MAX_SNIPPET_CHARS) -> str:
        """Render snippets for injection into the Pass-1 prompt (sec.3.3.4).

        Never returns an empty string when retrieval was actually attempted:
        a miss renders as NO_HITS_MESSAGE rather than "", so the model gets an
        explicit signal instead of an absence indistinguishable from
        "retrieval wasn't offered at all" (chainlink #171).
        """
        blocks = []
        for i, snippet in enumerate(self.snippets, start=1):
            text = snippet.text[:max_chars].strip()
            if text:
                blocks.append(f"[{i}] {snippet.title}: {text}")
        if not blocks:
            return NO_HITS_MESSAGE
        return "\n\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_query": self.query,
            "retrieval_mode": self.mode_used,
            "retrieval_ms": round(self.latency_ms, 1),
            "retrieval_hits": len(self.snippets),
            "retrieval_collections": sorted({s.collection for s in self.snippets if s.collection}),
            "retrieval_cached": self.from_cache,
            "retrieval_notes": self.notes,
        }


def clean_snippet(text: str) -> str:
    return _LOCATOR.sub("", str(text)).strip()


def _cache_path(mode: str, k: int, query: str) -> Path:
    digest = hashlib.sha256(f"{mode}|{k}|{query}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def _run_qmd(mode: str, query: str, n: int) -> list[dict[str, Any]]:
    """One QMD call. Returns [] on any failure -- retrieval must not kill a sweep."""
    try:
        proc = subprocess.run(
            ["qmd", mode, query, "--json", "-n", str(n)],
            capture_output=True, text=True, timeout=QMD_TIMEOUT_S,
        )
        parsed = json.loads(proc.stdout or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception:  # noqa: BLE001
        return []


def _dedupe(hits: list[dict[str, Any]], k: int) -> list[Snippet]:
    """Best-scoring hit per docid, capped at k. See CORPUS_DECISION.md."""
    seen: dict[str, Snippet] = {}
    for hit in hits:
        docid = hit.get("docid") or hit.get("file") or ""
        if not docid or docid in seen:
            continue
        text = clean_snippet(hit.get("snippet", ""))
        if not text:
            continue
        seen[docid] = Snippet(
            docid=docid,
            file=hit.get("file", ""),
            title=str(hit.get("title", "")).strip(),
            text=text,
            score=float(hit.get("score") or 0.0),
        )
        if len(seen) >= k:
            break
    return list(seen.values())


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    mode: str = DEFAULT_MODE,
    use_cache: bool = True,
    allow_fallback: bool = True,
) -> RetrievalResult:
    """Fetch k deduplicated snippets for `query`.

    This is the function the controller calls, and the one whose name the model
    sees prefilled at the call site.
    """
    query = (query or "").strip()
    if not query:
        return RetrievalResult(query="", snippets=[], mode_used=mode, latency_ms=0.0,
                               notes=["empty_query"])

    cache_file = _cache_path(mode, k, query)
    if use_cache and cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return RetrievalResult(
            query=query,
            snippets=[Snippet(**s) for s in payload["snippets"]],
            mode_used=payload["mode_used"],
            latency_ms=payload["latency_ms"],
            from_cache=True,
            notes=payload.get("notes", []),
        )

    notes: list[str] = []
    started = time.perf_counter()
    snippets = _dedupe(_run_qmd(mode, query, k * OVERFETCH_FACTOR), k)
    mode_used = mode

    if not snippets and allow_fallback and mode != FALLBACK_MODE:
        # BM25 misses ~15% of these questions outright (CORPUS_DECISION.md).
        notes.append(f"fallback_to_{FALLBACK_MODE}")
        snippets = _dedupe(_run_qmd(FALLBACK_MODE, query, k * OVERFETCH_FACTOR), k)
        mode_used = FALLBACK_MODE

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not snippets:
        notes.append("no_hits")

    result = RetrievalResult(query=query, snippets=snippets, mode_used=mode_used,
                             latency_ms=latency_ms, notes=notes)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "snippets": [vars(s) for s in result.snippets],
            "mode_used": result.mode_used,
            "latency_ms": result.latency_ms,
            "notes": result.notes,
        }), encoding="utf-8")
    return result
