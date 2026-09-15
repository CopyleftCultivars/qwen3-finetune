"""Static, pre-generation, query-based tool routing (#175, tier 1 of #172).

Faithful Python port of cactus-engine/src/rag.cpp's `select_relevant_tools`:
same tokenizer, same BM25 formula/constants, same RRF fusion of an embedding
-cosine ranking and a BM25 ranking. Ported rather than reinvented so the GPU
experiment measures the same routing behavior #177 will later get for free
from the on-device engine -- a Tier 1 result here should transfer, not need
re-validating against a different algorithm on-device.

Only the two *domain* tools (#174) are routed here. Retrieval stays a fixed,
always-executed step exactly as it already is in Arm R (LOCAL_RLM_SLICE_PLAN.md
sec.3.3's "fixed pipeline") -- this mirrors Cactus's own architecture, where
RAG-context injection (`retrieve_rag_context`) and tool selection
(`select_relevant_tools`) are two separate mechanisms, not one.

The controller calls this, not the model: per G9 (#156), the 0.6B champion
cannot reliably self-initiate a tool call, so Tier 1 must not require it to.
The router decides *which* domain tools are relevant to a query; argument
extraction and execution are a separate, controller-owned step (see
`multi_tool_context.py`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

RRF_K = 60.0
RRF_EMB_WEIGHT = 0.8
RRF_BM25_WEIGHT = 0.2
BM25_K1 = 1.5
BM25_B = 0.75

_WORD_RE = re.compile(r"[a-z0-9]+")
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # matches mcq_generation/quality/filter_contamination_dedup.py


def tokenize_words(text: str) -> list[str]:
    """cactus-engine's tokenize_words: alnum runs, lowercased, length > 2."""
    return [w for w in _WORD_RE.findall(str(text).lower()) if len(w) > 2]


def tool_to_text(tool: dict[str, Any]) -> str:
    """cactus-engine's tool_to_text: name + " " + description, verbatim."""
    return f"{tool.get('name', '')} {tool.get('description', '')}"


def compute_bm25_score(
    query_words: list[str],
    doc_content: str,
    avg_doc_len: float,
    doc_freqs: dict[str, int],
    total_docs: int,
) -> float:
    doc_words = tokenize_words(doc_content)
    doc_len = float(len(doc_words))
    tf: dict[str, int] = {}
    for w in doc_words:
        tf[w] = tf.get(w, 0) + 1

    score = 0.0
    for term in set(query_words):
        term_freq = tf.get(term)
        if term_freq is None:
            continue
        df = float(doc_freqs.get(term, 1))
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_component = (term_freq * (BM25_K1 + 1.0)) / (
            term_freq + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len))
        )
        score += idf * tf_component
    return score


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0


def compute_rrf_scores(
    emb_ranked: list[tuple[float, int]],
    bm25_ranked: list[tuple[float, int]],
    num_items: int,
) -> list[tuple[float, int]]:
    emb_sorted = sorted(emb_ranked, key=lambda p: p[0], reverse=True)
    bm25_sorted = sorted(bm25_ranked, key=lambda p: p[0], reverse=True)
    emb_rank = {idx: r + 1 for r, (_score, idx) in enumerate(emb_sorted)}
    bm25_rank = {idx: r + 1 for r, (_score, idx) in enumerate(bm25_sorted)}

    rrf_scored = []
    for i in range(num_items):
        rrf = RRF_EMB_WEIGHT / (RRF_K + emb_rank[i]) + RRF_BM25_WEIGHT / (RRF_K + bm25_rank[i])
        rrf_scored.append((rrf, i))
    rrf_scored.sort(key=lambda p: p[0], reverse=True)
    return rrf_scored


@lru_cache(maxsize=1)
def _embedder():
    """Lazy singleton so importing this module never requires torch/CUDA."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_EMBED_MODEL_NAME)


def _default_embed_fn(texts: list[str]) -> list[list[float]]:
    return _embedder().encode(texts, normalize_embeddings=False).tolist()


@dataclass
class ToolScore:
    name: str
    rrf_score: float
    bm25_score: float
    embedding_cosine: float


@dataclass
class RoutingResult:
    query: str
    scores: list[ToolScore] = field(default_factory=list)

    def selected(self, top_k: int | None = None, min_score: float | None = None) -> list[str]:
        ranked = sorted(self.scores, key=lambda s: s.rrf_score, reverse=True)
        if min_score is not None:
            ranked = [s for s in ranked if s.rrf_score >= min_score]
        if top_k is not None:
            ranked = ranked[:top_k]
        return [s.name for s in ranked]


def select_relevant_tools(
    query: str,
    tools: list[dict[str, Any]],
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> RoutingResult:
    """Rank every tool against `query`; the caller decides top_k/threshold.

    Always scores and returns every tool (unlike Cactus's early-return when
    `len(tools) <= top_k`) because #175's accept criterion is a per-question
    score record for every candidate, not just a final selection.

    `embed_fn` defaults to a lazily-loaded all-MiniLM-L6-v2 singleton; tests
    inject a stub so they don't require the model download / torch/CUDA.
    """
    if not tools:
        return RoutingResult(query=query, scores=[])

    texts = [tool_to_text(t) for t in tools]
    tool_words = [tokenize_words(t) for t in texts]
    query_words = tokenize_words(query)

    total_len = sum(len(w) for w in tool_words)
    avg_doc_len = total_len / len(texts) if texts else 1.0
    doc_freqs: dict[str, int] = {}
    for words in tool_words:
        for w in set(words):
            doc_freqs[w] = doc_freqs.get(w, 0) + 1

    embed = embed_fn or _default_embed_fn
    query_emb = embed([query])[0]
    tool_embs = embed(texts)

    emb_ranked = [(cosine_similarity(query_emb, emb), i) for i, emb in enumerate(tool_embs)]
    bm25_ranked = [
        (compute_bm25_score(query_words, texts[i], avg_doc_len, doc_freqs, len(texts)), i)
        for i in range(len(texts))
    ]
    rrf_scored = compute_rrf_scores(emb_ranked, bm25_ranked, len(texts))
    rrf_by_idx = dict((idx, score) for score, idx in rrf_scored)

    scores = [
        ToolScore(
            name=tools[i].get("name", f"tool_{i}"),
            rrf_score=rrf_by_idx[i],
            bm25_score=bm25_ranked[i][0],
            embedding_cosine=emb_ranked[i][0],
        )
        for i in range(len(tools))
    ]
    return RoutingResult(query=query, scores=scores)
