"""Controller-owned multi-tool context assembly (#175, tier 1 of #172).

Extends Arm R's pattern (retrieval.py: "the controller calls `retrieve`, the
model does not" -- #127 established the 0.6B champion never reliably
self-initiates a tool call) to the two new domain tools. The controller:

  1. always retrieves (Arm R's fixed step, unchanged);
  2. extracts heuristic arguments for each domain tool from the raw query,
     by substring-matching against the known crop/plant/region vocabularies
     the domain data itself defines (no model call, no NER model -- just the
     same closed vocabularies bioaccumulator_ratio/regional_recipe_lookup
     already validate against);
  3. scores every domain tool against the query via tool_router's Cactus-
     ported hybrid ranking (kept purely for the audit record and for
     ranking among viable tools -- see below, not as a reject-all gate);
  4. determines *viability* by actually running each tool whose arguments
     are available (both are cheap, local, deterministic lookups) and
     checking whether it produced real content -- then selects/injects only
     among viable tools.

Why viability isn't decided by the router's score: Cactus's own
`select_relevant_tools` (cactus-engine/src/rag.cpp, faithfully ported in
tool_router.py) always returns exactly top_k tools by RRF rank -- there is
no minimum-score floor, no "return nothing if nothing clears a threshold"
branch. #175's first sweep found this meant Arm M was forced to commit to
one of the two domain tools on 100% of items, and 66% of the time the
picked tool had no extractable arguments at all. A score-based reject-all
threshold was tried next and rejected: pulling real cosine/BM25 distributions
across the 113 val_v3 queries, split by whether `bioaccumulator_ratio`'s
arguments were actually extractable, showed the "on-topic" group did not
even score higher than the "off-topic" group (mean cosine 0.249 vs 0.290) --
the router's relevance signal and "does this query name an actual crop+plant
pair" are only loosely correlated at this scale, so any threshold fit to
that data would be arbitrary. Running the (cheap, local) tool and checking
its actual output is a strictly more reliable signal than predicting
relevance from a score beforehand, and mirrors retrieval.py's own principle
that a miss should be a signal, not silence.

This is what answers #175's central question: does adding this context (on
top of Arm R's retrieval) help or add noise, and does routing (vs. always
injecting every tool) matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from rlm_slice.domain_data import load_crop_requirements, load_local_plants_by_region, load_plant_mineral_profiles
from rlm_slice.domain_tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from rlm_slice.retrieval import RetrievalResult, retrieve
from rlm_slice.tool_router import RoutingResult, select_relevant_tools


@lru_cache(maxsize=1)
def _crop_vocab() -> list[str]:
    names = {str(row.get("Crop", "")).strip() for row in load_crop_requirements() if row.get("Crop")}
    return sorted(names, key=len, reverse=True)


@lru_cache(maxsize=1)
def _plant_vocab() -> list[str]:
    names: set[str] = set()
    for row in load_plant_mineral_profiles():
        for key in ("common_name", "scientific_name"):
            if row.get(key):
                names.add(str(row[key]).strip())
    return sorted(names, key=len, reverse=True)


@lru_cache(maxsize=1)
def _region_vocab() -> list[str]:
    """Location strings trimmed to their core place name (before " (")."""
    names: set[str] = set()
    for row in load_local_plants_by_region():
        location = str(row.get("Location", "")).strip()
        if location:
            names.add(location.split(" (")[0].strip())
    return sorted(names, key=len, reverse=True)


def _find_in_text(text: str, vocab: list[str]) -> str | None:
    """Longest-match, case-insensitive substring search over a closed vocabulary."""
    text_l = text.lower()
    for term in vocab:  # already sorted longest-first
        if len(term) > 2 and term.lower() in text_l:
            return term
    return None


@dataclass
class ToolCallRecord:
    name: str
    rrf_score: float
    args_available: bool
    viable: bool
    selected_by_router: bool
    executed: bool
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class MultiToolContext:
    query: str
    retrieval_context: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def as_context(self) -> str:
        """Retrieval context (Arm R, unchanged) + any *selected* domain-tool blocks.

        Gated on selected_by_router, not merely executed: a tool can execute
        cleanly but turn up nothing (not viable) or be viable but lose out to
        another tool under top_k -- only a selected tool's result is actually
        injected.
        """
        blocks = [self.retrieval_context]
        for call in self.tool_calls:
            if call.selected_by_router:
                blocks.append(f"[{call.name}] {call.result}")
        return "\n\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tool_calls": [
                {
                    "name": c.name,
                    "rrf_score": round(c.rrf_score, 6),
                    "args_available": c.args_available,
                    "viable": c.viable,
                    "selected_by_router": c.selected_by_router,
                    "executed": c.executed,
                    "args": c.args,
                }
                for c in self.tool_calls
            ],
        }


def extract_args(tool_name: str, query: str) -> dict[str, Any] | None:
    """Heuristic argument extraction for one domain tool, or None if not enough was found."""
    if tool_name == "bioaccumulator_ratio":
        crop = _find_in_text(query, _crop_vocab())
        plant = _find_in_text(query, _plant_vocab())
        if crop and plant:
            return {"crop": crop, "candidate_plant": plant}
        return None
    if tool_name == "regional_recipe_lookup":
        crop = _find_in_text(query, _crop_vocab())
        region = _find_in_text(query, _region_vocab())
        # problem always falls back to the raw query, so this tool can run
        # (and honestly report an empty match) whenever the router selects it.
        return {"crop": crop, "region": region, "problem": query}
    return None


def _has_real_content(tool_name: str, result: dict[str, Any]) -> bool:
    """Did this (already-executed) tool call actually find something worth
    injecting, as opposed to running cleanly but turning up empty?"""
    if "error" in result:
        return False
    if tool_name == "bioaccumulator_ratio":
        return bool(result.get("ratios_ppm_per_requirement_unit"))
    if tool_name == "regional_recipe_lookup":
        return bool(result.get("region_matches") or result.get("regions_growing_crop") or result.get("matching_recipes"))
    return True


def build_multi_tool_context(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    retrieval_k: int = 5,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    retrieve_fn: Callable[[str, int], RetrievalResult] | None = None,
) -> MultiToolContext:
    """Controller-owned: always retrieve, then extract+run+gate+route domain tools.

    Selection order matters (see module docstring for why): every tool whose
    arguments are available gets *run* first (cheap, local, deterministic),
    and only tools that actually found real content are "viable" candidates.
    The router's ranking then decides which viable tool(s) to keep (top_k /
    min_score) -- it is never asked to justify picking a tool that has
    nothing to say, so "select nothing" is a real, reachable outcome instead
    of the forced top-1 pick the first #175 sweep found.

    `embed_fn` defaults to tool_router's real all-MiniLM-L6-v2 singleton;
    `retrieve_fn` defaults to retrieval.retrieve (shells out to `qmd`). Tests
    inject stubs for both so they need neither the model download/torch/CUDA
    nor a live `qmd` subprocess call.
    """
    do_retrieve = retrieve_fn or (lambda q, k: retrieve(q, k=k))
    retrieval_result = do_retrieve(query, retrieval_k)
    ctx = MultiToolContext(query=query, retrieval_context=retrieval_result.as_context())

    routing = select_relevant_tools(query, TOOL_DEFINITIONS, embed_fn=embed_fn)

    records: dict[str, ToolCallRecord] = {}
    for score in routing.scores:
        args = extract_args(score.name, query)
        args_available = args is not None
        record = ToolCallRecord(
            name=score.name,
            rrf_score=score.rrf_score,
            args_available=args_available,
            viable=False,
            selected_by_router=False,
            executed=False,
            args=args or {},
        )
        if args_available:
            fn = TOOL_REGISTRY[score.name]
            kwargs = dict(args)
            if score.name == "regional_recipe_lookup":
                kwargs["use_qmd"] = False  # retrieval is already the controller's fixed step
            result = fn(**kwargs)
            record.executed = "error" not in result
            record.result = result
            record.viable = record.executed and _has_real_content(score.name, result)
        records[score.name] = record

    viable_scores = [s for s in routing.scores if records[s.name].viable]
    viable_result = RoutingResult(query=query, scores=viable_scores)
    selected_names = set(viable_result.selected(top_k=top_k, min_score=min_score))
    for name in selected_names:
        records[name].selected_by_router = True

    ctx.tool_calls = [records[s.name] for s in routing.scores]
    return ctx
