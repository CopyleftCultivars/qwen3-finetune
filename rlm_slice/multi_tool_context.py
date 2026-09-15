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
     ported hybrid ranking;
  4. executes and injects only tools that are BOTH selected by the router
     AND have the arguments to actually run -- a tool ranked relevant but
     missing required arguments is recorded, not silently skipped, so the
     eventual eval can distinguish "router said no" from "router said yes,
     but there was nothing to call it with".

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
from rlm_slice.tool_router import select_relevant_tools


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
    selected_by_router: bool
    args_available: bool
    executed: bool
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class MultiToolContext:
    query: str
    retrieval_context: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    def as_context(self) -> str:
        """Retrieval context (Arm R, unchanged) + any executed domain-tool blocks."""
        blocks = [self.retrieval_context]
        for call in self.tool_calls:
            if call.executed:
                blocks.append(f"[{call.name}] {call.result}")
        return "\n\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tool_calls": [
                {
                    "name": c.name,
                    "rrf_score": round(c.rrf_score, 6),
                    "selected_by_router": c.selected_by_router,
                    "args_available": c.args_available,
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


def build_multi_tool_context(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    retrieval_k: int = 5,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    retrieve_fn: Callable[[str, int], RetrievalResult] | None = None,
) -> MultiToolContext:
    """Controller-owned: always retrieve, then route+extract+execute domain tools.

    `embed_fn` defaults to tool_router's real all-MiniLM-L6-v2 singleton;
    `retrieve_fn` defaults to retrieval.retrieve (shells out to `qmd`). Tests
    inject stubs for both so they need neither the model download/torch/CUDA
    nor a live `qmd` subprocess call.
    """
    do_retrieve = retrieve_fn or (lambda q, k: retrieve(q, k=k))
    retrieval_result = do_retrieve(query, retrieval_k)
    ctx = MultiToolContext(query=query, retrieval_context=retrieval_result.as_context())

    routing = select_relevant_tools(query, TOOL_DEFINITIONS, embed_fn=embed_fn)
    selected_names = set(routing.selected(top_k=top_k, min_score=min_score))

    for score in routing.scores:
        selected = score.name in selected_names
        args = extract_args(score.name, query) if selected else None
        args_available = bool(args) and all(v for v in args.values() if not isinstance(v, str) or v)
        # regional_recipe_lookup's `problem` is always the raw query, so it is
        # "available" whenever selected; bioaccumulator_ratio needs both slots filled.
        if score.name == "regional_recipe_lookup":
            args_available = args is not None
        record = ToolCallRecord(
            name=score.name,
            rrf_score=score.rrf_score,
            selected_by_router=selected,
            args_available=args_available,
            executed=False,
            args=args or {},
        )
        if selected and args_available:
            fn = TOOL_REGISTRY[score.name]
            kwargs = dict(args)
            if score.name == "regional_recipe_lookup":
                kwargs["use_qmd"] = False  # retrieval is already the controller's fixed step
            result = fn(**kwargs)
            record.executed = "error" not in result
            record.result = result
        ctx.tool_calls.append(record)

    return ctx
