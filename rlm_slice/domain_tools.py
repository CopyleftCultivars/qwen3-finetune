"""Domain tool functions for the multi-tool epic (#172), built per #174.

Two tools, not three (#172 originally scoped a third "technique/recipe lookup"
tool separately from "regional lookup"; owner decision 2026-09-15 folded them
into one, since string-match-style lookup was already found to outperform
calculation for that combined case):

  * `bioaccumulator_ratio` -- pure calculation, no lookup ambiguity.
  * `regional_recipe_lookup` -- combined region + recipe/technique lookup,
    enriched with a supplementary QMD collection lookup where possible.

Each tool is exposed two ways:
  * as a plain Python callable, for GPU-side use inside rlm_arm.py's
    `custom_tools` REPL-exposure pattern (see `agronomy_tools()`);
  * as a Cactus-shaped `ToolFunction` dict (`{"name", "description",
    "parameters"}` -- see cactus-sdk/python/cactus/server.py's `ToolFunction`
    and cactus-engine/src/rag.cpp's `tool_to_text()`, which embeds exactly
    `name + " " + description` for `select_relevant_tools` to score against)
    via `TOOL_DEFINITIONS`, so the same definitions can be registered
    on-device later without redefinition (#177).
"""

from __future__ import annotations

import re
from typing import Any

from rlm_slice.domain_data import (
    load_crop_requirements,
    load_enriched_recipes,
    load_local_plants_by_region,
    load_plant_mineral_profiles,
)
from rlm_slice.retrieval import retrieve

_RANGE_RE = re.compile(r"-?\d+(?:\.\d+)?")

# plant_mineral_profiles.json tracks trace minerals (P, K, Ca, Fe, ...) per
# USDA composition data; it does not track nitrogen (N is protein-bound, not
# a "mineral" in that dataset's sense). So only P and K are ever comparable
# against crop_requirements.json's NPK Requirements -- nitrogen is always
# reported as not-comparable, not silently dropped.
_NUTRIENT_MAP = {"Phosphorus": "phosphorus", "Potassium": "potassium"}

_PROBLEM_NUTRIENT_HINTS = {
    "nitrogen": "N", "n deficiency": "N",
    "phosphorus": "P", "p deficiency": "P",
    "potassium": "K", "k deficiency": "K",
}


def _parse_range(text: str) -> tuple[float, float] | None:
    nums = [float(n) for n in _RANGE_RE.findall(str(text))]
    if not nums:
        return None
    return (min(nums), max(nums))


def _mid(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2.0


def _find_crop(name: str) -> dict[str, Any] | None:
    name_l = name.strip().lower()
    rows = load_crop_requirements()
    for row in rows:
        if str(row.get("Crop", "")).lower() == name_l or str(row.get("Scientific Name", "")).lower() == name_l:
            return row
    for row in rows:
        if name_l in str(row.get("Crop", "")).lower():
            return row
    return None


def _find_plant(name: str) -> dict[str, Any] | None:
    name_l = name.strip().lower()
    rows = load_plant_mineral_profiles()
    for row in rows:
        if str(row.get("common_name", "")).lower() == name_l or str(row.get("scientific_name", "")).lower() == name_l:
            return row
    for row in rows:
        if name_l in str(row.get("common_name", "")).lower() or name_l in str(row.get("scientific_name", "")).lower():
            return row
    return None


def _best_mineral_mid(plant_row: dict[str, Any], mineral_key: str) -> float | None:
    """Richest-part mid-range value for one mineral, ppm (max across plant parts)."""
    best: float | None = None
    for part_data in plant_row.get("parts", {}).values():
        mineral = part_data.get("minerals", {}).get(mineral_key)
        if not mineral:
            continue
        try:
            mid = (float(mineral["min"]) + float(mineral["max"])) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or mid > best:
            best = mid
    return best


def bioaccumulator_ratio(crop: str, candidate_plant: str) -> dict[str, Any]:
    """Compare a crop's NPK requirement against a candidate bioaccumulator plant's
    tissue mineral content (plant_mineral_profiles.json), one ratio per comparable
    nutrient (plant tissue mid-range ppm / crop mid-range requirement). Nitrogen is
    never comparable -- the mineral-profile source does not track it. Ratios are a
    relative signal for judging whether a plant is nutrient-dense enough, relative
    to the crop's stated need, to be worth cutting as green manure or steeping into
    a liquid feed -- not a calibrated fertilizer dosing formula (the two sources use
    different unit systems: tissue ppm vs. an agronomic NPK requirement index).
    """
    crop_row = _find_crop(crop)
    if crop_row is None:
        return {"error": f"no crop matching {crop!r} in crop_requirements.json"}
    plant_row = _find_plant(candidate_plant)
    if plant_row is None:
        return {"error": f"no plant matching {candidate_plant!r} in plant_mineral_profiles.json"}

    npk = crop_row.get("NPK Requirements", {})
    ratios: dict[str, float] = {}
    not_comparable: list[str] = []
    for req_key, mineral_key in _NUTRIENT_MAP.items():
        crop_bounds = _parse_range(npk.get(req_key, ""))
        plant_mid = _best_mineral_mid(plant_row, mineral_key)
        if crop_bounds is None or plant_mid is None:
            not_comparable.append(mineral_key)
            continue
        crop_mid = _mid(crop_bounds)
        ratios[mineral_key] = round(plant_mid / crop_mid, 3) if crop_mid else None
    not_comparable.append("nitrogen")  # never tracked by the mineral-profile source

    return {
        "crop": crop_row.get("Crop"),
        "candidate_plant": plant_row.get("common_name") or plant_row.get("scientific_name"),
        "ratios_ppm_per_requirement_unit": ratios,
        "not_comparable": not_comparable,
        "unit_caveat": (
            "ratios compare tissue ppm (plant_mineral_profiles.json) against an "
            "agronomic NPK requirement index (crop_requirements.json) -- a relative "
            "signal, not a unit-equivalent dosing calculation"
        ),
    }


def _token_overlap_score(query_terms: set[str], *texts: str) -> int:
    blob = " ".join(texts).lower()
    return sum(1 for term in query_terms if term and term in blob)


def regional_recipe_lookup(
    crop: str | None = None,
    region: str | None = None,
    problem: str | None = None,
    use_qmd: bool = True,
    qmd_k: int = 3,
) -> dict[str, Any]:
    """Look up candidate bioaccumulator plants native to a region
    (local_plants_by_region.json) and/or matching Korean Natural Farming (KNF) /
    JADAM technique or recipe for a crop or stated problem (enriched_recipes.json),
    combining structured lookups with a supplementary QMD collection search so the
    result surfaces both known structured facts and any related indexed reference
    material in one call. At least one of crop, region, or problem must be given.
    """
    if not (crop or region or problem):
        return {"error": "at least one of crop, region, or problem is required"}

    region_l = region.strip().lower() if region else None
    crop_l = crop.strip().lower() if crop else None

    region_matches = []
    regions_growing_crop = []
    for row in load_local_plants_by_region():
        location = str(row.get("Location", ""))
        plants = row.get("Plants", [])
        if region_l and region_l in location.lower():
            region_matches.append(row)
        if crop_l and any(crop_l in str(p).lower() for p in plants):
            regions_growing_crop.append(location)

    query_terms = {t.lower() for t in (crop, problem) if t}
    hinted_nutrient = None
    if problem:
        for hint, code in _PROBLEM_NUTRIENT_HINTS.items():
            if hint in problem.lower():
                hinted_nutrient = code
                break

    scored_recipes = []
    for recipe in load_enriched_recipes():
        score = _token_overlap_score(
            query_terms,
            str(recipe.get("recipe_name", "")),
            " ".join(recipe.get("suggested_bioaccumulators", [])),
            " ".join(recipe.get("ingredients", [])),
        )
        if hinted_nutrient and recipe.get("primary_nutrient") == hinted_nutrient:
            score += 2
        if score > 0:
            scored_recipes.append((score, recipe))
    scored_recipes.sort(key=lambda pair: pair[0], reverse=True)
    matching_recipes = [r for _score, r in scored_recipes[:5]]

    qmd_context = "qmd lookup skipped (use_qmd=False)"
    if use_qmd:
        qmd_query = " ".join(t for t in (crop, region, problem) if t).strip()
        if qmd_query:
            result = retrieve(qmd_query, k=qmd_k)
            qmd_context = result.as_context()
        else:
            qmd_context = "qmd lookup skipped (empty query)"

    return {
        "region_matches": region_matches,
        "regions_growing_crop": regions_growing_crop,
        "matching_recipes": matching_recipes,
        "qmd_context": qmd_context,
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "bioaccumulator_ratio",
        "description": bioaccumulator_ratio.__doc__.strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "Crop name, e.g. 'Corn'"},
                "candidate_plant": {
                    "type": "string",
                    "description": "Candidate bioaccumulator plant common or scientific name, e.g. 'Comfrey'",
                },
            },
            "required": ["crop", "candidate_plant"],
        },
    },
    {
        "name": "regional_recipe_lookup",
        "description": regional_recipe_lookup.__doc__.strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {"type": "string", "description": "Crop name, optional"},
                "region": {"type": "string", "description": "Region or location name, optional"},
                "problem": {"type": "string", "description": "Stated problem, e.g. 'nitrogen deficiency', optional"},
            },
            "required": [],
        },
    },
]

TOOL_REGISTRY: dict[str, Any] = {
    "bioaccumulator_ratio": bioaccumulator_ratio,
    "regional_recipe_lookup": regional_recipe_lookup,
}


def domain_tools() -> dict[str, Any]:
    """rlm_arm.py-style custom_tools dict, matching agronomy_tools()'s shape."""
    return dict(TOOL_REGISTRY)
