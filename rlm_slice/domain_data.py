"""Loaders for the natural-farming domain data behind #174's tool functions (#173).

All four sources were already ported into this repo years before the
multi-tool epic (#172) started -- via the gemma4 dataset-building pipeline,
before v5clean's original train.jsonl was built -- so this module only reads
them, it does not transform or re-port anything. Per owner decision
(2026-09-15):

  * `plant_mineral_profiles.json` is the sole bioaccumulator source (verified
    byte-identical to the authoritative Excel/PDF-converted original at
    BioaccumulatorList_Xls_Transform-.../data). The flat ingredient-N/P/K
    table that used to sit next to it (`BioaccumulatorsList.json`) is
    superseded and was renamed `BioaccumulatorsList.archived.json` -- it is
    not loaded here.
  * `crop_requirements.json` + `plant_mineral_profiles.json` back the ratio
    calculator tool.
  * `local_plants_by_region.json` + `enriched_recipes.json` back the combined
    regional/recipe lookup tool.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sources" / "gemma4" / "spaces" / "data"

PLANT_MINERAL_PROFILES = DATA_DIR / "plant_mineral_profiles.json"
CROP_REQUIREMENTS = DATA_DIR / "crop_requirements.json"
LOCAL_PLANTS_BY_REGION = DATA_DIR / "local_plants_by_region.json"
ENRICHED_RECIPES = DATA_DIR / "enriched_recipes.json"


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"domain data source missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_plant_mineral_profiles() -> list[dict[str, Any]]:
    """660 plants, scientific_name/common_name -> per-part mineral ranges (ppm)."""
    return _load_json(PLANT_MINERAL_PROFILES)


@lru_cache(maxsize=1)
def load_crop_requirements() -> list[dict[str, Any]]:
    """97 crops, Crop/Scientific Name -> NPK Requirements ranges."""
    return _load_json(CROP_REQUIREMENTS)


@lru_cache(maxsize=1)
def load_local_plants_by_region() -> list[dict[str, Any]]:
    """127 regions, Location -> candidate bioaccumulator/native Plants."""
    return _load_json(LOCAL_PLANTS_BY_REGION)


@lru_cache(maxsize=1)
def load_enriched_recipes() -> list[dict[str, Any]]:
    """14 KNF/JADAM recipes with suggested_bioaccumulators and dilution ratios."""
    return _load_json(ENRICHED_RECIPES)


def clear_cache() -> None:
    """Test helper: drop cached loads so a test can swap in fixture data."""
    for fn in (
        load_plant_mineral_profiles,
        load_crop_requirements,
        load_local_plants_by_region,
        load_enriched_recipes,
    ):
        fn.cache_clear()
