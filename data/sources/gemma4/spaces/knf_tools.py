"""
KNF tool-calling schemas and Python handler functions for Gemma4.
Loaded once at startup; all handlers are pure dict lookups (no ML at runtime).
"""

import json
from difflib import SequenceMatcher
from pathlib import Path

DATA = Path(__file__).parent / "data"

_recipes: dict | None = None       # abbrev → enriched recipe dict
_minerals: list | None = None      # plant_mineral_profiles list
_local_plants: list | None = None  # local_plants_by_region list
_crops: dict | None = None         # crop name (lower) → crop dict


_LOC_KEYS = ("Location", "Country", "Region", "State", "Subregion")


def _get_location(entry: dict) -> str:
    """Return the location string from whichever key the entry uses."""
    for k in _LOC_KEYS:
        if k in entry:
            return entry[k]
    return ""


def _load():
    global _recipes, _minerals, _local_plants, _crops
    _recipes = {r["recipe_abbrev"].upper(): r
                for r in json.loads((DATA / "enriched_recipes.json").read_text())}
    _minerals = json.loads((DATA / "plant_mineral_profiles.json").read_text())
    _local_plants = json.loads((DATA / "local_plants_by_region.json").read_text())
    _crops = {c["Crop"].lower(): c
              for c in json.loads((DATA / "crop_requirements.json").read_text())}


# ── Tool schemas (OpenAI-compatible, passed as tools= to create_chat_completion) ──

KNF_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_crop_requirements",
            "description": (
                "Returns NPK (nitrogen, phosphorus, potassium) requirements and "
                "growing conditions (temperature, humidity, pH, rainfall) for a target crop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_name": {
                        "type": "string",
                        "description": "Common name of the crop, e.g. 'Tomato', 'Rice', 'Corn', 'Kale'",
                    }
                },
                "required": ["crop_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_plants",
            "description": (
                "Returns plants locally available in the user's geographic region "
                "for the current season. Use this to find what fermentation materials "
                "the user can source locally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": (
                            "User's geographic region, e.g. 'Pacific Northwest', "
                            "'Mekong Delta', 'Andean Highlands', 'Sahel Region', "
                            "'Mediterranean Coast'"
                        ),
                    },
                    "month": {
                        "type": "integer",
                        "description": "Current calendar month as integer 1-12",
                    },
                },
                "required": ["region", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_plant_npk",
            "description": (
                "Returns mineral content (N, P, K, Ca, Mg, S and trace minerals) "
                "of a specific plant part for use in KNF fermentation. "
                "Use this to evaluate a plant's nutrient contribution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "Plant name — common or scientific, e.g. 'nettle', 'Urtica dioica', 'comfrey'",
                    },
                    "plant_part": {
                        "type": "string",
                        "enum": ["leaf", "fruit", "seed", "root", "whole"],
                        "description": "Which plant part to query (default: leaf)",
                    },
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_plants_by_nutrient",
            "description": (
                "Finds plants available in the user's region that are richest in a specific nutrient. "
                "Prioritizes locally available plants. Use this to recommend what to ferment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "User's geographic region",
                    },
                    "nutrient": {
                        "type": "string",
                        "enum": ["N", "P", "K", "Ca", "Mg"],
                        "description": "Target nutrient: N=nitrogen, P=phosphorus, K=potassium, Ca=calcium, Mg=magnesium",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top results to return (default 5)",
                    },
                },
                "required": ["region", "nutrient"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_knf_recipe",
            "description": (
                "Returns the full preparation instructions, ingredients, dilution ratios, "
                "and NPK contribution for a KNF fermented input."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "enum": [
                            "FAA", "FPJ", "FFJ", "OHN", "WCA", "WSP",
                            "LABS", "BRV", "IMO", "IMO-3", "SES", "FMC", "AIA",
                        ],
                        "description": "KNF recipe abbreviation",
                    }
                },
                "required": ["recipe_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_knf_dosage",
            "description": (
                "Calculates how many liters of KNF fermented input to prepare "
                "given the crop's NPK needs and the field size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_name": {"type": "string"},
                    "field_size_sqm": {
                        "type": "number",
                        "description": "Field area in square meters",
                    },
                    "target_nutrient": {
                        "type": "string",
                        "enum": ["N", "P", "K"],
                        "description": "Which nutrient to target",
                    },
                    "recipe_name": {
                        "type": "string",
                        "description": "KNF recipe to use, e.g. 'FAA' or 'FPJ' for N, 'FFJ' for K, 'WSP' for P",
                    },
                    "applications_per_season": {
                        "type": "integer",
                        "description": "Number of applications per growing season (default 3)",
                    },
                },
                "required": ["crop_name", "field_size_sqm", "target_nutrient", "recipe_name"],
            },
        },
    },
]


# ── Handlers ──────────────────────────────────────────────────────────────────

def _fuzzy_match(name: str, candidates: list, threshold: float = 0.45) -> str | None:
    best, best_score = None, 0.0
    for c in candidates:
        s = SequenceMatcher(None, name, str(c)).ratio()
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= threshold else None


def handle_get_crop_requirements(crop_name: str) -> dict:
    if _crops is None:
        _load()
    key = crop_name.lower().strip()
    crop = _crops.get(key) or _crops.get(_fuzzy_match(key, list(_crops.keys())) or "")
    if not crop:
        return {"error": f"Crop '{crop_name}' not found. Try: Tomato, Rice, Corn, Kale, Potato…"}
    npk = crop.get("NPK Requirements", {})
    env = crop.get("Additional Requirements", {})
    return {
        "crop": crop["Crop"],
        "N_range_kg_ha": npk.get("Nitrogen", "unknown"),
        "P_range_kg_ha": npk.get("Phosphorus", "unknown"),
        "K_range_kg_ha": npk.get("Potassium", "unknown"),
        "temp_C": env.get("Temperature (°C)", "unknown"),
        "humidity_pct": env.get("Humidity (%)", "unknown"),
        "pH": env.get("pH", "unknown"),
        "rainfall_mm": env.get("Rainfall (mm)", "unknown"),
    }


def handle_get_local_plants(region: str, month: int) -> dict:
    if _local_plants is None:
        _load()
    region_l = region.lower()
    entry = next(
        (r for r in _local_plants if region_l in _get_location(r).lower()),
        None,
    )
    if not entry:
        locs_lower = [_get_location(r).lower() for r in _local_plants]
        matched = _fuzzy_match(region_l, locs_lower)
        if matched:
            entry = next((r for r in _local_plants if _get_location(r).lower() == matched), None)
    if not entry:
        available = [_get_location(r) for r in _local_plants]
        return {"error": f"Region '{region}' not found.", "available_regions": available[:20]}

    TROPICAL = {"mekong", "amazon", "sahel", "java", "bali", "philippines",
                "indonesia", "sri lanka", "ghana", "kenya", "nigeria"}
    is_tropical = any(t in entry["Location"].lower() for t in TROPICAL)
    if is_tropical:
        seasonal = "Year-round growing; wet/dry seasons may affect specific plants"
    elif 4 <= month <= 10:
        seasonal = "Active growing season"
    else:
        seasonal = "Dormant/winter season — use stored, dried, or cold-hardy materials"

    return {
        "region": _get_location(entry),
        "month": month,
        "seasonal_note": seasonal,
        "seasonality_context": entry.get("seasonality_note", ""),
        "available_plants": entry.get("Plants", []),
    }


def _get_nutrient_avg(part_data: dict, nutrient_key: str) -> float:
    m = part_data.get("minerals", {}).get(nutrient_key)
    if m and isinstance(m, dict):
        mn, mx = m.get("min", 0) or 0, m.get("max", 0) or 0
        return (mn + mx) / 2
    return 0.0


def handle_get_plant_npk(plant_name: str, plant_part: str = "leaf") -> dict:
    if _minerals is None:
        _load()
    name_l = plant_name.lower().strip()

    entry = next(
        (p for p in _minerals
         if name_l in p.get("common_name", "").lower()
         or name_l in p.get("scientific_name", "").lower()),
        None,
    )
    if not entry:
        return {"error": f"Plant '{plant_name}' not found in mineral database."}

    parts_available = list(entry.get("parts", {}).keys())
    actual_part = plant_part if plant_part in parts_available else (parts_available[0] if parts_available else "leaf")
    part_data = entry["parts"].get(actual_part, {})

    minerals_out = {}
    for mineral, vals in part_data.get("minerals", {}).items():
        if isinstance(vals, dict):
            minerals_out[mineral] = {
                "min": vals.get("min"),
                "max": vals.get("max"),
                "unit": vals.get("unit", "ppm"),
            }

    return {
        "plant": entry.get("common_name", entry.get("scientific_name", plant_name)),
        "scientific_name": entry.get("scientific_name", ""),
        "part": actual_part,
        "parts_available": parts_available,
        "source": "plant_mineral_profiles",
        "qmd_corroborated": entry.get("qmd_corroborated", False),
        "minerals": minerals_out,
    }


def handle_find_plants_by_nutrient(region: str, nutrient: str, top_n: int = 5) -> list:
    if _minerals is None:
        _load()

    local_result = handle_get_local_plants(region, 6)
    regional_names = [p.lower() for p in local_result.get("available_plants", [])]

    nutrient_map = {
        "N": "nitrogen", "P": "phosphorus", "K": "potassium",
        "Ca": "calcium", "Mg": "magnesium",
    }
    nutrient_key = nutrient_map.get(nutrient, nutrient.lower())

    scored = []
    for p in _minerals:
        common = p.get("common_name", "").lower()
        sci = p.get("scientific_name", "").lower()
        locally = any(rn in common or rn in sci or common in rn for rn in regional_names)

        best_val, best_part = 0.0, "leaf"
        for part_name, part_data in p.get("parts", {}).items():
            val = _get_nutrient_avg(part_data, nutrient_key)
            if val > best_val:
                best_val, best_part = val, part_name

        if best_val > 0:
            scored.append({
                "plant": p.get("common_name", p.get("scientific_name", "")),
                "scientific_name": p.get("scientific_name", ""),
                "best_part": best_part,
                f"{nutrient_key}_ppm_avg": round(best_val, 1),
                "locally_available": locally,
            })

    scored.sort(key=lambda x: (not x["locally_available"], -x[f"{nutrient_key}_ppm_avg"]))
    return scored[:top_n]


def handle_get_knf_recipe(recipe_name: str) -> dict:
    if _recipes is None:
        _load()
    key = recipe_name.upper().strip()
    recipe = _recipes.get(key)
    if not recipe:
        return {"error": f"Recipe '{recipe_name}' not found.", "available": list(_recipes.keys())}
    # Omit qmd_passages from tool response (keep it compact)
    return {k: v for k, v in recipe.items() if k != "qmd_passages"}


def handle_calculate_knf_dosage(
    crop_name: str,
    field_size_sqm: float,
    target_nutrient: str,
    recipe_name: str,
    applications_per_season: int = 3,
) -> dict:
    if _crops is None:
        _load()
    crop_data = handle_get_crop_requirements(crop_name)
    if "error" in crop_data:
        return crop_data

    key_map = {"N": "N_range_kg_ha", "P": "P_range_kg_ha", "K": "K_range_kg_ha"}
    npk_range = str(crop_data.get(key_map.get(target_nutrient, "N_range_kg_ha"), "60-100"))

    try:
        parts = npk_range.split("-")
        npk_kg_ha = (float(parts[0]) + float(parts[-1])) / 2
    except (ValueError, IndexError):
        npk_kg_ha = 60.0

    # Conservative: liquid KNF inputs ~0.2% nutrient concentration by weight
    concentration = 0.002
    recipe_data = handle_get_knf_recipe(recipe_name)
    dilution_info = recipe_data.get("dilution_ratio", {}) if "error" not in recipe_data else {}

    npk_kg_sqm = (npk_kg_ha / 10_000) * field_size_sqm
    liters_total = npk_kg_sqm / concentration
    liters_per_app = round(liters_total / applications_per_season, 2)

    return {
        "crop": crop_name,
        "field_size_sqm": field_size_sqm,
        "target_nutrient": target_nutrient,
        "recipe": recipe_name,
        "npk_need_midpoint_kg_ha": round(npk_kg_ha, 1),
        "liters_concentrate_per_application": liters_per_app,
        "total_liters_per_season": round(liters_per_app * applications_per_season, 2),
        "applications_per_season": applications_per_season,
        "dilution_ratios": dilution_info,
        "note": (
            "These are liters of concentrated KNF input to prepare before diluting for application. "
            "Actual nutrient uptake depends on soil conditions and application method."
        ),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_HANDLERS = {
    "get_crop_requirements":   lambda a: handle_get_crop_requirements(**a),
    "get_local_plants":        lambda a: handle_get_local_plants(**a),
    "get_plant_npk":           lambda a: handle_get_plant_npk(**a),
    "find_plants_by_nutrient": lambda a: handle_find_plants_by_nutrient(**a),
    "get_knf_recipe":          lambda a: handle_get_knf_recipe(**a),
    "calculate_knf_dosage":    lambda a: handle_calculate_knf_dosage(**a),
}


def dispatch(tool_name: str, args: dict) -> str:
    handler = _HANDLERS.get(tool_name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = handler(args)
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result, ensure_ascii=False)
