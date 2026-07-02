#!/usr/bin/env python3
"""
Offline enrichment — run once before deploying to HF Spaces.

Uses `qmd query` to extract KNF-relevant passages from the indexed collections
and produces the 5 enriched tool-call JSON files in spaces/data/.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "datasets_for_rag"
OUT = ROOT / "spaces" / "data"
OUT.mkdir(parents=True, exist_ok=True)


# ── QMD helpers ───────────────────────────────────────────────────────────────

def qmd_query(q: str, top_n: int = 2) -> list:
    try:
        r = subprocess.run(["qmd", "query", q], capture_output=True, text=True, timeout=90)
        return _parse_qmd(r.stdout, top_n)
    except Exception as e:
        print(f"  [qmd warning] {e}")
        return []


def _parse_qmd(text: str, top_n: int) -> list:
    results = []
    for block in re.split(r"\n(?=qmd://)", text):
        if not block.startswith("qmd://"):
            continue
        src = re.match(r"(qmd://\S+)", block)
        score = re.search(r"Score:\s+(\d+)%", block)
        content = re.search(r"@@[^\n]*\n(.*)", block, re.DOTALL)
        if src and content:
            results.append({
                "source": src.group(1),
                "score": int(score.group(1)) if score else 0,
                "text": content.group(1).strip()[:500],
            })
        if len(results) >= top_n:
            break
    return results


def load_json_lenient(path: Path) -> list:
    """Load JSON files that have common formatting issues (missing commas) in this dataset."""
    text = path.read_text()
    # Fix 1: ] followed directly by "Recipe": without closing } and opening {
    text = re.sub(r'(\])\s*\n(\s*)"Recipe":', r'\1\n\2},\n\2{"Recipe":', text)
    # Fix 2: top-level array elements missing comma — targets only lines where } at 2-space
    # indent is followed immediately by { at 2-space indent (no }/{ inside nested structures)
    text = re.sub(r'\n( {2})\}\n( {2})\{', r'\n\1},\n\2{', text)
    return json.loads(text)


# ── NPK attribution table (sourced from KNF literature) ──────────────────────

NPK_TABLE = {
    "Fish Amino Acid (FAA)": {
        "abbrev": "FAA",
        "primary_nutrient": "N",
        "npk_provided": {"N": "high", "P": "trace", "K": "low"},
        "suggested_bioaccumulators": ["Fish (blue-back species)", "Comfrey leaf", "Nettle leaf"],
        "dilution_ratio": {"foliar": "1:500-1:1000", "soil_drench": "1:500", "IMO_activator": "1:1000"},
    },
    "Fermented Plant Juice (FPJ)": {
        "abbrev": "FPJ",
        "primary_nutrient": "N",
        "npk_provided": {"N": "high", "P": "low", "K": "moderate"},
        "suggested_bioaccumulators": [
            "Urtica dioica (Nettle)", "Symphytum officinale (Comfrey)",
            "Trifolium repens (Clover)", "Taraxacum officinale (Dandelion)",
        ],
        "dilution_ratio": {"foliar": "1:500", "soil_drench": "1:200"},
    },
    "Fermented Fruit Juice (FFJ)": {
        "abbrev": "FFJ",
        "primary_nutrient": "K",
        "npk_provided": {"N": "trace", "P": "moderate", "K": "high"},
        "suggested_bioaccumulators": [
            "Musa (Banana)", "Solanum lycopersicum (Tomato)",
            "Prunus persica (Peach)", "Mangifera indica (Mango)",
        ],
        "dilution_ratio": {"foliar": "1:500", "soil_drench": "1:500"},
    },
    "Oriental Herbal Nutrient (OHN)": {
        "abbrev": "OHN",
        "primary_nutrient": "micronutrients",
        "npk_provided": {"N": "trace", "P": "trace", "K": "trace"},
        "suggested_bioaccumulators": [
            "Allium sativum (Garlic)", "Zingiber officinale (Ginger)",
            "Cinnamomum (Cinnamon)", "Angelica archangelica",
        ],
        "dilution_ratio": {"foliar": "1:1000", "soil_drench": "1:1000", "pest_control": "1:500"},
    },
    "Water-Soluble Calcium (WCA)": {
        "abbrev": "WCA",
        "primary_nutrient": "Ca",
        "npk_provided": {"N": "none", "P": "none", "K": "none", "Ca": "high"},
        "suggested_bioaccumulators": ["Eggshell", "Oyster shell", "Limestone"],
        "dilution_ratio": {"foliar": "1:1000", "soil_drench": "1:1000"},
    },
    "Water-Soluble Phosphorus (WSP)": {
        "abbrev": "WSP",
        "primary_nutrient": "P",
        "npk_provided": {"N": "trace", "P": "high", "K": "low"},
        "suggested_bioaccumulators": ["Fish bones", "Animal bones"],
        "dilution_ratio": {"foliar": "1:1000", "soil_drench": "1:1000"},
    },
    "Lactic Acid Bacteria Serum (LABS)": {
        "abbrev": "LABS",
        "primary_nutrient": "soil_biology",
        "npk_provided": {"N": "trace", "P": "trace", "K": "trace"},
        "suggested_bioaccumulators": ["Rice wash water", "Raw milk"],
        "dilution_ratio": {"foliar": "1:1000", "soil_drench": "1:1000", "compost_activator": "1:500"},
    },
    "Brown Rice Vinegar (BRV)": {
        "abbrev": "BRV",
        "primary_nutrient": "pH_buffer",
        "npk_provided": {"N": "trace", "P": "trace", "K": "trace"},
        "suggested_bioaccumulators": ["Brown rice"],
        "dilution_ratio": {"foliar": "1:1000", "soil_drench": "1:1000"},
    },
    "Collection of IMOs from Native Soil": {
        "abbrev": "IMO",
        "primary_nutrient": "soil_biology",
        "npk_provided": {"N": "low", "P": "low", "K": "low"},
        "suggested_bioaccumulators": ["Forest floor material", "Bamboo grove soil", "Oak grove soil"],
        "dilution_ratio": {"soil_application": "broadcast directly after IMO-4 stage"},
    },
    "Preparation of IMO-3": {
        "abbrev": "IMO-3",
        "primary_nutrient": "soil_biology",
        "npk_provided": {"N": "moderate", "P": "moderate", "K": "moderate"},
        "suggested_bioaccumulators": ["Rice bran", "Rice flour", "Paddy straw"],
        "dilution_ratio": {"soil_drench": "1:1000", "compost_activator": "1:500"},
    },
    "Seed/Seedling Treatment (SES)": {
        "abbrev": "SES",
        "primary_nutrient": "mixed",
        "npk_provided": {"N": "moderate", "P": "moderate", "K": "moderate"},
        "suggested_bioaccumulators": ["FPJ", "FFJ", "OHN", "LABS", "BRV"],
        "dilution_ratio": {"seed_soak": "1:1000"},
    },
    "Fermented Mixed Compost (FMC)": {
        "abbrev": "FMC",
        "primary_nutrient": "mixed",
        "npk_provided": {"N": "high", "P": "moderate", "K": "moderate"},
        "suggested_bioaccumulators": ["Kitchen scraps", "Garden waste", "Animal manure"],
        "dilution_ratio": {"soil_amendment": "top dress or incorporate into soil"},
    },
    "Aromatic Insect Attractants (AIA)": {
        "abbrev": "AIA",
        "primary_nutrient": "none",
        "npk_provided": {"N": "none", "P": "none", "K": "none"},
        "suggested_bioaccumulators": ["Aromatic plants", "Citrus peel"],
        "dilution_ratio": {"trap": "use undiluted in field traps"},
    },
    "Fermented Wildcrafted Plant Fertilizer": {
        "abbrev": "Wildcrafted",
        "primary_nutrient": "variable",
        "npk_provided": {"N": "variable", "P": "variable", "K": "variable"},
        "suggested_bioaccumulators": ["Varies — select based on target nutrient"],
        "dilution_ratio": {"foliar": "1:10", "soil_drench": "1:10"},
    },
}

RECIPE_QUERIES = {
    "FAA":        "FAA fish amino acid nitrogen content application rate",
    "FPJ":        "FPJ fermented plant juice nitrogen growth young shoots",
    "FFJ":        "FFJ fermented fruit juice phosphorus potassium fruiting stage",
    "OHN":        "OHN oriental herbal nutrient garlic ginger pest disease control",
    "WCA":        "WCA water soluble calcium eggshell vinegar cell wall structure",
    "WSP":        "WSP water soluble phosphorus bones root development",
    "LABS":       "LAB lactic acid bacteria serum soil biology probiotic rice wash",
    "BRV":        "BRV brown rice vinegar microbial activity pH buffer",
    "IMO":        "IMO indigenous microorganism collection forest soil aerobic",
    "IMO-3":      "IMO-3 propagation rice bran soil fertility beneficial microbes",
    "Wildcrafted": "wildcrafted plant fermentation nutrient profile natural farming",
}


def enrich_recipes():
    print("\n── enriched_recipes.json ──")
    src = DATASETS / "Natural_Farming_Recipes_Datachunks" / "NF_Recipes.json"
    recipes_raw = load_json_lenient(src)
    enriched = []
    for r in recipes_raw:
        name = r["Recipe"]
        npk = NPK_TABLE.get(name, {
            "abbrev": name.split("(")[-1].rstrip(")") if "(" in name else name[:8],
            "primary_nutrient": "unknown",
            "npk_provided": {"N": "unknown", "P": "unknown", "K": "unknown"},
            "suggested_bioaccumulators": [],
            "dilution_ratio": {"general": "1:500-1:1000"},
        })
        abbrev = npk["abbrev"]
        query = RECIPE_QUERIES.get(abbrev, f"{abbrev} natural farming preparation KNF")
        print(f"  qmd: {abbrev}...", end=" ", flush=True)
        passages = qmd_query(query, top_n=2)
        print(f"{len(passages)} passages")
        enriched.append({
            "recipe_abbrev": abbrev,
            "recipe_name": name,
            "primary_nutrient": npk["primary_nutrient"],
            "npk_provided": npk["npk_provided"],
            "suggested_bioaccumulators": npk["suggested_bioaccumulators"],
            "dilution_ratio": npk["dilution_ratio"],
            "ingredients": r.get("Ingredients", []),
            "steps": r.get("Steps", []),
            "uses": r.get("Uses", []),
            "qmd_passages": passages,
        })
    out = OUT / "enriched_recipes.json"
    out.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    print(f"  → {out} ({len(enriched)} recipes)")


def enrich_plant_minerals():
    print("\n── plant_mineral_profiles.json ──")
    src = DATASETS / "crop_mineral_data" / "plant_mineral_profiles.json"
    plants = json.loads(src.read_text())
    # Add qmd_corroborated flag; spot-check a few key KNF bioaccumulators
    spot_checks = {
        "nettle": "nettle Urtica dioica nitrogen bioaccumulator mineral content",
        "comfrey": "comfrey Symphytum officinale potassium mineral content",
        "dandelion": "dandelion Taraxacum mineral content natural farming",
    }
    corroborated = set()
    for key, query in spot_checks.items():
        print(f"  qmd: {key}...", end=" ", flush=True)
        passages = qmd_query(query, top_n=1)
        if passages:
            corroborated.add(key)
        print(f"{'ok' if passages else 'no result'}")

    for p in plants:
        common = p.get("common_name", "").lower()
        p["qmd_corroborated"] = any(k in common for k in corroborated)

    out = OUT / "plant_mineral_profiles.json"
    out.write_text(json.dumps(plants, indent=2, ensure_ascii=False))
    print(f"  → {out} ({len(plants)} plants)")


def enrich_local_plants():
    print("\n── local_plants_by_region.json ──")
    src = DATASETS / "Locally_Growing_Plants_by_Region.json"
    all_items = load_json_lenient(src)
    # Normalize: file uses Location/Country/Region/State/Subregion as the location key
    LOCATION_KEYS = ("Location", "Country", "Region", "State", "Subregion")
    regions = []
    for item in all_items:
        if not isinstance(item, dict):
            continue
        loc_key = next((k for k in LOCATION_KEYS if k in item), None)
        if loc_key:
            item["Location"] = item[loc_key]  # normalise to "Location"
            regions.append(item)
    print(f"  normalised {len(regions)} region entries (from Location/Country/Region/State/Subregion)")
    # Add a brief seasonality_note per region based on climate zone heuristic
    TROPICAL = {"mekong", "amazon", "sahel", "java", "bali", "philippines", "indonesia",
                "sri lanka", "ghana", "kenya", "nigeria", "equator"}
    MEDITERRANEAN = {"mediterranean", "california", "chile", "cape", "south africa", "australia"}
    for region in regions:
        loc = region["Location"].lower()
        if any(t in loc for t in TROPICAL):
            region["seasonality_note"] = "Tropical — year-round growing; wet/dry seasons affect availability"
        elif any(m in loc for m in MEDITERRANEAN):
            region["seasonality_note"] = "Mediterranean — growing season Mar-Oct; dry summer stress"
        else:
            region["seasonality_note"] = "Temperate — active growing season Apr-Oct; dormant Nov-Mar"
    out = OUT / "local_plants_by_region.json"
    out.write_text(json.dumps(regions, indent=2, ensure_ascii=False))
    print(f"  → {out} ({len(regions)} regions)")


def copy_crop_requirements():
    print("\n── crop_requirements.json ──")
    src = DATASETS / "SemiSynthetic_Crop_Requirements" / "Crop_Requirements.json"
    data = json.loads(src.read_text())
    out = OUT / "crop_requirements.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  → {out} ({len(data)} crops)")


def copy_bioaccumulators():
    print("\n── bioaccumulators.json ──")
    src = DATASETS / "BioaccumulatorsList.json"
    data = load_json_lenient(src)
    out = OUT / "bioaccumulators.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  → {out} ({len(data)} entries)")


if __name__ == "__main__":
    print(f"Output directory: {OUT}")
    enrich_recipes()
    enrich_plant_minerals()
    enrich_local_plants()
    copy_crop_requirements()
    copy_bioaccumulators()
    print("\nDone. Review spaces/data/ before deploying.")
