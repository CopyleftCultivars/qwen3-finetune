"""Build a small dual-relevant eval slice for #175's routing question.

val_v3's 113-item sweep found 0/113 items where both domain tools
(bioaccumulator_ratio, regional_recipe_lookup) were simultaneously viable --
so Arm M (routed top-1) and Arm N (all viable) could never diverge, and the
"does routing beat no-routing" question went untested. This script generates
MCQ items specifically constructed so both tools fire on the same item.

Ground truth comes from regional_recipe_lookup's own regions_growing_crop
field (called directly, for perfect self-consistency): each item asks which
of 4 named regions a crop is documented as growing in, with the correct
region and 3 genuine non-matching distractors, position randomized.

Earlier attempt and why it was abandoned: the first version asked whether a
plant's tissue phosphorus ratio was higher/lower than the crop's requirement
(bioaccumulator_ratio's own output). Sampling 1,319 crop x plant pairs across
the *entire* dataset (not just a hand-picked few) showed the ratio classifies
as "higher" 100% of the time -- tissue ppm concentrations and this crop
dataset's NPK requirement numbers are different unit systems entirely (as
bioaccumulator_ratio's own unit_caveat already says), so any "ratio vs a
fixed threshold" or "phosphorus vs potassium" framing is a systematic
artifact of units/biology, not a per-item fact worth testing. A named-region
question doesn't have this problem: which specific regions grow which
specific crop is real, per-item-varying structured data.

A plant name is still named in each question stem (needed for
bioaccumulator_ratio's extraction to find both a crop and a plant), and
bioaccumulator_ratio's own viability check (does it produce a non-empty
ratio at all, now that the range-parsing bug is fixed) is separately true for
essentially any valid crop+plant pair -- so both tools are viable, without
the MCQ's graded answer needing to depend on bioaccumulator_ratio's specific
numbers.

Output format matches the MCQ shape score_mcq_ll.normalize_records expects
(question/A/B/C/D/answer/question_id), so eval_slice.py can run it unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlm_slice.domain_data import load_crop_requirements, load_local_plants_by_region  # noqa: E402
from rlm_slice.domain_tools import regional_recipe_lookup  # noqa: E402
from rlm_slice.multi_tool_context import build_multi_tool_context  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 175

# A plant name mentioned in the question stem so bioaccumulator_ratio's
# extraction finds a crop+plant pair; rotated across items for variety. Any
# of these paired with any real crop produces a non-empty ratio post-fix, so
# the specific choice doesn't affect the graded answer.
CANDIDATE_PLANTS = ["Comfrey", "Stinging Nettle", "Dandelion", "Red Clover", "Yarrow", "Borage"]

MIN_REGIONS, MAX_REGIONS = 1, 25

QUESTION_TEMPLATE = (
    "A grower is deciding whether to plant {crop}, and is separately considering "
    "{plant} as a companion bioaccumulator to help meet its nutrient needs. "
    "Based on regional growing data, which of the following regions is {crop} "
    "documented as growing in?"
)


def build_item(crop: str, matching_regions: list[str], all_regions: list[str], rng: random.Random, idx: int) -> dict:
    correct_region = rng.choice(matching_regions)
    non_matching = [r for r in all_regions if r not in matching_regions]
    distractors = rng.sample(non_matching, 3)

    options = [correct_region] + distractors
    rng.shuffle(options)
    letters = ["A", "B", "C", "D"]
    answer = letters[options.index(correct_region)]

    plant = CANDIDATE_PLANTS[idx % len(CANDIDATE_PLANTS)]
    question = QUESTION_TEMPLATE.format(crop=crop, plant=plant)

    return {
        "question_id": f"dual175_{idx:03d}",
        "question": question,
        "A": options[0], "B": options[1], "C": options[2], "D": options[3],
        "answer": answer,
        "topic": "regional_recipe_lookup_dual_relevant",
        "source_document": "synthetic:dual_relevant_eval_175",
        "_crop": crop, "_plant": plant, "_correct_region": correct_region,
    }


def verify_dual_viable(item: dict) -> tuple[bool, bool]:
    """Return (bioaccumulator_ratio viable, regional_recipe_lookup viable)."""
    ctx = build_multi_tool_context(item["question"], top_k=None)
    viable = {c.name: c.viable for c in ctx.tool_calls}
    return viable.get("bioaccumulator_ratio", False), viable.get("regional_recipe_lookup", False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/dual_relevant_eval_175.jsonl")
    parser.add_argument("--n-items", type=int, default=24)
    args = parser.parse_args()

    rng = random.Random(SEED)
    all_regions = [row["Location"] for row in load_local_plants_by_region()]
    crops = [r["Crop"] for r in load_crop_requirements()]

    eligible = []
    for crop in crops:
        result = regional_recipe_lookup(crop=crop, use_qmd=False)
        regions = sorted(set(result.get("regions_growing_crop", [])))
        if MIN_REGIONS <= len(regions) <= MAX_REGIONS:
            eligible.append((crop, regions))

    print(f"{len(eligible)} crops eligible (1-25 matching regions)")
    rng.shuffle(eligible)
    chosen = eligible[: args.n_items]

    items = [build_item(crop, regions, all_regions, rng, idx) for idx, (crop, regions) in enumerate(chosen)]

    print(f"generated {len(items)} items; verifying dual-viability on each (real embedder + real retrieval)...")
    n_both_viable = 0
    for item in items:
        ratio_viable, recipe_viable = verify_dual_viable(item)
        item["_ratio_tool_viable"] = ratio_viable
        item["_recipe_tool_viable"] = recipe_viable
        if ratio_viable and recipe_viable:
            n_both_viable += 1
        else:
            print(f"  WARNING {item['question_id']}: ratio_viable={ratio_viable} recipe_viable={recipe_viable}")

    print(f"{n_both_viable}/{len(items)} items have BOTH tools viable")

    answer_counts: dict[str, int] = {}
    for item in items:
        answer_counts[item["answer"]] = answer_counts.get(item["answer"], 0) + 1
    print(f"answer distribution: {answer_counts}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
