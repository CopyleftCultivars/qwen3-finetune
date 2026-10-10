#!/usr/bin/env python3
"""Screen the qTMD markdown corpus and create disjoint free-form eval slices."""

from __future__ import annotations

import hashlib
import os
import json
import random
import re
from collections import Counter
from pathlib import Path


CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "data/sources"))
STATED_CORPUS_COUNT = 1855
OUTPUT_DIR = Path(__file__).resolve().parent
SCREEN_PATH = OUTPUT_DIR / "corpus_screen.json"
REPORT_PATH = OUTPUT_DIR / "corpus_coverage_report.md"
SLICE_DIR = OUTPUT_DIR / "corpus_slices"
SEED = 20260715

ITEM_TYPES = [
    "multi_hop_factual",
    "diagnostic_reasoning",
    "free_form_explanation",
    "constraint_satisfaction",
    "quantitative",
]

TOPIC_KEYWORDS = {
    "soil_fertility": [
        "soil",
        "compost",
        "fertility",
        "fertilizer",
        "nutrient",
        "nitrogen",
        "phosphorus",
        "potassium",
        "microbe",
        "mycorrhiza",
        "humus",
        "manure",
        "biochar",
    ],
    "water_irrigation": [
        "irrigation",
        "water",
        "drip",
        "mulch",
        "drought",
        "moisture",
        "runoff",
        "infiltration",
        "rainwater",
        "transpiration",
    ],
    "pest_disease": [
        "pest",
        "disease",
        "fungus",
        "fungal",
        "insect",
        "mite",
        "pathogen",
        "blight",
        "mildew",
        "nematode",
        "aphid",
        "weed",
    ],
    "crop_horticulture": [
        "crop",
        "plant",
        "seed",
        "seedling",
        "vegetative",
        "flower",
        "harvest",
        "prune",
        "greenhouse",
        "horticulture",
        "garden",
        "orchard",
        "nursery",
        "cannabis",
        "hemp",
    ],
    "agroecology_systems": [
        "agroecology",
        "agroforestry",
        "permaculture",
        "regenerative",
        "organic",
        "sustainable",
        "farming",
        "farm",
        "pasture",
        "livestock",
        "cover crop",
        "rotation",
        "polyculture",
        "intercrop",
    ],
}

NEGATIVE_KEYWORDS = [
    "cannabinoid receptor",
    "cannabis dependence",
    "psychiatry",
    "neural",
    "clinical trial",
    "patients",
    "pharmacology",
    "vacuum physics",
    "falling liquid films",
    "quantum",
    "mathematical sciences",
    "electrodynamics",
    "conference proceedings",
    "patent",
]

REFERENCE_ONLY_PATTERNS = [
    "bibliography",
    "references",
    "table of contents",
    "index",
    "acknowledgements",
]


def clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]*\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*_`|>\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rel_doc_path(path: Path) -> str:
    return "../qtmd_workspace/markdown_corpus/" + str(path.relative_to(CORPUS_ROOT))


def topic_scores(haystack: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += len(re.findall(rf"\b{re.escape(keyword)}\b", haystack))
        scores[topic] = score
    return scores


def topic_tag(scores: dict[str, int]) -> str:
    topic, score = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
    return topic if score > 0 else "off_topic"


def classify_doc(directory: Path, seen_fingerprints: set[str]) -> dict[str, object]:
    markdown_files = sorted(directory.glob("*.md"))
    if not markdown_files:
        return {
            "path": rel_doc_path(directory),
            "eligible": False,
            "topic_tag": "missing_markdown",
            "reason": "Top-level corpus directory has no markdown file to screen.",
        }

    path = markdown_files[0]
    raw_text = path.read_text(errors="ignore")
    cleaned = clean_markdown(raw_text)
    sample = (directory.name + " " + cleaned[:6000]).lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", cleaned)

    if len(words) < 250:
        return {
            "path": rel_doc_path(path),
            "eligible": False,
            "topic_tag": "too_thin",
            "reason": f"Only {len(words)} prose-like words after markdown cleanup.",
        }

    fingerprint = hashlib.sha256(
        re.sub(r"\W+", " ", cleaned[:3000].lower()).encode("utf-8")
    ).hexdigest()
    if fingerprint in seen_fingerprints:
        return {
            "path": rel_doc_path(path),
            "eligible": False,
            "topic_tag": "near_duplicate",
            "reason": "Opening prose fingerprint duplicates an already-screened document.",
        }
    seen_fingerprints.add(fingerprint)

    scores = topic_scores(sample)
    tag = topic_tag(scores)
    positive_score = sum(scores.values())
    negative_hits = [kw for kw in NEGATIVE_KEYWORDS if kw in sample]
    reference_only_hits = [kw for kw in REFERENCE_ONLY_PATTERNS if kw in sample[:1500]]

    if reference_only_hits and positive_score < 12:
        return {
            "path": rel_doc_path(path),
            "eligible": False,
            "topic_tag": "reference_only",
            "reason": "Appears to be reference/front-matter material with little practical prose.",
        }

    if negative_hits and positive_score < 10:
        return {
            "path": rel_doc_path(path),
            "eligible": False,
            "topic_tag": "off_topic",
            "reason": "Topic terms indicate non-practical or non-agricultural material: "
            + ", ".join(sorted(set(negative_hits))[:3])
            + ".",
        }

    if positive_score < 5:
        return {
            "path": rel_doc_path(path),
            "eligible": False,
            "topic_tag": "off_topic",
            "reason": f"Only {positive_score} farming/gardening/agroecology keyword hits in title/opening prose.",
        }

    return {
        "path": rel_doc_path(path),
        "eligible": True,
        "topic_tag": tag,
        "reason": f"{positive_score} farming/gardening/agroecology keyword hits; strongest tag is {tag}.",
    }


def build_report(summary: dict[str, object], topic_counts: Counter[str], slice_sizes: dict[str, int]) -> str:
    screened = int(summary["screened_count"])
    eligible = int(summary["eligible_count"])
    ineligible = screened - eligible
    observed_count = int(summary["observed_corpus_count"])
    stated_fraction = float(summary["coverage_fraction_of_stated_corpus"])
    observed_fraction = float(summary["coverage_fraction_of_observed_corpus"])
    conclusion = (
        "most of the screened corpus is on-topic"
        if eligible > ineligible
        else "most of the screened corpus is off-topic or unusable for this eval"
    )

    lines = [
        "# Free-form eval corpus eligibility coverage",
        "",
        f"- Screened directories: {screened} / {observed_count} observed top-level corpus directories.",
        f"- Coverage vs observed corpus: {observed_fraction:.2%}.",
        f"- Coverage vs issue-stated 1,855-doc corpus: {screened} / {STATED_CORPUS_COUNT} = {stated_fraction:.2%}.",
        f"- Markdown files found: {summary['markdown_file_count']}.",
        f"- Eligible documents: {eligible}.",
        f"- Ineligible documents: {ineligible}.",
        f"- Conclusion: {conclusion}.",
        f"- Partition seed: {SEED}.",
        "",
        "The observed corpus has 1,854 top-level directories, not the 1,855 stated in #42; "
        "268 of those directories contain no markdown file and were still counted as screened/ineligible.",
        "Because every observed top-level directory was screened, no early stopping justification is needed.",
        "",
        "## Topic counts",
        "",
    ]
    for topic, count in sorted(topic_counts.items()):
        lines.append(f"- {topic}: {count}")
    lines.extend(["", "## Slice sizes", ""])
    for item_type in ITEM_TYPES:
        lines.append(f"- {item_type}: {slice_sizes[item_type]} eligible source docs")
    lines.extend(
        [
            "",
            "## Heuristic",
            "",
            "Each directory was screened using its title and opening prose after markdown cleanup. "
            "A document was eligible when it had at least 250 prose-like words and practical "
            "farming, gardening, horticulture, soil, water, pest, crop, or agroecology signal. "
            "Directories with missing markdown, thin extracted text, off-topic scientific/medical "
            "material, reference-only front matter, or duplicate opening prose were marked ineligible.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not CORPUS_ROOT.exists():
        raise SystemExit(f"Corpus root not found: {CORPUS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SLICE_DIR.mkdir(parents=True, exist_ok=True)

    directories = sorted(path for path in CORPUS_ROOT.iterdir() if path.is_dir())
    seen_fingerprints: set[str] = set()
    documents = [classify_doc(directory, seen_fingerprints) for directory in directories]
    eligible_paths = [doc["path"] for doc in documents if doc["eligible"]]

    rng = random.Random(SEED)
    shuffled = list(eligible_paths)
    rng.shuffle(shuffled)
    slices = {item_type: [] for item_type in ITEM_TYPES}
    for index, doc_path in enumerate(shuffled):
        slices[ITEM_TYPES[index % len(ITEM_TYPES)]].append(doc_path)
    for item_type in ITEM_TYPES:
        slices[item_type].sort()
        (SLICE_DIR / f"{item_type}.json").write_text(
            json.dumps(slices[item_type], indent=2) + "\n"
        )

    observed_count = len(directories)
    markdown_file_count = sum(1 for directory in directories if list(directory.glob("*.md")))
    summary = {
        "screened_count": len(documents),
        "observed_corpus_count": observed_count,
        "stated_corpus_count": STATED_CORPUS_COUNT,
        "markdown_file_count": markdown_file_count,
        "eligible_count": len(eligible_paths),
        "ineligible_count": len(documents) - len(eligible_paths),
        "coverage_fraction_of_observed_corpus": len(documents) / observed_count,
        "coverage_fraction_of_stated_corpus": len(documents) / STATED_CORPUS_COUNT,
        "stopping_justification": "Full observed corpus screened; no early stop.",
        "partition_seed": SEED,
        "slice_sizes": {item_type: len(paths) for item_type, paths in slices.items()},
    }

    payload = {"summary": summary, "documents": documents}
    SCREEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    topic_counts = Counter(str(doc["topic_tag"]) for doc in documents if doc["eligible"])
    REPORT_PATH.write_text(build_report(summary, topic_counts, summary["slice_sizes"]))


if __name__ == "__main__":
    main()
