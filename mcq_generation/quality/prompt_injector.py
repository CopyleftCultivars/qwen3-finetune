#!/usr/bin/env python3
"""
Prompt injector for MCQ regeneration (Issue #65).

Inserts a <feedback> block into a base generation prompt based on the
per-topic rejection signals produced by feedback_aggregator.py.

The <feedback> block is injected between the </doc> tag and the guidelines
section of the prompt, following the MCQG-SRefine pattern of providing
critique context before asking for generation.

Public API
----------
inject_feedback(base_prompt, topic, feedback_context) -> str
    Returns a modified prompt string with the feedback block inserted.
    If the topic has no feedback entry, returns the base prompt unchanged.

load_feedback_context(category) -> dict
    Load quality/feedback/{category}_feedback_context.json.
    Returns {} if the file doesn't exist.

Usage as a standalone script
-----------------------------
    # Preview the injected prompt for a topic (stdout)
    conda run -n nf-benchmark python quality/prompt_injector.py \\
        --category natural_fertilizers \\
        --topic pest_control \\
        --prompt-file prompts/prompt_for_natural_fertilizers.txt
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
FEEDBACK_DIR = WORKSPACE / "quality" / "feedback"

# ---------------------------------------------------------------------------
# Dimension-specific remediation instructions
# (only injected when that dimension is identified as weakest)
# ---------------------------------------------------------------------------
_DIM_INSTRUCTIONS: dict[str, str] = {
    "distractor_quality": (
        "DISTRACTOR QUALITY: Ensure all 4 options are similar in length and grammatical structure. "
        "Each distractor must be plausible to a non-expert (not obviously absurd) "
        "but clearly wrong to a domain expert. "
        "Ground distractors in real-world misconceptions or common confusions — never invent implausible options."
    ),
    "difficulty_appropriateness": (
        "DIFFICULTY: The question must require genuine domain knowledge to answer — "
        "it should NOT be answerable by common sense, general biology, or process of elimination alone. "
        "Target the boundary between 'informed non-specialist' and 'practitioner' difficulty."
    ),
    "factual_correctness": (
        "FACTUAL ACCURACY: Verify the correct answer against the source passage verbatim. "
        "Double-check organism names, nutrient percentages, ratios, and process descriptions. "
        "If the passage is ambiguous, do not generate a question on that point."
    ),
    "domain_relevance": (
        "DOMAIN RELEVANCE: The question must test knowledge specific to this topic area, "
        "not general biology, document navigation, chapter recall, or peripheral subjects. "
        "A question about 'natural fertilizers' should require understanding of fertilizer mechanisms, "
        "not merely recall of which page discusses a topic."
    ),
    "question_clarity": (
        "QUESTION CLARITY: The question must be fully self-contained — a domain expert "
        "must be able to answer it without access to the source document. "
        "Avoid questions that require the reader to know what a specific text says; "
        "test the underlying knowledge, not the reading."
    ),
}

# Injection anchor: insert the <feedback> block immediately after </doc>
_INJECTION_ANCHOR = "</doc>"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_feedback_context(category: str) -> dict:
    """
    Load quality/feedback/{category}_feedback_context.json.
    Returns an empty dict if the file doesn't exist.
    """
    path = FEEDBACK_DIR / f"{category}_feedback_context.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inject_feedback(
    base_prompt: str,
    topic: str,
    feedback_context: dict,
) -> str:
    """
    Insert a <feedback> block into base_prompt for the given topic.

    Parameters
    ----------
    base_prompt : str
        The raw prompt template (may contain {DOCUMENT_TEXT} placeholder).
    topic : str
        The topic key to look up in feedback_context.
    feedback_context : dict
        Loaded feedback context (from load_feedback_context or direct JSON load).

    Returns
    -------
    str
        Modified prompt with <feedback> block injected after </doc>, or
        the original base_prompt if topic has no feedback entry.
    """
    entry = feedback_context.get(topic)
    if not entry:
        return base_prompt

    block = _build_feedback_block(topic, entry)

    # Inject immediately after the </doc> closing tag
    if _INJECTION_ANCHOR in base_prompt:
        return base_prompt.replace(
            _INJECTION_ANCHOR,
            f"{_INJECTION_ANCHOR}\n\n{block}",
            1,  # replace only the first occurrence
        )

    # Fallback: prepend the block before the first guideline line
    return f"{block}\n\n{base_prompt}"


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------


def _build_feedback_block(topic: str, entry: dict) -> str:
    """Build the full <feedback> block string for one topic."""
    lines: list[str] = []
    lines.append("<feedback>")
    lines.append(
        f'You are regenerating MCQs for the topic "{topic}" because previous items '
        "were rejected during quality review. "
        "Learn from the following failure patterns and avoid repeating them."
    )
    lines.append("")

    items_needed = entry.get("items_needed", "?")
    total_rejected = entry.get("total_rejected", "?")
    lines.append(f"ITEMS NEEDED: {items_needed}")
    lines.append(f"PREVIOUS ITEMS REJECTED: {total_rejected}")
    lines.append("")

    weakest = entry.get("weakest_dimensions", [])
    if weakest:
        lines.append("WEAKEST QUALITY DIMENSIONS FROM PREVIOUS ROUND:")
        for dim_entry in weakest:
            dim = dim_entry["dimension"]
            avg = dim_entry.get("avg_score", "?")
            examples = dim_entry.get("failure_examples", [])
            example_str = "; ".join(examples[:2]) if examples else ""
            lines.append(f"- {dim} (avg score: {avg}/5)")
            if example_str:
                lines.append(f"  Examples of failures: {example_str}")
        lines.append("")

    common = entry.get("common_rejection_reasons", [])
    if common:
        lines.append("COMMON REJECTION REASONS:")
        for reason in common[:3]:
            lines.append(f"- {reason}")
        lines.append("")

    # Conditional remediation instructions — only for weakest dimensions
    weak_dim_names = {d["dimension"] for d in weakest}
    relevant_instructions = [
        _DIM_INSTRUCTIONS[dim]
        for dim in _DIM_INSTRUCTIONS
        if dim in weak_dim_names
    ]
    if relevant_instructions:
        lines.append("CRITICAL INSTRUCTIONS FOR THIS REGENERATION:")
        for instruction in relevant_instructions:
            lines.append(f"- {instruction}")
        lines.append(
            "- Do NOT generate questions semantically similar to previously rejected ones."
        )
        lines.append("")

    lines.append("</feedback>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (preview mode)
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview a feedback-injected prompt for a given topic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--category", required=True, metavar="NAME",
                        help="Category name (e.g. natural_fertilizers)")
    parser.add_argument("--topic", required=True, metavar="TOPIC",
                        help="Topic to inject feedback for")
    parser.add_argument("--prompt-file", required=True, type=Path,
                        metavar="PATH", help="Base prompt .txt file")
    args = parser.parse_args()

    if not args.prompt_file.exists():
        print(f"ERROR: prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 1

    feedback_context = load_feedback_context(args.category)
    if not feedback_context:
        print(
            f"WARNING: no feedback context found for '{args.category}'. "
            "Run quality/feedback_aggregator.py first.",
            file=sys.stderr,
        )

    base_prompt = args.prompt_file.read_text(encoding="utf-8")
    modified = inject_feedback(base_prompt, args.topic, feedback_context)

    print(modified)
    return 0


if __name__ == "__main__":
    sys.exit(main())
