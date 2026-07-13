#!/usr/bin/env python3
"""
Contamination and semantic deduplication filter for MCQ benchmarks.

This script implements two filters:
1. Contamination check: Tests if an LLM can identify the correct answer from
   answer options alone (without the question stem).
2. Semantic deduplication: Uses SentenceBERT embeddings to detect near-duplicate
   questions (cosine similarity > 0.95).

Part of Layer 1 automated filtering (Issue #51).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# OpenRouter API for contamination check (using Claude Sonnet 4.6)
import requests
import os


def load_benchmark(jsonl_path: Path) -> List[Dict]:
    """Load benchmark items from JSONL file."""
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                items.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
    return items


def check_contamination_batch(
    items: List[Dict],
    api_key: str,
    batch_size: int = 10
) -> Set[str]:
    """
    Check if Claude Sonnet 4.6 can identify correct answers from options alone.

    Returns set of question_ids that are contaminated (LLM guessed correctly).
    """
    contaminated = set()

    if not api_key:
        print("Warning: No OPENROUTER_API_KEY set. Skipping contamination check.", file=sys.stderr)
        return contaminated

    print(f"Running contamination check on {len(items)} items...")

    # Process in batches to reduce API calls
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        # Build prompt asking LLM to guess answer from options only
        prompt_parts = [
            "For each question below, I will give you ONLY the answer choices "
            "(A, B, C, D) without the question stem. Try to identify which option "
            "is most likely correct based solely on the options.\n\n"
        ]

        for idx, item in enumerate(batch, 1):
            prompt_parts.append(f"Question {idx} (ID: {item['question_id']}):\n")
            prompt_parts.append(f"A: {item['A']}\n")
            prompt_parts.append(f"B: {item['B']}\n")
            prompt_parts.append(f"C: {item['C']}\n")
            prompt_parts.append(f"D: {item['D']}\n")
            prompt_parts.append("\n")

        prompt_parts.append(
            "For each question, respond with ONLY the question ID and your "
            "guess (A, B, C, or D), one per line.\n"
        )
        prompt_parts.append("Format: <question_id>: <A|B|C|D>\n")
        prompt_parts.append("Example: NF-0001: B")

        prompt = "".join(prompt_parts)

        # Call OpenRouter API with Claude Sonnet 4.6
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "anthropic/claude-sonnet-4.6",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            llm_response = result['choices'][0]['message']['content']

            # Parse LLM guesses
            for line in llm_response.strip().split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue

                try:
                    qid_part, guess_part = line.split(':', 1)
                    qid = qid_part.strip()
                    guess = guess_part.strip().upper()

                    # Find matching item in batch
                    matching_items = [item for item in batch if item['question_id'] == qid]
                    if matching_items:
                        item = matching_items[0]
                        if guess == item['answer']:
                            contaminated.add(qid)
                            print(
                                f"  Contaminated: {qid} "
                                f"(LLM guessed {guess} correctly)",
                                file=sys.stderr
                            )
                except (ValueError, KeyError):
                    continue

        except requests.exceptions.RequestException as e:
            print(
                f"Warning: API call failed for batch {i//batch_size + 1}: {e}",
                file=sys.stderr
            )
            continue

    return contaminated


def compute_embeddings(items: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Compute SentenceBERT embeddings for all question stems.

    Uses a lightweight model (all-MiniLM-L6-v2) that's fast and effective.
    """
    print(f"Loading SentenceBERT model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Extract question stems
    questions = [item['question'] for item in items]

    print(f"Computing embeddings for {len(questions)} questions...")
    embeddings = model.encode(
        questions,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


def find_duplicates(
    items: List[Dict],
    embeddings: np.ndarray,
    threshold: float = 0.95
) -> Set[str]:
    """
    Find semantic duplicates using cosine similarity on embeddings.

    Returns set of question_ids to remove (keeps first occurrence).
    """
    duplicates = set()
    n = len(items)

    print(
        f"Computing pairwise similarities for {n} items "
        f"(threshold={threshold})..."
    )

    # Compute cosine similarity matrix
    similarities = cosine_similarity(embeddings)

    # Find pairs above threshold (excluding self-similarity)
    duplicate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] >= threshold:
                duplicate_pairs.append((i, j, similarities[i, j]))

    print(f"Found {len(duplicate_pairs)} duplicate pairs")

    # Keep first occurrence, mark later ones for removal
    for i, j, sim in duplicate_pairs:
        qid_i = items[i]['question_id']
        qid_j = items[j]['question_id']

        # Remove the one that comes later (higher index)
        duplicates.add(qid_j)
        print(
            f"  Duplicate: {qid_j} (similar to {qid_i}, sim={sim:.3f})",
            file=sys.stderr
        )

    return duplicates


def filter_benchmark(
    items: List[Dict],
    contaminated: Set[str],
    duplicates: Set[str]
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Filter out contaminated and duplicate items.

    Returns:
        - List of clean items
        - Dict of rejected items by reason
    """
    clean_items = []
    rejected = defaultdict(list)

    for item in items:
        qid = item['question_id']

        if qid in contaminated:
            rejected['contaminated'].append(item)
        elif qid in duplicates:
            rejected['duplicate'].append(item)
        else:
            clean_items.append(item)

    return clean_items, dict(rejected)


def save_results(
    clean_items: List[Dict],
    rejected: Dict[str, List[Dict]],
    output_path: Path,
    report_path: Path
):
    """Save filtered benchmark and rejection report."""

    # Save clean benchmark
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in clean_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(clean_items)} clean items to: {output_path}")

    # Save rejection report
    total_items = len(clean_items) + sum(len(items) for items in rejected.values())

    report = {
        'total_input': total_items,
        'total_output': len(clean_items),
        'total_rejected': sum(len(items) for items in rejected.values()),
        'rejected_by_reason': {
            reason: len(items) for reason, items in rejected.items()
        },
        'retention_rate': len(clean_items) / total_items if total_items > 0 else 0,
        'rejected_items': rejected
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved rejection report to: {report_path}")

    # Print summary
    print("\n=== Filtering Summary ===")
    print(f"Total input items: {report['total_input']}")
    print(f"Total output items: {report['total_output']}")
    print(f"Total rejected: {report['total_rejected']}")
    print(f"Retention rate: {report['retention_rate']:.1%}")
    print("\nRejected by reason:")
    for reason, count in report['rejected_by_reason'].items():
        print(f"  {reason}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter contaminated and duplicate MCQ items"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL benchmark file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSONL file (default: input_filtered.jsonl)"
    )
    parser.add_argument(
        "-r", "--report",
        type=Path,
        help="Rejection report JSON file (default: input_filter_report.json)"
    )
    parser.add_argument(
        "--skip-contamination",
        action="store_true",
        help="Skip contamination check (only do deduplication)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for deduplication (default: 0.95)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

    # Set default output paths
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_filtered.jsonl"
    if args.report is None:
        args.report = args.input.parent / f"{args.input.stem}_filter_report.json"

    # Load benchmark
    print(f"Loading benchmark from: {args.input}")
    items = load_benchmark(args.input)
    print(f"Loaded {len(items)} items")

    # Check contamination
    if args.skip_contamination:
        print("\nSkipping contamination check (--skip-contamination)")
        contaminated = set()
    else:
        api_key = os.environ.get('OPENROUTER_API_KEY')
        contaminated = check_contamination_batch(items, api_key)
        print(f"\nContamination check complete: {len(contaminated)} contaminated items")

    # Compute embeddings and find duplicates
    print("\n=== Semantic Deduplication ===")
    embeddings = compute_embeddings(items, model_name=args.model)
    duplicates = find_duplicates(items, embeddings, threshold=args.similarity_threshold)
    print(f"Deduplication complete: {len(duplicates)} duplicate items")

    # Filter benchmark
    print("\n=== Filtering ===")
    clean_items, rejected = filter_benchmark(items, contaminated, duplicates)

    # Save results
    save_results(clean_items, rejected, args.output, args.report)


if __name__ == "__main__":
    main()
