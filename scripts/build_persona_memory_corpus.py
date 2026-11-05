#!/usr/bin/env python3
"""#191 step 1: turn the real Gemma4 logs into probe material + generator seeds.

Reads $PERSONA_CONVERSATIONS_DIR (the
real ``conversations/`` directory; the sibling ``conversations (1)/`` is a strict
subset and is not read), drops ``evaluation == "bad"`` turns, and splits by
structure:

  * **multi-session** -- a real >12h gap between consecutive turns under one
    ``conversation_id``. Genuine longitudinal/persistent-identity data. Written to
    ``heldout_multisession.jsonl`` and HELD OUT of training *and* of synthetic
    augmentation, so #196's probe set stays clean. A contamination check still
    runs on it (a probe that overlaps the existing benchmark is unusable for
    interpretation too).
  * **single-session** -- everything else (~1541 conversations, mostly one
    sitting). Written to ``seeds_single_session.jsonl`` and used as the
    synthetic generator's real-vocabulary seed material, not as direct
    cross-session-recall training data.

Contamination filtering is on by default: any conversation whose turns echo a
ladder/val_v3/eval_freeform-dev item is dropped from the output and recorded in
the report instead. See scripts/persona_memory_contamination.py.

Usage:
  python scripts/build_persona_memory_corpus.py
  python scripts/build_persona_memory_corpus.py --output-dir data/persona_memory
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from persona_memory_common import (  # noqa: E402
    DEFAULT_CONVERSATIONS_DIR,
    REPO_ROOT,
    SESSION_GAP_HOURS,
    Conversation,
    annotate_sessions,
    load_conversations,
    session_to_json,
    split_sessions,
    write_jsonl,
)
from persona_memory_contamination import (  # noqa: E402
    ContaminationIndex,
    candidate_text,
    check_rows,
    load_eval_sources,
)

SEED_CUE_RE = re.compile(
    r"\b(zone|hardiness|i am in|i'm in|my (?:family )?(?:farm|garden|orchard|homestead|property)|"
    r"acres?|hectares?|knf|jadam|vrikshayurveda|compost|vermicompost|biochar|fermented|"
    r"faa|fpj|imo|lab|wca|ohn|comfrey|nettle|borage|yarrow|buckwheat|corn|tomato|pepper|"
    r"cannabis|brassica|orchard|aphid|soil|fertiliz|irrigat|mulch|loam|clay|sand|ph|"
    r"washington|oklahoma|california|texas|oregion|hawaii|india|philippines|australia)\b",
    re.IGNORECASE,
)


def conversation_to_row(conversation: Conversation, source: str) -> dict[str, Any]:
    session_lists = split_sessions(conversation.turns)
    sessions = [session_to_json(s, i) for i, s in enumerate(session_lists)]
    return {
        "persona_id": conversation.persona_id(),
        "conversation_id": conversation.conversation_id,
        "source": source,
        "n_turns": len(conversation.turns),
        "session_count": conversation.session_count,
        "span_hours": round(conversation.span_hours, 3),
        "max_gap_hours": conversation.max_gap_hours,
        "start": conversation.turns[0].created_at.isoformat(),
        "end": conversation.turns[-1].created_at.isoformat(),
        "sessions": sessions,
    }


def seed_fragments(conversation: Conversation) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for turn in conversation.turns:
        if not turn.is_human:
            continue
        text = turn.text.strip()
        if not (15 <= len(text) <= 800):
            continue
        if turn.evaluation == "good" or SEED_CUE_RE.search(text):
            fragments.append(
                {
                    "text": text,
                    "createdAt": turn.created_at.isoformat(),
                    "human_rated": turn.evaluation == "good",
                }
            )
    return fragments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conversations-dir", type=Path, default=DEFAULT_CONVERSATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "persona_memory")
    parser.add_argument("--gap-hours", type=float, default=SESSION_GAP_HOURS)
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument(
        "--skip-contamination",
        action="store_true",
        help="Write the splits without dropping contaminated conversations (report only).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conversations = load_conversations(args.conversations_dir, min_turns=args.min_turns)
    conversations = [annotate_sessions(c, args.gap_hours) for c in conversations]

    multi = [c for c in conversations if c.is_multisession]
    single = [c for c in conversations if not c.is_multisession]
    total_turns = sum(len(c.turns) for c in conversations)

    heldout_rows = [conversation_to_row(c, "gemma4_real_multisession_heldout") for c in multi]
    seed_rows = [conversation_to_row(c, "gemma4_real_single_session") for c in single]
    for row, conversation in zip(seed_rows, single):
        row["seed_fragments"] = seed_fragments(conversation)

    index = ContaminationIndex(load_eval_sources())
    if args.skip_contamination:
        clean_heldout, annotated_heldout, clean_seeds, dirty_seeds = heldout_rows, [], seed_rows, []
    else:
        clean_seeds, dirty_seeds = check_rows(seed_rows, index)
        # The held-out sessions are never trained on, so an overlap with val_v3
        # does not leak into training. But val_v3 turns are verbatim slices of
        # these same logs (confirmed in the report), so #196 must know which
        # conversations are already represented in the existing benchmark.
        # Keep every conversation, annotate the overlap, and expose the clean
        # subset for probe construction.
        annotated_heldout, clean_heldout = [], []
        for position, row in enumerate(heldout_rows):
            hits = index.classify(candidate_text(row))
            if hits:
                row = dict(row)
                row["eval_overlap"] = hits
                heldout_rows[position] = row
                annotated_heldout.append(row)
            else:
                clean_heldout.append(row)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    n_heldout = write_jsonl(out / "heldout_multisession.jsonl", heldout_rows)
    n_heldout_clean = write_jsonl(out / "heldout_multisession_clean.jsonl", clean_heldout)
    n_seeds = write_jsonl(out / "seeds_single_session.jsonl", clean_seeds)
    if dirty_seeds:
        write_jsonl(out / "seeds_single_session_contaminated.jsonl", dirty_seeds)

    all_multi_turns = sum(c["n_turns"] for c in heldout_rows)
    all_seed_turns = sum(c["n_turns"] for c in seed_rows)
    n_fragments = sum(len(r.get("seed_fragments", [])) for r in seed_rows)
    manifest = {
        "format": "persona_memory_real_split_v1",
        "issue": 191,
        "conversations_dir": str(args.conversations_dir),
        "gap_hours": args.gap_hours,
        "min_turns": args.min_turns,
        "conversations_total": len(conversations),
        "turns_total_after_bad_filter": total_turns,
        "multisession_conversations": len(multi),
        "multisession_with_ge_10_turns": sum(1 for c in multi if len(c.turns) >= 10),
        "single_session_conversations": len(single),
        "heldout_rows_written": n_heldout,
        "heldout_rows_clean": n_heldout_clean,
        "heldout_eval_overlap": len(annotated_heldout),
        "heldout_turns": all_multi_turns,
        "seeds_rows_written": n_seeds,
        "seed_turns": all_seed_turns,
        "seed_fragments": n_fragments,
        "seeds_contaminated_dropped": len(dirty_seeds),
        "contamination_filter_applied": not args.skip_contamination,
        "eval_sources_checked": sorted({item.source for item in index.items.values()}),
        "finding": (
            "val_v3 turns are verbatim slices of these same Gemma4 logs; held-out "
            "multi-session conversations are annotated with eval_overlap (not dropped, "
            "since they are never trained on) and the overlap-free subset is written "
            "to heldout_multisession_clean.jsonl for #196."
        ),
        "outputs": {
            "heldout_multisession": str(out / "heldout_multisession.jsonl"),
            "heldout_multisession_clean": str(out / "heldout_multisession_clean.jsonl"),
            "seeds_single_session": str(out / "seeds_single_session.jsonl"),
            "seeds_single_session_contaminated": str(out / "seeds_single_session_contaminated.jsonl"),
        },
    }
    report = {
        "format": "persona_memory_real_contamination_v1",
        "issue": 191,
        "sealed_eval_freeform_test_opened": False,
        "eval_sources": sorted({item.source for item in index.items.values()}),
        "seeds_contaminated": dirty_seeds,
        "heldout_eval_overlap": annotated_heldout,
    }
    report_path = REPO_ROOT / "results" / "persona_memory_real_contamination.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "real_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
