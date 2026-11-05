#!/usr/bin/env python3
"""#191 step 2: synthesize cross-session persona-memory dialogues.

69 real multi-session conversations is far too little to train an LPM on, so
this generates the missing cross-session volume. It is deliberately NOT
generating from scratch: each dialogue is seeded with real user turns from the
single-session Gemma4 logs (``data/persona_memory/seeds_single_session.jsonl``)
and grounded in the domain data inventoried for #172
(``data/sources/gemma4/spaces/data/{plant_mineral_profiles,crop_requirements,
local_plants_by_region,enriched_recipes}.json``), so the corpus speaks the real
vocabulary of this project's users and crops rather than inventing personas.

Shape of a generated dialogue (LPM's own training objective needs this):
  * an early session states durable facts (crop, region/hardiness, technique
    KNF/JADAM, a preference or constraint),
  * a later session asks something whose answer depends on recalling an earlier
    fact WITHOUT restating it,
  * a configurable fraction also include a fact CHANGING over time (e.g. moving
    from KNF to JADAM), so the module sees preference evolution, not only static
    recall. #196's probe categories (exact-fact recall, temporal,
    preference-evolution) come from this same material, so train and eval test
    the same kind of memory.

Teacher backend is pluggable exactly like data/make_synthetic_traces.py
(chainlink #111):
  - claude   (default): `claude -p`, local subscription, no metered cost.
  - codex / opencode: local non-interactive CLIs.
  - openrouter: metered fallback, needs OPENROUTER_API_KEY.

The held-out real multi-session conversations are never read here.

Usage:
  # smoke test: 5 dialogues via the local claude CLI
  python data/make_persona_memory_corpus.py --limit 5

  # full run (target ~600 dialogues) with parallelism
  python data/make_persona_memory_corpus.py --limit 600 --concurrency 6

  # metered fallback
  python data/make_persona_memory_corpus.py --teacher-backend openrouter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from persona_memory_common import read_jsonl, write_jsonl  # noqa: E402

log = logging.getLogger("make_persona_memory_corpus")

DEFAULT_SEEDS = REPO_ROOT / "data" / "persona_memory" / "seeds_single_session.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "persona_memory" / "synthetic_multisession.jsonl"

_CLI_RETRY_DELAYS = [2, 5, 15]
_SESSION_COUNT_CHOICES = (2, 3, 3, 4)


# --------------------------------------------------------------------------- #
# Pluggable teacher backends (mirrors data/make_synthetic_traces.py -- #111)
# --------------------------------------------------------------------------- #

def _run_teacher_subprocess(cmd: list[str], timeout: int) -> str:
    for attempt, delay in enumerate([0] + _CLI_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            if attempt == len(_CLI_RETRY_DELAYS):
                raise
            log.warning(
                "teacher CLI error (attempt %d): rc=%s stderr=%s -- retrying",
                attempt + 1, exc.returncode, (exc.stderr or "").strip()[:400],
            )
        except subprocess.TimeoutExpired as exc:
            if attempt == len(_CLI_RETRY_DELAYS):
                raise
            log.warning("teacher CLI timeout (attempt %d): %s -- retrying", attempt + 1, exc)
    raise RuntimeError("All retries exhausted")


def call_teacher_claude(system: str, prompt: str, model: str | None, max_tokens: int) -> str:
    cmd = ["claude", "-p", prompt, "--append-system-prompt", system, "--output-format", "text", "--restricted"]
    if model:
        cmd.extend(["--model", model])
    return _run_teacher_subprocess(cmd, timeout=180)


def call_teacher_codex(system: str, prompt: str, model: str | None, max_tokens: int) -> str:
    cmd = ["codex", "exec", f"{system}\n\n{prompt}"]
    if model:
        cmd.extend(["-c", f'model="{model}"'])
    return _run_teacher_subprocess(cmd, timeout=180)


def call_teacher_opencode(system: str, prompt: str, model: str | None, max_tokens: int) -> str:
    cmd = ["opencode", "run", f"{system}\n\n{prompt}", "--format", "default"]
    if model:
        cmd.extend(["--model", model])
    return _run_teacher_subprocess(cmd, timeout=180)


def make_teacher_caller(
    backend: str, model: str | None, max_tokens: int, openrouter_key: str | None
) -> Callable[[str, str], str]:
    if backend == "claude":
        return lambda system, prompt: call_teacher_claude(system, prompt, model, max_tokens)
    if backend == "codex":
        return lambda system, prompt: call_teacher_codex(system, prompt, model, max_tokens)
    if backend == "opencode":
        return lambda system, prompt: call_teacher_opencode(system, prompt, model, max_tokens)
    if backend == "openrouter":
        from judge_training_data import (  # noqa: E402
            DEFAULT_MODEL,
            _call_judge,
            _get_client,
            _load_api_key,
        )

        api_key = _load_api_key(openrouter_key)
        client, resolved_model = _get_client(api_key, model or DEFAULT_MODEL)
        return lambda system, prompt: _call_judge(system, prompt, client, resolved_model, max_tokens=max_tokens)
    raise ValueError(f"unknown --teacher-backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Seeding: real single-session vocabulary + #172 domain ground truth
# --------------------------------------------------------------------------- #

TEACHER_SYSTEM_PROMPT = (
    "You generate realistic, multi-session natural-farming advisory dialogues for training "
    "a long-term personal-memory module. The human is a real small-scale grower with a specific "
    "crop, region, and farming technique. Return ONLY valid JSON, with no markdown code fences "
    "and no commentary, matching the requested schema exactly."
)

TEACHER_USER_TEMPLATE = """Write one multi-session grower<->assistant dialogue.

Requirements:
- {n_sessions} sessions, separated in time (the first has days_later 0; later sessions get a plausible gap of weeks to months).
- Session 1 states durable personal facts naturally in conversation: a crop, a region/climate (use the region below), and a farming technique (KNF/JADAM/Vrikshayurveda-style, or a stated constraint/preference).
- At least one LATER session asks a question whose useful answer depends on recalling an earlier fact WITHOUT the human restating it.
- {evolution_instruction}
- Keep the human's voice plain and specific, like a small farmer texting an assistant. Assistant turns are concise and practical.
- Ground any nutrient/recipe/crop values in the DOMAIN GROUND TRUTH below where relevant; never contradict it.
- These are NEW personas. Do not copy any example sentence verbatim.

DOMAIN GROUND TRUTH (region, crop, bioaccumulator, recipe)
{domain_block}

REAL USER VOICE EXAMPLES (vocabulary/scenario cues only -- do not reuse verbatim, do not copy facts from them)
{voice_block}

Return ONLY this JSON object:
{{
  "sessions": [
    {{"days_later": 0, "turns": [{{"from": "human", "text": "..."}}, {{"from": "assistant", "text": "..."}}]}}
  ],
  "facts": [
    {{"fact": "<the durable fact, e.g. crop or region or technique>", "stated_session": 0, "recalled_session": 1, "changed": false, "changed_to": null}}
  ]
}}
Include 2-4 entries in "facts". If a fact changes, set changed=true and put the new value in "changed_to".
"""


def load_seeds(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for fragment in row.get("seed_fragments", []):
                text = str(fragment.get("text", "")).strip()
                if text:
                    seeds.append({"conversation_id": row.get("conversation_id"), "text": text})
            if limit is not None and len(seeds) >= limit:
                break
    return seeds


def load_domain_reference() -> dict[str, list[dict[str, Any]]]:
    """The valid, already-ported #172 sources (see persona_memory_common)."""
    from persona_memory_common import load_domain_reference as _load

    return _load()


def build_briefs(
    seeds: list[dict[str, Any]],
    domain: dict[str, list[dict[str, Any]]],
    count: int,
    *,
    seed: int = 20260918,
    evolution_fraction: float = 1 / 3,
) -> list[dict[str, Any]]:
    """Deterministically assemble ``count`` generation briefs.

    Each brief pins a region/crop/bioaccumulator/recipe so generated facts are
    in-domain, plus a few real user-voice fragments sampled without replacement
    until the seed pool is exhausted.
    """
    rng = random.Random(seed)
    regions = domain.get("local_plants_by_region", [])
    crops = domain.get("crop_requirements", [])
    plants = domain.get("plant_mineral_profiles", [])
    recipes = domain.get("enriched_recipes", [])
    if not (regions and crops and recipes):
        raise SystemExit("domain reference data is missing; cannot ground generation")

    # Rotate without replacement through the seed pool so every real fragment can
    # influence some brief, then reshuffle for the next cycle.
    pool = list(seeds)
    rng.shuffle(pool)

    briefs: list[dict[str, Any]] = []
    for index in range(count):
        region = rng.choice(regions)
        crop = rng.choice(crops)
        recipe = rng.choice(recipes)
        plant = rng.choice(plants) if plants else {}
        if not pool:
            pool = list(seeds)
            rng.shuffle(pool)
        voice: list[dict[str, Any]] = []
        for _ in range(3):
            if pool:
                voice.append(pool.pop())
        n_sessions = rng.choice(_SESSION_COUNT_CHOICES)
        brief = {
            "brief_index": index,
            "n_sessions": n_sessions,
            "region": region.get("Location"),
            "region_plants": region.get("Plants", [])[:5],
            "region_seasonality": region.get("seasonality_note"),
            "crop": crop.get("Crop"),
            "crop_scientific_name": crop.get("Scientific Name"),
            "crop_npk": crop.get("NPK Requirements"),
            "crop_additional": crop.get("Additional Requirements"),
            "bioaccumulator": plant.get("common_name") or plant.get("scientific_name"),
            "recipe": recipe.get("recipe_name"),
            "recipe_abbrev": recipe.get("recipe_abbrev"),
            "recipe_nutrient": recipe.get("primary_nutrient"),
            "recipe_dilution": recipe.get("dilution_ratio"),
            "preference_evolution": rng.random() < evolution_fraction,
            "voice_examples": [v["text"] for v in voice],
            "seed_refs": [hashlib.sha1(v["text"].encode()).hexdigest()[:12] for v in voice],
            "domain_refs": {
                "region": region.get("Location"),
                "crop": crop.get("Crop"),
                "recipe": recipe.get("recipe_abbrev"),
                "bioaccumulator": plant.get("common_name"),
            },
        }
        briefs.append(brief)
    return briefs


def _domain_block(brief: dict[str, Any]) -> str:
    return json.dumps(
        {
            "region": brief["region"],
            "region_native_plants": brief["region_plants"],
            "crop": brief["crop"],
            "crop_scientific_name": brief["crop_scientific_name"],
            "crop_npk_requirements": brief["crop_npk"],
            "crop_growing_conditions": brief["crop_additional"],
            "bioaccumulator": brief["bioaccumulator"],
            "recipe": brief["recipe"],
            "recipe_abbrev": brief["recipe_abbrev"],
            "recipe_primary_nutrient": brief["recipe_nutrient"],
            "recipe_dilution_ratio": brief["recipe_dilution"],
        },
        ensure_ascii=False,
        indent=2,
    )


def build_teacher_prompt(brief: dict[str, Any]) -> tuple[str, str]:
    if brief["preference_evolution"]:
        evolution = (
            "In one later session, have a previously stated preference or technique CHANGE "
            "(e.g. switching from one natural-farming method to another, or changing the target crop), "
            "and record that change in \"facts\"."
        )
    else:
        evolution = "Keep the durable facts stable across sessions (static recall only)."
    voice_block = "\n".join(f"- {text}" for text in brief["voice_examples"]) or "- (no voice examples available)"
    prompt = TEACHER_USER_TEMPLATE.format(
        n_sessions=brief["n_sessions"],
        evolution_instruction=evolution,
        domain_block=_domain_block(brief),
        voice_block=voice_block,
    )
    return TEACHER_SYSTEM_PROMPT, prompt


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_dialogue(raw: str) -> dict[str, Any]:
    """Strip fences/prose and parse the first JSON object; raise ValueError if invalid."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise ValueError("no JSON object in teacher output")
        payload = json.loads(match.group(0))
    validate_dialogue(payload)
    return payload


def validate_dialogue(payload: dict[str, Any]) -> None:
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or len(sessions) < 2:
        raise ValueError("need at least 2 sessions")
    for session in sessions:
        turns = session.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("each session needs at least one turn")
        for turn in turns:
            if turn.get("from") not in ("human", "assistant") or not str(turn.get("text", "")).strip():
                raise ValueError("turn needs from in {human,assistant} and non-empty text")
    facts = payload.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("need at least one fact")
    recalled = [
        f for f in facts
        if isinstance(f, dict) and int(f.get("recalled_session", 0) or 0) > int(f.get("stated_session", 0) or 0)
    ]
    if not recalled:
        raise ValueError("no fact is recalled in a later session")


def normalize_dialogue(payload: dict[str, Any], brief: dict[str, Any], backend: str, model: str | None) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    for index, session in enumerate(payload["sessions"]):
        sessions.append(
            {
                "session_index": index,
                "days_later": int(session.get("days_later", 0) or 0),
                "turns": [
                    {"from": turn["from"], "text": str(turn["text"]).strip()}
                    for turn in session["turns"]
                ],
            }
        )
    digest = hashlib.sha1(
        json.dumps(sessions, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    return {
        "persona_id": f"synth:{digest}",
        "brief_index": brief["brief_index"],
        "source": "synthetic_multisession",
        "generator": "data/make_persona_memory_corpus.py",
        "teacher_backend": backend,
        "teacher_model": model,
        "seed_refs": brief["seed_refs"],
        "domain_refs": brief["domain_refs"],
        "preference_evolution": brief["preference_evolution"],
        "sessions": sessions,
        "facts": payload["facts"],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100, help="Number of dialogues to generate.")
    parser.add_argument("--seed", type=int, default=20260918, help="Brief-sampling RNG seed.")
    parser.add_argument("--evolution-fraction", type=float, default=1 / 3)
    parser.add_argument(
        "--teacher-backend",
        choices=("claude", "codex", "opencode", "openrouter"),
        default="claude",
        help="claude/codex/opencode use a local subscription CLI; openrouter is the metered fallback.",
    )
    parser.add_argument("--model", default=None, help="Backend-specific model override.")
    parser.add_argument("--openrouter-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3, help="Retries per brief on invalid JSON.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Load an existing --output and only generate briefs not already present "
        "(resumable top-up after transient teacher failures).",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.seeds.exists():
        raise SystemExit(
            f"seeds not found: {args.seeds}. Run scripts/build_persona_memory_corpus.py first."
        )

    seeds = load_seeds(args.seeds)
    domain = load_domain_reference()
    briefs = build_briefs(
        seeds, domain, args.limit, seed=args.seed, evolution_fraction=args.evolution_fraction
    )
    n_evolution = sum(1 for b in briefs if b["preference_evolution"])
    log.info("%d seeds -> %d briefs (%d preference-evolution)", len(seeds), len(briefs), n_evolution)

    existing: list[dict[str, Any]] = []
    if args.merge and args.output.exists():
        existing = read_jsonl(args.output)
        done = {row.get("brief_index") for row in existing}
        before = len(briefs)
        briefs = [b for b in briefs if b["brief_index"] not in done]
        log.info("merge: %d existing dialogues cover %d briefs; %d/%d briefs left",
                 len(existing), len(done), len(briefs), before)

    teacher_call = make_teacher_caller(args.teacher_backend, args.model, args.max_tokens, args.openrouter_key)

    def process_one(brief: dict[str, Any]) -> dict[str, Any] | None:
        system, prompt = build_teacher_prompt(brief)
        last_error: Exception | None = None
        for attempt in range(args.max_attempts):
            try:
                raw = teacher_call(system, prompt)
                payload = parse_dialogue(raw)
                return normalize_dialogue(payload, brief, args.teacher_backend, args.model)
            except Exception as exc:  # noqa: BLE001 -- teacher/parse failures are per-brief
                last_error = exc
                log.warning("brief %d attempt %d failed: %s", brief["brief_index"], attempt + 1, exc)
        log.error("brief %d exhausted attempts: %s", brief["brief_index"], last_error)
        return None

    results: list[dict[str, Any]] = []
    failures = 0
    if args.concurrency <= 1:
        iterator = (process_one(b) for b in briefs)
        for result in iterator:
            if result is None:
                failures += 1
            else:
                results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for result in pool.map(process_one, briefs):
                if result is None:
                    failures += 1
                else:
                    results.append(result)

    combined = existing + results
    n_written = write_jsonl(args.output, combined)
    print(f"wrote {args.output} ({n_written} dialogues, {failures} failed this pass)")
    print(f"briefs_left={len(briefs)} preference_evolution={n_evolution}")
    return 0 if combined else 1


if __name__ == "__main__":
    raise SystemExit(main())
