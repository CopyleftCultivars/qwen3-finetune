#!/usr/bin/env python3
"""
grpo_dataset.py - Load CopyleftCultivars/Agriculture-Agent-RL-Training-Data rollout
prompts for GRPO training, excluding sources with known reward-inflation bias.

Usage:
  from autoresearch.grpo_dataset import load_rollout_prompts
  ds = load_rollout_prompts()  # training-host only, requires `datasets`

Dataset shape (HF): {prompt, completion, reward, task, info}, ~6.19k rows, single
`train` split. GRPOTrainer generates its own completions online during training —
only `prompt` (+ `info`/`task` passthrough as reward_fn kwargs) is used as the
rollout seed set. The dataset's own `completion`/`reward` columns are the
*original* export's recorded rollout and are dropped here; they are not replayed
as training pairs.

Known bias (confirmed via full-dataset scan, 2026-08-13, n=6191):
  generator_agent  mean_reward  frac_reward==1.0
  agy              0.999        0.996   <- self-judging inflation, EXCLUDED
  hermes           0.819        0.541
  hermes_cli       0.698        0.285
`agy`-sourced rows are identified via info.generation_meta.generator_agent == "agy"
(not a top-level "source" field). Excluded by default everywhere this dataset is
touched — both as rollout seeds (so the model doesn't preferentially rehearse
agy's prompt distribution) and, separately, autoresearch/grpo_reward.py's reward
functions are written from scratch against info.reward_components' *semantics*
rather than calibrated to agy's inflated numbers. Do not re-include without a new,
falsified lever entry in levers.md's Tier 4 (RL second-stage) section.

Known scaffolding-tool contamination (found 2026-08-13, scanning every recorded
`completion` tool_call across the full non-agy pool, n=4964): the dataset card
describes a clean ~9-tool cultivars-mcp/PubChem surface, but the actual recorded
completions call ~37 distinct tool names. Most of the extras are legitimate
plant-genomics/chemistry lookups the card just didn't enumerate (get_orthologs,
search_pubmed_for_gene, get_molecular_formula, lookup_uniprot_entry, etc. --
folded into DOMAIN_TOOL_NAMES below). But 22.4% of rows (1110/4964) call a tool
that is clearly agent-harness scaffolding, not a deployed farming-assistant
capability: terminal, read_file, search_files, web_search, memory, todo,
session_search, skill_view, tool_search, tool_describe, get_prompt, list_prompts,
list_resources, read_resource, plus a residual few malformed `tool_call`-wrapper
artifacts (see SCAFFOLDING_TOOL_NAMES). These reflect the generating agent's own
research/scratchpad tools (generator_agent hermes/hermes_cli, seed_generator_agent
claude) leaking into the recorded rollout, not something a deployed model should
learn to imitate. filter_scaffolding_contaminated() drops any row whose recorded
completion touches a non-domain tool; 77.6% of non-agy rows survive this filter.

Separately: the dataset's `prompt` column never included the tool JSON schemas
that a real deployment would show the model via a `tools=` argument to
apply_chat_template() -- TRL's GRPOTrainer doesn't support a per-row tools column
either, so schemas were never rendered into any prompt in training or eval before
this was caught. bake_tools_into_system_message() reconstructs the schemas
(TOOL_SCHEMAS, built from observed argument-key frequency in the real data) and
bakes the exact same `<tools>...</tools>` block the tokenizer's own chat_template.jinja
would render given `tools=`, directly into the system message text -- this works
regardless of whether the downstream caller's chat-template plumbing supports a
`tools=` kwarg.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

DEFAULT_DATASET_ID = "CopyleftCultivars/Agriculture-Agent-RL-Training-Data"
EXCLUDED_GENERATOR_AGENTS = frozenset({"agy"})
DEFAULT_HOLDOUT_FRACTION = 0.1
DEFAULT_HOLDOUT_SEED = 42


def generator_agent_of(row: dict) -> str:
    return ((row.get("info") or {}).get("generation_meta") or {}).get("generator_agent", "?")


def record_uuid_of(row: dict) -> str:
    return (row.get("info") or {}).get("record_uuid", "")


def holdout_bucket(record_uuid: str, seed: int = DEFAULT_HOLDOUT_SEED, fraction: float = DEFAULT_HOLDOUT_FRACTION) -> str:
    """Deterministically assign a row to 'train' or 'eval' from its record_uuid.

    Same (record_uuid, seed, fraction) always yields the same bucket regardless
    of --max-samples or dataset ordering, so train_grpo.py (holdout="train") and
    score_grpo_primary.py (holdout="eval") never see overlapping prompts even
    across separate invocations/runs.
    """
    digest = hashlib.sha256(f"{seed}:{record_uuid}".encode("utf-8")).hexdigest()
    bucket_value = int(digest[:8], 16) / 0xFFFFFFFF
    return "eval" if bucket_value < fraction else "train"


def iter_local_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_excluded_sources(
    rows: Iterable[dict],
    exclude_sources: frozenset[str] = EXCLUDED_GENERATOR_AGENTS,
) -> Iterator[dict]:
    for row in rows:
        if generator_agent_of(row) not in exclude_sources:
            yield row


def source_counts(rows: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        src = generator_agent_of(row)
        counts[src] = counts.get(src, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Tool taxonomy -- schemas for the ~9-tool cultivars-mcp/PubChem-style domain
# surface described by the dataset card, expanded to the ~22 legitimate
# plant-genomics/chemistry lookups actually observed in the recorded
# completions (frequency-ranked argument keys from the real data, capped to the
# most common few per tool -- this is a reconstruction, not the true cultivars-mcp
# API contract, which isn't available to this repo). SCAFFOLDING_TOOL_NAMES is
# the complementary set of harness/meta tools + malformed artifacts that
# filter_scaffolding_contaminated() uses to drop contaminated rows.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "find_trait_genes": {
        "type": "function",
        "function": {
            "name": "find_trait_genes",
            "description": "Find genes underlying a named agronomic trait.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trait": {"type": "string", "description": "Trait name, e.g. 'drought tolerance'."},
                    "target_species": {"type": "string", "description": "Optional species to restrict results to."},
                },
                "required": ["trait"],
            },
        },
    },
    "list_trait_categories": {
        "type": "function",
        "function": {
            "name": "list_trait_categories",
            "description": "List known trait categories, optionally filtered by a query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional filter string."},
                    "limit": {"type": "integer", "description": "Max results to return."},
                },
                "required": [],
            },
        },
    },
    "lookup_gene": {
        "type": "function",
        "function": {
            "name": "lookup_gene",
            "description": "Look up a gene by symbol/id within a species.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string", "description": "Gene symbol or identifier."},
                    "species": {"type": "string", "description": "Species name."},
                    "expand": {"type": "boolean", "description": "Include expanded cross-reference detail."},
                },
                "required": ["gene", "species"],
            },
        },
    },
    "lookup_gene_evidence": {
        "type": "function",
        "function": {
            "name": "lookup_gene_evidence",
            "description": "Get the curation/evidence xref chain for a specific gene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stable_id": {"type": "string", "description": "Ensembl-style stable gene id."},
                    "species": {"type": "string", "description": "Species name."},
                },
                "required": ["stable_id"],
            },
        },
    },
    "translate_trait_to_species": {
        "type": "function",
        "function": {
            "name": "translate_trait_to_species",
            "description": "Translate a trait to its known genetic basis in one or more target species.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trait": {"type": "string"},
                    "target_species": {"type": "string", "description": "Species name, or a list of species."},
                },
                "required": ["trait"],
            },
        },
    },
    "list_maize_nam_founders": {
        "type": "function",
        "function": {
            "name": "list_maize_nam_founders",
            "description": "List maize NAM founder lines, optionally filtered by subpopulation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subpopulation": {"type": "string", "description": "Optional subpopulation filter, e.g. 'Popcorn'."},
                },
                "required": [],
            },
        },
    },
    "lookup_compound": {
        "type": "function",
        "function": {
            "name": "lookup_compound",
            "description": "Look up a chemical compound by name (PubChem-backed).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Compound name."}},
                "required": ["name"],
            },
        },
    },
    "resolve_accession": {
        "type": "function",
        "function": {
            "name": "resolve_accession",
            "description": "Resolve a farmer-described heritage/seed variety to a genebank accession.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Variety/seed name as described by the farmer."},
                    "crop_type": {"type": "string"},
                    "region": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    "list_orphan_crop_requests": {
        "type": "function",
        "function": {
            "name": "list_orphan_crop_requests",
            "description": "List outstanding orphan-crop data requests, optionally filtered by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    "get_gene": {
        "type": "function",
        "function": {
            "name": "get_gene",
            "description": "Fetch a gene record by its stable id.",
            "parameters": {
                "type": "object",
                "properties": {"gene_id": {"type": "string"}},
                "required": ["gene_id"],
            },
        },
    },
    "get_orthologs": {
        "type": "function",
        "function": {
            "name": "get_orthologs",
            "description": "Find orthologs of a gene in one or more target species.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene": {"type": "string"},
                    "species": {"type": "string", "description": "Source species."},
                    "target_species": {"type": "string", "description": "Target species (or list of species)."},
                },
                "required": ["gene"],
            },
        },
    },
    "get_sequence": {
        "type": "function",
        "function": {
            "name": "get_sequence",
            "description": "Fetch the sequence for a gene by stable id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stable_id": {"type": "string"},
                    "species": {"type": "string"},
                },
                "required": ["stable_id"],
            },
        },
    },
    "get_string_interactions": {
        "type": "function",
        "function": {
            "name": "get_string_interactions",
            "description": "Fetch known protein-protein interactions for a gene (STRING-backed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "species": {"type": "string"},
                    "stable_id": {"type": "string"},
                },
                "required": ["stable_id"],
            },
        },
    },
    "get_variant": {
        "type": "function",
        "function": {
            "name": "get_variant",
            "description": "Fetch a known genetic variant by line or stable id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line": {"type": "string"},
                    "stable_id": {"type": "string"},
                    "species": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    "get_molecular_formula": {
        "type": "function",
        "function": {
            "name": "get_molecular_formula",
            "description": "Get the molecular formula for a named compound (PubChem-backed).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    "lookup_uniprot_entry": {
        "type": "function",
        "function": {
            "name": "lookup_uniprot_entry",
            "description": "Look up a UniProt entry by id.",
            "parameters": {
                "type": "object",
                "properties": {"uniprot_id": {"type": "string"}},
                "required": ["uniprot_id"],
            },
        },
    },
    "lookup_ens_stable_id": {
        "type": "function",
        "function": {
            "name": "lookup_ens_stable_id",
            "description": "Look up an Ensembl Plants record by stable id.",
            "parameters": {
                "type": "object",
                "properties": {"stable_id": {"type": "string"}},
                "required": ["stable_id"],
            },
        },
    },
    "search_pubmed_for_gene": {
        "type": "function",
        "function": {
            "name": "search_pubmed_for_gene",
            "description": "Search PubMed literature for evidence about a gene or trait.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "gene": {"type": "string"},
                    "page_size": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    "query_community_phenotypes": {
        "type": "function",
        "function": {
            "name": "query_community_phenotypes",
            "description": "Query community-submitted phenotype observations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "species": {"type": "string"},
                    "trait_category": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    "list_plant_species": {
        "type": "function",
        "function": {
            "name": "list_plant_species",
            "description": "List known plant species, optionally filtered by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    "compare_variants": {
        "type": "function",
        "function": {
            "name": "compare_variants",
            "description": "Compare known variants for a named line.",
            "parameters": {
                "type": "object",
                "properties": {"line": {"type": "string"}},
                "required": ["line"],
            },
        },
    },
    "search_variants_in_region": {
        "type": "function",
        "function": {
            "name": "search_variants_in_region",
            "description": "Search for known variants within a genomic region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
}

DOMAIN_TOOL_NAMES: frozenset[str] = frozenset(TOOL_SCHEMAS)

# Agent-harness/meta tools observed in the raw completions (generator scaffolding,
# not a deployed farming assistant's capability) plus malformed-artifact names.
SCAFFOLDING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "terminal", "read_file", "search_files", "web_search", "mcp__fumigate_web_search",
        "memory", "todo", "skill_view", "session_search", "tool_search", "tool_describe",
        "get_prompt", "list_prompts", "list_resources", "read_resource", "tool_call",
    }
)


def _tool_names_in_row(row: dict) -> set[str]:
    names: set[str] = set()
    for turn in row.get("completion") or []:
        for call in turn.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            if name:
                names.add(name)
    return names


def is_domain_clean(row: dict, domain_tools: frozenset[str] = DOMAIN_TOOL_NAMES) -> bool:
    """True if the row's recorded completion never calls a tool outside `domain_tools`
    (rows with no tool calls at all -- farmer_dialogue/scenario_critique shapes --
    are always clean)."""
    return _tool_names_in_row(row) <= domain_tools


def render_tools_block(tools: list[dict]) -> str:
    """Byte-identical to what Qwen3's chat_template.jinja renders for the system
    turn's tools preamble when `tools` is passed to apply_chat_template -- see
    bake_tools_into_system_message() for why this is baked into text instead."""
    parts = [
        "# Tools\n\nYou may call one or more functions to assist with the user query."
        "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    ]
    for tool in tools:
        parts.append("\n" + json.dumps(tool))  # default separators (", ", ": ") match jinja's tojson filter
    parts.append(
        "\n</tools>\n\nFor each function call, return a json object with function name "
        'and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
        '<function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )
    return "".join(parts)


def bake_tools_into_system_message(
    messages: list[dict[str, str]],
    tools: list[dict] = None,
) -> list[dict[str, str]]:
    """Return a copy of `messages` with the tools preamble appended to the system
    message content, so a plain apply_chat_template(messages) call (no `tools=`
    kwarg, no per-row dataset support needed) renders the identical text the
    tools-enabled branch of chat_template.jinja would have produced."""
    if tools is None:
        tools = list(TOOL_SCHEMAS.values())
    if not messages or messages[0].get("role") != "system":
        return messages
    augmented = dict(messages[0])
    augmented["content"] = augmented["content"] + "\n\n" + render_tools_block(tools)
    return [augmented] + list(messages[1:])


def load_rollout_prompts(
    dataset_id: str = DEFAULT_DATASET_ID,
    local_path: Optional[Path] = None,
    exclude_sources: frozenset[str] = EXCLUDED_GENERATOR_AGENTS,
    hf_split: str = "train",
    holdout: str = "train",
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
    require_domain_tools_only: bool = True,
    bake_tools: bool = True,
):
    """Return a `datasets.Dataset` of rollout seed prompts, agy-excluded.

    Requires the `datasets` package (training-host only — not installed in the
    dev sandbox this file was authored in; autoresearch/grpo_reward.py and this
    module's non-`datasets` helpers above are dependency-light on purpose so
    Stage 0 validation can run without it).

    `holdout` selects the deterministic train/eval bucket (see holdout_bucket()):
    "train" for GRPOTrainer's rollout seeds, "eval" for score_grpo_primary.py's
    held-out evaluation slice, "all" to bypass the split entirely (e.g. Stage 0/1
    validation against the full pool). Same seed/fraction on both sides is what
    keeps the eval slice out of the training set.

    `require_domain_tools_only` drops rows whose recorded completion calls a
    scaffolding tool (see SCAFFOLDING_TOOL_NAMES / is_domain_clean) -- 22.4% of
    non-agy rows, confirmed on the real data. `bake_tools` appends the
    reconstructed <tools> schema block to every prompt's system message (see
    bake_tools_into_system_message) so the model actually sees tool names/args
    instead of guessing from the vague natural-language system text alone.
    """
    import datasets  # deferred: training-host-only dependency

    if local_path is not None:
        ds = datasets.Dataset.from_list(list(iter_local_jsonl(local_path)))
    else:
        ds = datasets.load_dataset(dataset_id, split=hf_split)

    ds = ds.filter(lambda row: generator_agent_of(row) not in exclude_sources)
    if holdout != "all":
        ds = ds.filter(
            lambda row: holdout_bucket(record_uuid_of(row), holdout_seed, holdout_fraction) == holdout
        )
    if require_domain_tools_only:
        ds = ds.filter(is_domain_clean)
    ds = ds.remove_columns([c for c in ("completion", "reward") if c in ds.column_names])
    if bake_tools:
        ds = ds.map(lambda row: {"prompt": bake_tools_into_system_message(row["prompt"])})
    return ds
