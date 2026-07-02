"""
prepare_dataset.py

Merges 6 heterogeneous KNF/natural-farming datasets into a single
merged_sharegpt.jsonl file ready for Unsloth/TRL SFT training.

Output format (ShareGPT):
  {"conversations": [{"from": "system"|"human"|"gpt", "value": "..."}, ...]}

Merge order:
  1. Multi-turn conversations (deduped from conversations/ + conversations (1)/)
  2. Natural-Farming-Real-QandA (36,919 rows)
  3. KNF-Methods-QnA (100 rows)
  4. Semisynthetic_Data (48 rows)
  5. syntheticdata-distiset (30 rows, Parquet)

Run: python prepare_dataset.py
"""

import glob
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
CONV_DIR_A = BASE / "conversations"
CONV_DIR_B = BASE / "conversations (1)"
NF_QA_CSV  = BASE / "Natural-Farming-Real-QandA-Conversations-Q1-2024-Update" / "Trainable_NF_conversations_Q1_2024_PLUS.csv"
KNF_CSV    = BASE / "KNF-Methods-QnA" / "KNF-methods-dataset1.csv"
SEMI_JSON  = BASE / "Semisynthetic_Data_Natural_Farming_Fundamentals" / "Semisynthetic_Data_Natural_Farming_Fundamentals.json"
SYNTH_PARQ = BASE / "syntheticdata-distiset-farming-chemistry" / "data" / "train-00000-of-00001.parquet"
SHORT_SYS  = BASE / "system_prompt_short.txt"
OUTPUT     = BASE / "merged_sharegpt.jsonl"

# ---------------------------------------------------------------------------
# Load shortened system prompt (required for multi-turn conversations)
# ---------------------------------------------------------------------------
if not SHORT_SYS.exists():
    print(f"ERROR: {SHORT_SYS} not found. Create it before running this script.")
    sys.exit(1)

SHORT_SYSTEM = SHORT_SYS.read_text(encoding="utf-8").strip()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_record(turns: list[dict]) -> dict:
    """Wrap a list of turn dicts into a ShareGPT record."""
    return {"conversations": turns}


def is_empty(text) -> bool:
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    # Skip placeholder non-answers
    lower = stripped.lower()
    skip_phrases = ["i don't know", "i do not know", "n/a", "none", "unknown"]
    if all(p in lower for p in ["i don't know"]) and len(stripped) < 60:
        return True
    return False


# ---------------------------------------------------------------------------
# Gemma4 conversation normalizer
# ---------------------------------------------------------------------------

def normalize_for_gemma4(turns: list[dict]) -> list[dict] | None:
    """
    Gemma4's chat template requires strict human/gpt alternation with no
    system turns. Applies three fixes at data-prep time so the JSONL on
    HuggingFace is already clean:
      1. Merge system content into the first human turn
      2. Merge consecutive same-role turns (joined with a blank line)
      3. Strip leading gpt turns and trailing human turns
    Returns None if the result has fewer than 2 turns (record is dropped).
    """
    if not turns:
        return None

    # 1. Collect system content and strip system turns
    system_prefix = ""
    non_system = []
    for t in turns:
        if t["from"] == "system":
            system_prefix += t["value"].strip() + "\n\n"
        else:
            non_system.append(dict(t))

    if system_prefix:
        for t in non_system:
            if t["from"] == "human":
                t["value"] = system_prefix.rstrip() + "\n\n" + t["value"]
                break

    # 2. Merge consecutive same-role turns
    merged = []
    for t in non_system:
        if merged and merged[-1]["from"] == t["from"]:
            merged[-1]["value"] += "\n\n" + t["value"]
        else:
            merged.append(t)

    # 3. Must start with human, end with gpt
    while merged and merged[0]["from"] != "human":
        merged.pop(0)
    while merged and merged[-1]["from"] != "gpt":
        merged.pop()

    if len(merged) < 2:
        return None
    return merged


# ---------------------------------------------------------------------------
# Phase 1 — Multi-turn conversation CSVs
# ---------------------------------------------------------------------------

def load_conversation_csvs() -> pd.DataFrame:
    """Load and concatenate all CSVs from both conversation directories."""
    csv_files = (
        glob.glob(str(CONV_DIR_A / "*.csv")) +
        glob.glob(str(CONV_DIR_B / "*.csv"))
    )
    if not csv_files:
        print("WARNING: No conversation CSVs found.")
        return pd.DataFrame()

    print(f"Loading {len(csv_files):,} conversation CSV files...")
    chunks = []
    for f in tqdm(csv_files, desc="Reading CSVs", unit="file"):
        try:
            df = pd.read_csv(f, dtype=str)
            chunks.append(df)
        except Exception as e:
            print(f"  Skipped {f}: {e}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Rows before dedup: {len(df):,}")
    df = df.drop_duplicates(subset="message_id")
    print(f"  Rows after dedup on message_id: {len(df):,}")
    return df


def build_conversation_records(df: pd.DataFrame) -> list[dict]:
    """Group by conversation_id, sort by createdAt, build ShareGPT records."""
    if df.empty:
        return []

    # Parse timestamps for sorting; keep as string fallback if parse fails
    try:
        df["_ts"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    except Exception:
        df["_ts"] = pd.NaT

    records = []
    skipped_short = 0
    skipped_empty = 0
    skipped_no_alternate = 0

    grouped = df.groupby("conversation_id")
    for conv_id, group in tqdm(grouped, desc="Building conversations", unit="conv"):
        group = group.sort_values("_ts", na_position="last")

        turns = []
        for _, row in group.iterrows():
            role = "human" if str(row.get("from", "")).strip().lower() == "human" else "gpt"
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            turns.append({"from": role, "value": text})

        if len(turns) < 2:
            skipped_short += 1
            continue

        # Check that conversation has at least one human and one gpt turn
        roles_present = {t["from"] for t in turns}
        if "human" not in roles_present or "gpt" not in roles_present:
            skipped_no_alternate += 1
            continue

        # Inject shortened system prompt as first turn
        full_turns = [{"from": "system", "value": SHORT_SYSTEM}] + turns
        records.append(make_record(full_turns))

    print(f"  Conversation records built: {len(records):,}")
    print(f"  Skipped (< 2 turns): {skipped_short:,}  |  no alternation: {skipped_no_alternate:,}")
    return records


# ---------------------------------------------------------------------------
# Phase 2 — Natural-Farming-Real-QandA
# ---------------------------------------------------------------------------

def load_nf_qa() -> list[dict]:
    print(f"\nLoading Natural-Farming-Real-QandA from {NF_QA_CSV.name}...")
    df = pd.read_csv(NF_QA_CSV, dtype=str)
    print(f"  Rows loaded: {len(df):,}")

    records = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="NF-QA", unit="row"):
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        if is_empty(q) or is_empty(a):
            skipped += 1
            continue
        records.append(make_record([
            {"from": "human", "value": q},
            {"from": "gpt",   "value": a},
        ]))

    print(f"  Records: {len(records):,}  |  Skipped (empty/no-answer): {skipped:,}")
    return records


# ---------------------------------------------------------------------------
# Phase 3 — KNF-Methods-QnA
# ---------------------------------------------------------------------------

def load_knf_methods() -> list[dict]:
    print(f"\nLoading KNF-Methods-QnA from {KNF_CSV.name}...")
    df = pd.read_csv(KNF_CSV, dtype=str)
    print(f"  Rows loaded: {len(df):,}")

    records = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="KNF-Methods", unit="row"):
        q = str(row.get("Question", "")).strip()
        a = str(row.get("Answer", "")).strip()
        phil = str(row.get("Philosophical Element", "")).strip()

        if is_empty(q) or is_empty(a):
            skipped += 1
            continue

        turns = []
        if phil and not is_empty(phil):
            turns.append({"from": "system", "value": phil})
        turns.append({"from": "human", "value": q})
        turns.append({"from": "gpt",   "value": a})
        records.append(make_record(turns))

    print(f"  Records: {len(records):,}  |  Skipped: {skipped:,}")
    return records


# ---------------------------------------------------------------------------
# Phase 4 — Semisynthetic_Data
# ---------------------------------------------------------------------------

def load_semisynthetic() -> list[dict]:
    print(f"\nLoading Semisynthetic_Data from {SEMI_JSON.name}...")
    with open(SEMI_JSON, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Objects loaded: {len(data):,}")

    records = []
    skipped = 0
    for obj in tqdm(data, desc="Semisynthetic", unit="obj"):
        q = str(obj.get("Question", "")).strip()
        a = str(obj.get("Answer", "")).strip()
        if is_empty(q) or is_empty(a):
            skipped += 1
            continue
        records.append(make_record([
            {"from": "human", "value": q},
            {"from": "gpt",   "value": a},
        ]))

    print(f"  Records: {len(records):,}  |  Skipped: {skipped:,}")
    return records


# ---------------------------------------------------------------------------
# Phase 5 — syntheticdata-distiset (Parquet)
# ---------------------------------------------------------------------------

def load_synth_distiset() -> list[dict]:
    print(f"\nLoading syntheticdata-distiset from {SYNTH_PARQ.name}...")
    df = pd.read_parquet(SYNTH_PARQ)
    print(f"  Rows loaded: {len(df):,}")

    records = []
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Distiset", unit="row"):
        sys_p  = str(row.get("system_prompt", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        compl  = str(row.get("completion", "")).strip()

        if is_empty(prompt) or is_empty(compl):
            skipped += 1
            continue

        turns = []
        if sys_p and not is_empty(sys_p):
            turns.append({"from": "system", "value": sys_p})
        turns.append({"from": "human", "value": prompt})
        turns.append({"from": "gpt",   "value": compl})
        records.append(make_record(turns))

    print(f"  Records: {len(records):,}  |  Skipped: {skipped:,}")
    return records


# ---------------------------------------------------------------------------
# Verification — print 5 random samples
# ---------------------------------------------------------------------------

def verify(records: list[dict]) -> None:
    print("\n--- Verification: 5 random samples ---")
    samples = random.sample(records, min(5, len(records)))
    for i, rec in enumerate(samples, 1):
        convs = rec["conversations"]
        roles = [t["from"] for t in convs]
        preview = convs[-1]["value"][:120].replace("\n", " ")
        print(f"  [{i}] turns={len(convs)} roles={roles}  last='{preview}'")
    print("--------------------------------------")

    # Schema check — no system turns, strict alternation
    for rec in records:
        assert "conversations" in rec, "Missing 'conversations' key"
        turns = rec["conversations"]
        for turn in turns:
            assert "from" in turn and "value" in turn, f"Bad turn: {turn}"
            assert turn["from"] in ("human", "gpt"), f"Unexpected role after normalization: {turn['from']}"
        roles = [t["from"] for t in turns]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1], f"Consecutive same-role turns at index {i}: {roles}"
    print("Schema check PASSED.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_records = []

    # 1. Multi-turn conversations
    df_conv = load_conversation_csvs()
    conv_records = build_conversation_records(df_conv)
    all_records.extend(conv_records)

    # 2. Natural-Farming-Real-QandA
    all_records.extend(load_nf_qa())

    # 3. KNF-Methods-QnA
    all_records.extend(load_knf_methods())

    # 4. Semisynthetic_Data
    all_records.extend(load_semisynthetic())

    # 5. syntheticdata-distiset
    all_records.extend(load_synth_distiset())

    print(f"\nTotal records before normalization: {len(all_records):,}")

    # Normalize all records for Gemma4's strict alternation requirement
    normalized = []
    dropped = 0
    for rec in all_records:
        turns = normalize_for_gemma4(rec["conversations"])
        if turns is None:
            dropped += 1
        else:
            normalized.append({"conversations": turns})
    all_records = normalized
    print(f"Dropped (unfixable after normalization): {dropped:,}")
    print(f"Records after normalization: {len(all_records):,}")

    verify(all_records)

    print(f"Writing {OUTPUT} ...")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. {OUTPUT} written with {len(all_records):,} records.")


if __name__ == "__main__":
    main()
