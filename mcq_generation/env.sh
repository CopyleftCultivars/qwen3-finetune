#!/usr/bin/env bash
# env.sh — Workspace shell helpers for the [LOCAL_FOLDER] benchmark pipeline.
#
# Source this file in your shell session (or add to ~/.bashrc):
#   source mcq_generation/env.sh

# ---------------------------------------------------------------------------
# next_round CATEGORY
#   Returns the next regeneration round number for a category.
#   Inspects existing regenerated_mcqs/{CATEGORY}_round* directories and
#   returns max(existing) + 1.  Defaults to 2 when no rounds exist yet.
#
# Examples:
#   ROUND=$(next_round natural_fertilizers)   # → 3 if round2 exists
#   ROUND=$(next_round agroecology)           # → 2 if no rounds yet
# ---------------------------------------------------------------------------
next_round() {
  local category="$1"
  local workspace
  workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || pwd)"
  local last
  last=$(ls -d "${workspace}/regenerated_mcqs/${category}_round"* 2>/dev/null \
    | grep -o 'round[0-9]*' | grep -o '[0-9]*' | sort -n | tail -1)
  echo $(( ${last:-1} + 1 ))
}

# Convenience: show current round status for a category
regen_status() {
  local category="${1:-natural_fertilizers}"
  local workspace
  workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || pwd)"
  local round
  round=$(next_round "$category")
  echo "Category: $category"
  echo "Next round: $round  (current = $((round - 1)))"
  local stats="${workspace}/regenerated_mcqs/${category}_round$((round-1))/round_stats.json"
  if [ -f "$stats" ]; then
    python3 -c "
import json, sys
d = json.load(open('$stats'))
print('  acceptance_rate:', d.get('acceptance_rate'))
print('  topics_filled:  ', d.get('topics_filled', []))
print('  topics_remaining:', d.get('topics_remaining', []))
print('  errors:         ', d.get('errors', 0))
"
  fi
}

export -f next_round
export -f regen_status
