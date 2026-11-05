#!/usr/bin/env bash
# Link the training data that must not live in git into this checkout.
#
#   scripts/link_local_training_data.sh
#
# The conversation logs behind the training sets, the sets derived from them, and a few result files that quote them are kept in a
# folder outside every repository (default: ../training-data/qwen3-finetune next to this checkout, override with
# QWEN3_TRAINING_DATA_DIR). data/local_paths.txt lists the paths; each one is symlinked into place, and .gitignore keeps them out of git.
# Existing real files or folders are never replaced.
set -euo pipefail
cd "$(dirname "$0")/.."
data_dir="${QWEN3_TRAINING_DATA_DIR:-$(cd .. && pwd)/training-data/qwen3-finetune}"
[ -d "$data_dir" ] || { echo "training data folder not found: $data_dir (set QWEN3_TRAINING_DATA_DIR)" >&2; exit 1; }
linked=0 present=0 missing=0
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  if [ -e "$rel" ] || [ -L "$rel" ]; then present=$((present + 1)); continue; fi
  if [ ! -e "$data_dir/$rel" ]; then echo "not in data folder: $rel" >&2; missing=$((missing + 1)); continue; fi
  mkdir -p "$(dirname "$rel")"
  ln -s "$data_dir/$rel" "$rel"
  linked=$((linked + 1))
done < data/local_paths.txt
echo "linked $linked, already present $present, missing from data folder $missing"
