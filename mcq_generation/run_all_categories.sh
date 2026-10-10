#!/usr/bin/env bash
# Phase 3a: generate MCQs for all 9 categories, full topic set (not dry-run).
# Sequential (not parallel) to stay within OpenRouter rate limits and keep logs readable.
set -uo pipefail
cd "$(dirname "$0")"

declare -A COLLECTION=(
  [permaculture]=Permaculture
  [organic_farming]=Organic_Farming
  [natural_fertilizers]=Natural_Fertilizers
  [regenerative_agriculture]=Regenerative_Agriculture
  [agroecology]=Agroecology
  [soil_science_and_microbiology]=Soil_Science_and_Microbiology
  [sustainable_water_management]=Sustainable_Water_Management
  [integrated_farming_systems]=Integrated_Farming_Systems
  [indigenous_and_traditional_knowledge]=Indigenous_and_Traditional_Knowledge
)
declare -A PREFIX=(
  [permaculture]=PC-TRAIN
  [organic_farming]=OF-TRAIN
  [natural_fertilizers]=NF-TRAIN
  [regenerative_agriculture]=RA-TRAIN
  [agroecology]=AG-TRAIN
  [soil_science_and_microbiology]=SS-TRAIN
  [sustainable_water_management]=SWM-TRAIN
  [integrated_farming_systems]=IFS-TRAIN
  [indigenous_and_traditional_knowledge]=ITK-TRAIN
)
declare -A CATLABEL=(
  [permaculture]="Permaculture Benchmark"
  [organic_farming]="Organic Farming Benchmark"
  [natural_fertilizers]="Natural Fertilizers Benchmark"
  [regenerative_agriculture]="Regenerative Agriculture Benchmark"
  [agroecology]="Agroecology Benchmark"
  [soil_science_and_microbiology]="Soil Science and Microbiology Benchmark"
  [sustainable_water_management]="Sustainable Water Management Benchmark"
  [integrated_farming_systems]="Integrated Farming Systems Benchmark"
  [indigenous_and_traditional_knowledge]="Indigenous and Traditional Knowledge Benchmark"
)

BENCHMARK="${BENCHMARK_ROOT:?set BENCHMARK_ROOT}"
mkdir -p output logs

for cat in permaculture organic_farming natural_fertilizers regenerative_agriculture \
           agroecology soil_science_and_microbiology sustainable_water_management \
           integrated_farming_systems indigenous_and_traditional_knowledge; do
  coll="${COLLECTION[$cat]}"
  prefix="${PREFIX[$cat]}"
  label="${CATLABEL[$cat]}"
  outdir="output/${cat}"
  log="logs/${cat}.log"

  echo "=== [$(date +%H:%M:%S)] $cat ($coll, prefix=$prefix) ==="

  conda run -n nf-benchmark python generate_mcqs_rag.py \
    --category "$coll" \
    --collection "$coll" \
    --output-dir "$outdir" \
    --prompt-file "prompts/prompt_for_${cat}.txt" \
    --prefix "$prefix" \
    > "$log" 2>&1
  gen_status=$?
  echo "  generate: exit $gen_status ($(grep -c '"question"' "$outdir"/*.json 2>/dev/null | awk -F: '{s+=$2} END{print s}') questions across files)"

  python3 dedup_against_benchmark.py \
    --input-dir "$outdir" \
    --benchmark "$BENCHMARK" \
    --report "output/${cat}_dedup_report.json" \
    >> "$log" 2>&1

  python3 flatten_to_jsonl.py \
    --input-dir "$outdir" \
    --category "$label" \
    --prefix "$prefix" \
    --output "output/${cat}.jsonl" \
    >> "$log" 2>&1

  conda run -n nf-benchmark python quality/llm_judge.py \
    --input "output/${cat}.jsonl" \
    --category "$cat" \
    --model anthropic/claude-haiku-4.5 \
    >> "$log" 2>&1
  echo "  done: $cat"
done

echo "=== ALL CATEGORIES COMPLETE ==="
