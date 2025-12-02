#!/bin/bash

set -euo pipefail

# Env knobs:
#   METHODS="confidence semi_ar"             # method labels to analyze
#   METHOD_FILE_FILTERS="confidence=_of04"   # semicolon-delimited filters passed to --method_file_filter
#   METHOD_FILE_EXCLUDES="confidence=_of08"  # semicolon-delimited excludes passed to --method_file_exclude
RESULT_DIR="${RESULT_DIR:-results_n100}"
RAW_DIR="${RAW_DIR:-${RESULT_DIR}/raw}"
FINAL_DIR="${FINAL_DIR:-${RESULT_DIR}/final}"
METHODS="${METHODS:-confidence semi_ar margin conv halton}"
INCLUDE_RANDOM="${INCLUDE_RANDOM:-1}"
if [[ "$INCLUDE_RANDOM" == "1" && "$METHODS" != *"random"* ]]; then
  METHODS="${METHODS} random"
fi
SEMI_AR_BLOCK="${SEMI_AR_BLOCK:-16}"
CSV_NAME="${CSV_NAME:-perm_ar_similarity_metrics.csv}"
HIST_BINS="${HIST_BINS:-40}"
METHOD_FILE_FILTERS="${METHOD_FILE_FILTERS:-}"
METHOD_FILE_EXCLUDES="${METHOD_FILE_EXCLUDES:-}"

read -r -a METHOD_ARR <<< "$METHODS"

CMD=(python3 perm_ar_similarity.py --in_dir "$RAW_DIR" --raw_out_dir "$RAW_DIR" --final_out_dir "$FINAL_DIR" --csv_name "$CSV_NAME" --hist_bins "$HIST_BINS")

if (( ${#METHOD_ARR[@]} > 0 )); then
  CMD+=(--methods)
  CMD+=("${METHOD_ARR[@]}")
fi

if [[ "$METHODS" == *"semi_ar"* ]]; then
  CMD+=(--semi_ar_block_size "$SEMI_AR_BLOCK")
fi

if [[ -n "$METHOD_FILE_FILTERS" ]]; then
  IFS=';' read -ra filters <<< "$METHOD_FILE_FILTERS"
  for entry in "${filters[@]}"; do
    [[ -z "$entry" ]] && continue
    CMD+=(--method_file_filter "$entry")
  done
fi
if [[ -n "$METHOD_FILE_EXCLUDES" ]]; then
  IFS=';' read -ra excludes <<< "$METHOD_FILE_EXCLUDES"
  for entry in "${excludes[@]}"; do
    [[ -z "$entry" ]] && continue
    CMD+=(--method_file_exclude "$entry")
  done
fi

echo "== Running: ${CMD[*]} =="
"${CMD[@]}"
