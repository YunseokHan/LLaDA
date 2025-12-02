#!/bin/bash

set -euo pipefail

# Example helper to visualize successful permutations.
# Customize via env vars:
#   RESULT_DIR=/path/to/results_n100
#   METHOD=semi_ar        # or margin/conv/halton/random/confidence
#   SEMI_AR_BLOCK=32
#   NUM_ROW=4
#   SEED=42
#   HALTON_METHOD_LABEL="halton (steps=64, rand)"  # optional override when METHOD=halton
#   METHOD_FILE_FILTERS="confidence=_of04"         # semicolon-delimited list -> --method_file_filter
#   METHOD_FILE_EXCLUDES="confidence=_of08"        # semicolon-delimited list -> --method_file_exclude

RESULT_DIR="${RESULT_DIR:-results_n100}"
IN_DIR="${IN_DIR:-${RESULT_DIR}/raw}"
OUT_DIR="${OUT_DIR:-${RESULT_DIR}/final}"
NUM_ROW="${NUM_ROW:-4}"
SEED="${SEED:-1234}"
METHOD="${METHOD:-semi_ar}"
SEMI_AR_BLOCK="${SEMI_AR_BLOCK:-16}"
HALTON_METHOD_LABEL="${HALTON_METHOD_LABEL:-}"
METHOD_FILE_FILTERS="${METHOD_FILE_FILTERS:-}"
METHOD_FILE_EXCLUDES="${METHOD_FILE_EXCLUDES:-}"

CMD=(python3 perm_visualize.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR" --num_row "$NUM_ROW" --seed "$SEED")

if [[ -n "${METHOD}" ]]; then
  method_arg="$METHOD"
  if [[ "$METHOD" == "halton" && -n "$HALTON_METHOD_LABEL" ]]; then
    method_arg="$HALTON_METHOD_LABEL"
  fi
  CMD+=(--method "$method_arg")
  if [[ "$METHOD" == "semi_ar" && -n "${SEMI_AR_BLOCK}" ]]; then
    CMD+=(--semi_ar_block_size "$SEMI_AR_BLOCK")
  fi
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
