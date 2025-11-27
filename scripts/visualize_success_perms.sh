#!/bin/bash

set -euo pipefail

# Example helper to visualize successful permutations.
# Customize via env vars:
#   RESULT_DIR=/path/to/results_n100
#   METHOD=semi_ar
#   SEMI_AR_BLOCK=32
#   NUM_ROW=4
#   SEED=42

RESULT_DIR="${RESULT_DIR:-results_n100}"
IN_DIR="${IN_DIR:-${RESULT_DIR}/raw}"
OUT_DIR="${OUT_DIR:-${RESULT_DIR}/final}"
NUM_ROW="${NUM_ROW:-4}"
SEED="${SEED:-1234}"
METHOD="${METHOD:-semi_ar}"
SEMI_AR_BLOCK="${SEMI_AR_BLOCK:-16}"

CMD=(python3 perm_visualize.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR" --num_row "$NUM_ROW" --seed "$SEED")

if [[ -n "${METHOD}" ]]; then
  CMD+=(--method "$METHOD")
  if [[ "$METHOD" == "semi_ar" && -n "${SEMI_AR_BLOCK}" ]]; then
    CMD+=(--semi_ar_block_size "$SEMI_AR_BLOCK")
  fi
fi

echo "== Running: ${CMD[*]} =="
"${CMD[@]}"
