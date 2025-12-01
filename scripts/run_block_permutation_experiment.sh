#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/block_permutation}"
NUM_GPUS="${NUM_GPUS:-1}"

python "${ROOT_DIR}/block_permutation_experiment.py" \
  --model_name "${MODEL_NAME}" \
  --gen_length 256 \
  --block_size 4 \
  --num_problems 1 \
  --num_gpus "${NUM_GPUS}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
