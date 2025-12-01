#!/bin/bash

set -euo pipefail

LOG_DIR="${LOG_DIR:-logs/search_n100}"
RESULT_DIR="${RESULT_DIR:-results_n100}"
RAW_DIR="${RESULT_DIR}/raw"
FINAL_DIR="${RESULT_DIR}/final"

mkdir -p "$LOG_DIR" "$RAW_DIR" "$FINAL_DIR"

# ==== Weights & Biases (recommended) ====
export WANDB__SERVICE_WAIT=300
export WANDB_CONSOLE=off
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-llada-passk}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"

PY=python3
SCRIPT=test_ordering.py
MODEL="GSAI-ML/LLaDA-8B-Instruct"
JOB_ID="${JOB_ID:-manual-$(date +%Y%m%d_%H%M%S)}"

# ===== Config =====
N_PROBLEMS=100
TRIALS=1024
GEN_LEN=256
FORCE_K1="--force_k1"
BATCH=4
CFG=0.0
SEED=1234
OUT_DIR="$RAW_DIR"
SEMI_AR_BLOCKS="${SEMI_AR_BLOCKS:-8}"  # space-separated list
HALTON_EXTRA_ARGS="${HALTON_EXTRA_ARGS:-}"  # extra CLI args for halton method

SAFE_MODEL="${MODEL//\//_}"
WANDB_GROUP="${WANDB_GROUP:-passk-${SAFE_MODEL}}"

declare -a METHOD_ARGS=()
GPU_ID_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo "[ERR] --gpus requires a comma-separated list (e.g., --gpus 4,6)." >&2
        exit 1
      fi
      GPU_ID_ARG="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/test_ordering.sh [--gpus ID0,ID1,...] [method ...]
  methods: any of ar, random, semi_ar, confidence, margin, halton (default: semi_ar confidence)
  For halton, you can supply HALTON_EXTRA_ARGS="--halton_steps 80 --halton_randomize" to tweak decoding.
USAGE
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      METHOD_ARGS+=("$1")
      shift
      ;;
  esac
done

if (( ${#METHOD_ARGS[@]} == 0 )); then
  METHOD_ARGS=(semi_ar confidence)
fi

MAX_LOCAL_GPUS="${MAX_LOCAL_GPUS:-2}"
GPU_ID_LIST="${GPU_ID_ARG:-${GPU_IDS:-${GPU_ID_LIST:-}}}"
declare -a GPU_DEVICES=()
if [[ -n "$GPU_ID_LIST" ]]; then
  IFS=',' read -ra GPU_DEVICES <<< "$GPU_ID_LIST"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
fi
if (( ${#GPU_DEVICES[@]} == 0 )); then
  for ((i=0; i<MAX_LOCAL_GPUS; i++)); do
    GPU_DEVICES+=("$i")
  done
fi
if (( ${#GPU_DEVICES[@]} > MAX_LOCAL_GPUS )); then
  GPU_DEVICES=("${GPU_DEVICES[@]:0:MAX_LOCAL_GPUS}")
fi
TOTAL_GPUS=${#GPU_DEVICES[@]}
if (( TOTAL_GPUS == 0 )); then
  echo "[ERR] Unable to determine GPUs to use. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
echo "== Local GPU workers (${TOTAL_GPUS}): ${GPU_DEVICES[*]} =="

run_method_sharded () {
  local method="$1"
  shift
  local extra_args=("$@")
  local upper_method
  upper_method=$(echo "$method" | tr '[:lower:]' '[:upper:]')
  echo "== Running ${upper_method} (${TOTAL_GPUS} shards) =="
  local -a cmd=(
    "$PY" "$SCRIPT" worker
    --method "$method" --model_name "$MODEL"
    --n "$N_PROBLEMS" --gen_length "$GEN_LEN"
    --batch_size "$BATCH" --seed "$SEED" --device cuda
    --trials "$TRIALS" --num_shards "$TOTAL_GPUS"
    --out_dir "$OUT_DIR"
    --wandb_mode "$WANDB_MODE"
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "$WANDB_GROUP"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    cmd+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$FORCE_K1" ]]; then
    cmd+=("$FORCE_K1")
  fi
  if (( ${#extra_args[@]} )); then
    cmd+=("${extra_args[@]}")
  fi

  local method_tag="$method"
  if (( ${#extra_args[@]} > 0 )); then
    local idx=0
    while (( idx < ${#extra_args[@]} )); do
      if [[ "${extra_args[idx]}" == "--semi_ar_block_size" && $((idx + 1)) -lt ${#extra_args[@]} ]]; then
        method_tag="${method_tag}_bs${extra_args[idx+1]}"
        break
      fi
      idx=$((idx + 1))
    done
  fi

  local log_prefix="${LOG_DIR}/${JOB_ID}.${method_tag}.shard"

  local -a pids=()
  local shard_idx=0
  for gpu_id in "${GPU_DEVICES[@]}"; do
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      shard_cmd=( "${cmd[@]}" )
      shard_cmd+=( --trial_shard "$shard_idx" )
      log_base="${log_prefix}${shard_idx}"
      "${shard_cmd[@]}" >"${log_base}.out" 2>"${log_base}.err"
    ) &
    pids+=("$!")
    shard_idx=$((shard_idx + 1))
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "[ERR] ${method} shards failed. Check logs in ${LOG_DIR}." >&2
    exit 1
  fi
}

for raw_method in "${METHOD_ARGS[@]}"; do
  method=$(echo "$raw_method" | tr '[:upper:]' '[:lower:]')
  case "$method" in
    ar)
      run_method_sharded ar
      ;;
    random)
      run_method_sharded random --save_success_perms --save_failed_perms
      ;;
    semi_ar)
      for block in $SEMI_AR_BLOCKS; do
        run_method_sharded semi_ar --semi_ar_block_size "$block" --save_success_perms --save_failed_perms
      done
      ;;
    confidence)
      run_method_sharded confidence --save_success_perms --save_failed_perms
      ;;
    margin)
      run_method_sharded margin --save_success_perms --save_failed_perms
      ;;
    halton)
      halton_extra=()
      if [[ -n "$HALTON_EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        halton_extra=($HALTON_EXTRA_ARGS)
      fi
      run_method_sharded halton --save_success_perms --save_failed_perms "${halton_extra[@]}"
      ;;
    *)
      echo "[ERR] Unknown method '$raw_method'. Supported: ar, random, semi_ar, confidence, margin, halton." >&2
      exit 1
      ;;
  esac
done

# ================================
# Aggregate
# ================================
echo "== Aggregate =="
$PY $SCRIPT aggregate \
  --in_dir "$OUT_DIR" --out_dir "$FINAL_DIR" \
  --trials "$TRIALS" --n "$N_PROBLEMS" --model_name "$MODEL" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  $( [[ -n "$WANDB_ENTITY" ]] && echo --wandb_entity "$WANDB_ENTITY" ) \
  --wandb_group "$WANDB_GROUP" \
  --wandb_run_name "aggregate-${JOB_ID}_n${N_PROBLEMS}"

echo "== Outputs =="
echo "  - Shard CSVs: $OUT_DIR/trials_*.csv"
echo "  - success perms: $OUT_DIR/success_perms_*.jsonl"
echo "  - pass@k summary: $FINAL_DIR/passk_summary.csv"
echo "  - plot: $FINAL_DIR/passk_plot.pdf / .png"
