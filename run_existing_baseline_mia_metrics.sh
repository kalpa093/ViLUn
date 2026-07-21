#!/usr/bin/env bash
set -euo pipefail

# Evaluate already-trained baseline checkpoints with output- and
# representation-level MIA. No model is trained or unlearned by this script.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
SEED="${SEED:-42}"

SOURCE_ROOT="${SOURCE_ROOT:-${REPO_DIR}/vilun_runs}"
SAVE_DIR="${SAVE_DIR:-${SOURCE_ROOT}/saved_models}"
OUTPUT_ROOT="${BASELINE_MIA_OUTPUT_ROOT:-${SOURCE_ROOT}/baseline_mia_metrics}"
RUN_DIR="${OUTPUT_ROOT}/runs"
LOG_DIR="${OUTPUT_ROOT}/logs"
DONE_DIR="${OUTPUT_ROOT}/.done"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$DONE_DIR"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
read -r -a DATASETS <<< "${DATASETS:-cifar10 cifar100 tinyimagenet}"
read -r -a BASELINE_METHODS <<< "${BASELINE_METHODS:-original retrain gradient_ascent delete ps salun sisa}"

REPR_ATTACK_EPOCHS="${REPR_ATTACK_EPOCHS:-30}"
REPR_ATTACK_BATCH_SIZE="${REPR_ATTACK_BATCH_SIZE:-256}"
REPR_ATTACK_LR="${REPR_ATTACK_LR:-0.001}"
REPR_ATTACK_TRAIN_FRACTION="${REPR_ATTACK_TRAIN_FRACTION:-0.5}"
REPR_ATTACK_NONMEMBER_SOURCE="${REPR_ATTACK_NONMEMBER_SOURCE:-test}"
INCLUDE_GRADIENT_MIA="${INCLUDE_GRADIENT_MIA:-0}"
FORCE="${FORCE:-0}"

declare -A DS_MODEL=(
  [cifar10]=resnet18
  [cifar100]=resnet50
  [tinyimagenet]=resnet50
)
declare -A GPU_PID
declare -A GPU_TAG
for gpu in "${GPUS[@]}"; do
  GPU_PID[$gpu]=""
  GPU_TAG[$gpu]=""
done

FREE_GPU=""
JOB_FAILED=0

find_free_gpu() {
  while true; do
    for gpu in "${GPUS[@]}"; do
      local pid="${GPU_PID[$gpu]}"
      if [ -z "$pid" ]; then
        FREE_GPU="$gpu"
        return
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        if ! wait "$pid"; then
          echo "[ERROR] GPU${gpu} job failed: ${GPU_TAG[$gpu]}" >&2
          JOB_FAILED=1
        fi
        GPU_PID[$gpu]=""
        GPU_TAG[$gpu]=""
        FREE_GPU="$gpu"
        return
      fi
    done
    sleep 3
  done
}

submit_once() {
  local tag="$1"
  local history_dir="$2"
  shift 2
  local done_file="${DONE_DIR}/${tag}.done"
  if [ "$FORCE" = "1" ]; then
    rm -f "$done_file"
  elif [ -f "$done_file" ]; then
    echo "[SKIP] ${tag}"
    return
  fi

  find_free_gpu
  local gpu="$FREE_GPU"
  local logfile="${LOG_DIR}/${tag}.log"
  mkdir -p "$history_dir"
  rm -f "${history_dir}/summary_mia.csv"
  echo "[$(date '+%H:%M:%S')] START GPU${gpu} ${tag}"
  (
    set -e
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$@" 2>&1 |
      sed --unbuffered "s/^/[GPU${gpu}|${tag}] /" | tee "$logfile"
    touch "$done_file"
  ) &
  GPU_PID[$gpu]=$!
  GPU_TAG[$gpu]="$tag"
}

wait_all() {
  for gpu in "${GPUS[@]}"; do
    local pid="${GPU_PID[$gpu]}"
    if [ -n "$pid" ] && ! wait "$pid"; then
      echo "[ERROR] GPU${gpu} job failed: ${GPU_TAG[$gpu]}" >&2
      JOB_FAILED=1
    fi
    GPU_PID[$gpu]=""
    GPU_TAG[$gpu]=""
  done
}

resolve_checkpoint() {
  local method="$1" dataset="$2" model="$3"
  local candidates=()
  case "$method" in
    original|retrain|gradient_ascent|delete)
      candidates=(
        "${SAVE_DIR}/${method}_${dataset}_${model}_seed${SEED}.pth"
      )
      ;;
    ps)
      candidates=(
        "${SOURCE_ROOT}/ps_models/ps_${dataset}_${model}_seed${SEED}.pth"
        "${SAVE_DIR}/ps_${dataset}_${model}_seed${SEED}.pth"
      )
      ;;
    salun)
      candidates=(
        "${SOURCE_ROOT}/salun_models/salun_${dataset}_${model}_seed${SEED}.pth"
        "${SAVE_DIR}/salun_${dataset}_${model}_seed${SEED}.pth"
      )
      ;;
    sisa)
      candidates=(
        "${SOURCE_ROOT}/sisa_models/containers/${dataset}_${model}_${SEED}"
        "${SAVE_DIR}/containers/${dataset}_${model}_${SEED}"
      )
      ;;
    *)
      candidates=(
        "${SAVE_DIR}/${method}_${dataset}_${model}_seed${SEED}.pth"
      )
      ;;
  esac

  local candidate
  for candidate in "${candidates[@]}"; do
    if { [ "$method" = "sisa" ] && [ -d "$candidate" ]; } ||
       { [ "$method" != "sisa" ] && [ -f "$candidate" ]; }; then
      echo "$candidate"
      return
    fi
  done
  echo "${candidates[0]}"
}

resolve_forget_indices() {
  local method="$1" dataset="$2" model="$3"
  local method_token="$method"
  [ "$method" = "gradient_ascent" ] && method_token="gradient_ascent"

  local candidates=(
    "${SOURCE_ROOT}/history/forget_indices_${method_token}_${dataset}_${model}_seed${SEED}.pt"
    "${SOURCE_ROOT}/history/forget_indices_${dataset}_${model}_seed${SEED}.pt"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}

run_one() {
  local method="$1" dataset="$2" model="$3"
  local checkpoint
  checkpoint="$(resolve_checkpoint "$method" "$dataset" "$model")"
  local forget_path
  forget_path="$(resolve_forget_indices "$method" "$dataset" "$model")"
  local history_dir="${RUN_DIR}/${method}_${dataset}_${model}_seed${SEED}"
  local tag="mia_${method}_${dataset}_${model}_seed${SEED}"

  if [ -z "$forget_path" ]; then
    echo "[SKIP] Missing forget indices: ${method}/${dataset}/${model}"
    return
  fi

  local command=(
    "$PYTHON_BIN" "${REPO_DIR}/evaluate_mia.py"
    --method "$method" --dataset "$dataset" --model "$model"
    --data_dir "$DATA_DIR" --history_dir "$history_dir"
    --forget_indices_path "$forget_path" --seed "$SEED"
    --representation_mia
    --repr_attack_epochs "$REPR_ATTACK_EPOCHS"
    --repr_attack_batch_size "$REPR_ATTACK_BATCH_SIZE"
    --repr_attack_lr "$REPR_ATTACK_LR"
    --repr_attack_train_fraction "$REPR_ATTACK_TRAIN_FRACTION"
    --repr_attack_nonmember_source "$REPR_ATTACK_NONMEMBER_SOURCE"
  )

  if [ "$method" = "sisa" ]; then
    if [ ! -d "$checkpoint" ]; then
      echo "[SKIP] Missing SISA container: ${checkpoint}"
      return
    fi
    command+=(--sisa_dir "$(dirname "$checkpoint")")
  else
    if [ ! -f "$checkpoint" ]; then
      echo "[SKIP] Missing checkpoint: ${checkpoint}"
      return
    fi
    command+=(--model_path "$checkpoint")
    if [ "$INCLUDE_GRADIENT_MIA" = "1" ]; then
      command+=(--gradient_mia)
    fi
  fi

  echo "[FOUND] ${method}/${dataset}: ${checkpoint}"
  echo "        forget indices: ${forget_path}"
  submit_once "$tag" "$history_dir" "${command[@]}"
}

echo "============================================================"
echo "Existing baseline output + representation MIA"
echo "repo       : ${REPO_DIR}"
echo "source     : ${SOURCE_ROOT}"
echo "data       : ${DATA_DIR}"
echo "output     : ${OUTPUT_ROOT}"
echo "datasets   : ${DATASETS[*]}"
echo "methods    : ${BASELINE_METHODS[*]}"
echo "GPUs       : ${GPUS[*]}"
echo "============================================================"

for dataset in "${DATASETS[@]}"; do
  if [ -z "${DS_MODEL[$dataset]+x}" ]; then
    echo "[ERROR] Unsupported dataset: ${dataset}" >&2
    exit 2
  fi
  model="${DS_MODEL[$dataset]}"
  for method in "${BASELINE_METHODS[@]}"; do
    run_one "$method" "$dataset" "$model"
  done
done

wait_all

"$PYTHON_BIN" - "$RUN_DIR" "${OUTPUT_ROOT}/summary_baseline_mia_metrics.csv" <<'PY'
import csv
import os
import sys

run_dir, output_path = sys.argv[1:]
rows = []
fieldnames = []
for root, _, files in os.walk(run_dir):
    if "summary_mia.csv" not in files:
        continue
    path = os.path.join(root, "summary_mia.csv")
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            row["Result_Path"] = path
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

rows.sort(key=lambda row: (row.get("Method", ""), row.get("Dataset", ""), row.get("Model", "")))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"[SAVE] Combined {len(rows)} rows -> {output_path}")
PY

if [ "$JOB_FAILED" -ne 0 ]; then
  echo "[ERROR] One or more MIA jobs failed. Check ${LOG_DIR}." >&2
  exit 1
fi

echo "[DONE] Existing baseline MIA evaluation completed."
