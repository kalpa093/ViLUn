#!/usr/bin/env bash
set -euo pipefail

# Evaluate the six existing ViLUn/ViLUn_f checkpoints with output- and
# representation-level MIA. This script does not train or unlearn models.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_DIR}/vilun_runs}"
SAVE_DIR="${SAVE_DIR:-${SOURCE_ROOT}/saved_models}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
SEED="${SEED:-42}"

OUTPUT_ROOT="${UNLEARNED_MIA_OUTPUT_ROOT:-${SOURCE_ROOT}/unlearned_mia_metrics}"
RUN_DIR="${OUTPUT_ROOT}/runs"
LOG_DIR="${OUTPUT_ROOT}/logs"
DONE_DIR="${OUTPUT_ROOT}/.done"
SUMMARY_PATH="${OUTPUT_ROOT}/summary_unlearned_mia_metrics.csv"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${SOURCE_ROOT}/baseline_mia_metrics/summary_baseline_mia_metrics.csv}"
MERGED_SUMMARY="${MERGED_SUMMARY:-${OUTPUT_ROOT}/summary_baseline_with_vilun_mia_metrics.csv}"
mkdir -p "$RUN_DIR" "$LOG_DIR" "$DONE_DIR"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
read -r -a DATASETS <<< "${DATASETS:-cifar10 cifar100 tinyimagenet}"
read -r -a METHODS <<< "${METHODS:-vilun vilun_f}"

REPR_ATTACK_EPOCHS="${REPR_ATTACK_EPOCHS:-30}"
REPR_ATTACK_BATCH_SIZE="${REPR_ATTACK_BATCH_SIZE:-256}"
REPR_ATTACK_LR="${REPR_ATTACK_LR:-0.001}"
REPR_ATTACK_TRAIN_FRACTION="${REPR_ATTACK_TRAIN_FRACTION:-0.5}"
REPR_ATTACK_NONMEMBER_SOURCE="${REPR_ATTACK_NONMEMBER_SOURCE:-test}"
INCLUDE_GRADIENT_MIA="${INCLUDE_GRADIENT_MIA:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
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

checkpoint_path() {
  local method="$1" dataset="$2" model="$3"
  local alpha_token="${DEFAULT_ALPHA:-1}"
  local beta_token="${DEFAULT_BETA:-2}"
  alpha_token="${alpha_token//./p}"
  beta_token="${beta_token//./p}"
  local candidates=()
  case "$method" in
    vilun)
      candidates=(
        "${SAVE_DIR}/unlearned_vilun_heldout_main_vilun_${dataset}_${model}_rnn_a${alpha_token}_b${beta_token}_seed${SEED}.pth"
      )
      ;;
    vilun_f)
      candidates=(
        "${SAVE_DIR}/vilun_main_vilunf_${dataset}_${model}_rnn_a${alpha_token}_b${beta_token}_seed${SEED}.pth"
      )
      ;;
    *)
      echo "[ERROR] Unsupported method: ${method}" >&2
      return 2
      ;;
  esac
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return
    fi
  done
  echo "${candidates[0]}"
}

forget_indices_path() {
  local dataset="$1" model="$2"
  local candidates=(
    "${SOURCE_ROOT}/history/forget_indices_${dataset}_${model}_seed${SEED}.pt"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return
    fi
  done
  echo "${candidates[0]}"
}

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
  local tag="$1" history_dir="$2"
  shift 2
  local done_file="${DONE_DIR}/${tag}.done"
  local result_file="${history_dir}/summary_mia.csv"

  if [ "$FORCE" = "1" ]; then
    rm -f "$done_file"
  elif [ -f "$done_file" ] && [ -s "$result_file" ]; then
    echo "[SKIP] ${tag}"
    return
  fi

  find_free_gpu
  local gpu="$FREE_GPU"
  local logfile="${LOG_DIR}/${tag}.log"
  mkdir -p "$history_dir"
  rm -f "$result_file"
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

run_one() {
  local method="$1" dataset="$2" model="$3"
  local checkpoint forget_path history_dir tag
  checkpoint="$(checkpoint_path "$method" "$dataset" "$model")"
  forget_path="$(forget_indices_path "$dataset" "$model")"
  history_dir="${RUN_DIR}/${method}_${dataset}_${model}_seed${SEED}"
  tag="mia_${method}_${dataset}_${model}_seed${SEED}"

  local command=(
    "$PYTHON_BIN" "${REPO_DIR}/evaluate_mia.py"
    --method "$method"
    --dataset "$dataset"
    --model "$model"
    --data_dir "$DATA_DIR"
    --history_dir "$history_dir"
    --model_path "$checkpoint"
    --forget_indices_path "$forget_path"
    --seed "$SEED"
    --representation_mia
    --repr_attack_epochs "$REPR_ATTACK_EPOCHS"
    --repr_attack_batch_size "$REPR_ATTACK_BATCH_SIZE"
    --repr_attack_lr "$REPR_ATTACK_LR"
    --repr_attack_train_fraction "$REPR_ATTACK_TRAIN_FRACTION"
    --repr_attack_nonmember_source "$REPR_ATTACK_NONMEMBER_SOURCE"
  )
  if [ "$INCLUDE_GRADIENT_MIA" = "1" ]; then
    command+=(--gradient_mia)
  fi

  echo "[FOUND] ${method}/${dataset}: ${checkpoint}"
  echo "        forget indices: ${forget_path}"
  submit_once "$tag" "$history_dir" "${command[@]}"
}

echo "============================================================"
echo "Existing ViLUn/ViLUn_f output + representation MIA"
echo "repo       : ${REPO_DIR}"
echo "source     : ${SOURCE_ROOT}"
echo "checkpoints: ${SAVE_DIR}"
echo "data       : ${DATA_DIR}"
echo "output     : ${OUTPUT_ROOT}"
echo "datasets   : ${DATASETS[*]}"
echo "methods    : ${METHODS[*]}"
echo "GPUs       : ${GPUS[*]}"
echo "============================================================"

# Fail before launching any job if one of the requested artifacts is absent.
PREFLIGHT_FAILED=0
for dataset in "${DATASETS[@]}"; do
  if [ -z "${DS_MODEL[$dataset]+x}" ]; then
    echo "[ERROR] Unsupported dataset: ${dataset}" >&2
    PREFLIGHT_FAILED=1
    continue
  fi
  model="${DS_MODEL[$dataset]}"
  forget_path="$(forget_indices_path "$dataset" "$model")"
  if [ ! -f "$forget_path" ]; then
    echo "[ERROR] Missing forget indices: ${forget_path}" >&2
    PREFLIGHT_FAILED=1
  fi
  for method in "${METHODS[@]}"; do
    if ! checkpoint="$(checkpoint_path "$method" "$dataset" "$model")"; then
      PREFLIGHT_FAILED=1
      continue
    fi
    if [ ! -f "$checkpoint" ]; then
      echo "[ERROR] Missing checkpoint: ${checkpoint}" >&2
      PREFLIGHT_FAILED=1
    else
      echo "[CHECK] ${method}/${dataset}: ${checkpoint}"
    fi
  done
done

if [ "$PREFLIGHT_FAILED" -ne 0 ]; then
  echo "[ERROR] Preflight failed; no MIA job was launched." >&2
  exit 2
fi
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "[DONE] Preflight completed; no MIA job was launched."
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  model="${DS_MODEL[$dataset]}"
  for method in "${METHODS[@]}"; do
    run_one "$method" "$dataset" "$model"
  done
done

wait_all

"$PYTHON_BIN" - "$RUN_DIR" "$SUMMARY_PATH" "${#DATASETS[@]}" "${#METHODS[@]}" <<'PY'
import csv
import os
import sys

run_dir, output_path, dataset_count, method_count = sys.argv[1:]
expected = int(dataset_count) * int(method_count)
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
if len(rows) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} rows, but found {len(rows)}")
PY

"$PYTHON_BIN" - "$BASELINE_SUMMARY" "$SUMMARY_PATH" "$MERGED_SUMMARY" <<'PY'
import csv
import os
import sys

baseline_path, unlearned_path, output_path = sys.argv[1:]
rows = []
fieldnames = []

def add_file(path):
    if not os.path.isfile(path):
        return False
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    return True

has_baseline = add_file(baseline_path)
add_file(unlearned_path)

# New ViLUn rows take precedence if an older merged file already contained them.
deduplicated = {}
for row in rows:
    key = (row.get("Method", ""), row.get("Dataset", ""), row.get("Model", ""))
    deduplicated[key] = row
rows = sorted(deduplicated.values(), key=lambda row: (
    row.get("Method", ""), row.get("Dataset", ""), row.get("Model", "")
))

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"[SAVE] Figure-ready merged summary ({len(rows)} rows) -> {output_path}")
if not has_baseline:
    print(f"[WARN] Baseline summary not found: {baseline_path}")
    print("       The merged file currently contains only ViLUn/ViLUn_f rows.")
PY

if [ "$JOB_FAILED" -ne 0 ]; then
  echo "[ERROR] One or more MIA jobs failed. Check ${LOG_DIR}." >&2
  exit 1
fi

echo "[DONE] Existing ViLUn/ViLUn_f MIA evaluation completed."
echo "       ViLUn only : ${SUMMARY_PATH}"
echo "       Figure CSV : ${MERGED_SUMMARY}"
