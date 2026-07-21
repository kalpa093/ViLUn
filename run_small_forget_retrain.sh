#!/usr/bin/env bash
set -euo pipefail

# Train one Retrain oracle for every existing small-forget split.
# This script does not rerun ViLUn or ViLUn_f.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
SEED="${SEED:-42}"

BASE_ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/vilun_runs}"
SMALL_ROOT_DIR="${SMALL_ROOT_DIR:-${BASE_ROOT_DIR}/small_forget}"
INDEX_DIR="${SMALL_INDEX_DIR:-${SMALL_ROOT_DIR}/history/small_forget_indices}"
OUTPUT_ROOT="${SMALL_RETRAIN_ROOT:-${SMALL_ROOT_DIR}/retrain}"
SAVE_DIR="${SMALL_RETRAIN_SAVE_DIR:-${OUTPUT_ROOT}/saved_models}"
RUN_DIR="${SMALL_RETRAIN_RUN_DIR:-${OUTPUT_ROOT}/runs}"
LOG_DIR="${SMALL_RETRAIN_LOG_DIR:-${OUTPUT_ROOT}/logs}"
DONE_DIR="${SMALL_RETRAIN_DONE_DIR:-${OUTPUT_ROOT}/.done}"
SUMMARY_PATH="${SMALL_RETRAIN_SUMMARY:-${SMALL_ROOT_DIR}/history/summary_small_forget_retrain.csv}"
mkdir -p "$SAVE_DIR" "$RUN_DIR" "$LOG_DIR" "$DONE_DIR" "$(dirname "$SUMMARY_PATH")"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
read -r -a DATASETS <<< "${DATASETS:-cifar10 cifar100 tinyimagenet}"
read -r -a FORGET_RATIOS <<< "${SMALL_FORGET_RATIOS:-0.01 0.005 0.001 0.0005 0.0001}"

PATIENCE="${RETRAIN_PATIENCE:-50}"
FORCE="${FORCE:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

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

fmt_token() {
  local raw="$1"
  echo "${raw//./p}"
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
  local result_file="${history_dir}/summary_retrain_ga.csv"

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
  local dataset="$1" model="$2" ratio="$3"
  local ratio_token tag forget_path history_dir
  ratio_token="$(fmt_token "$ratio")"
  tag="smallforget_retrain_${dataset}_${model}_fr${ratio_token}_seed${SEED}"
  forget_path="${INDEX_DIR}/forget_indices_${dataset}_${model}_seed${SEED}_fr${ratio_token}.pt"
  history_dir="${RUN_DIR}/${tag}"

  local command=(
    "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py"
    --unlearning retrain
    --dataset "$dataset"
    --model "$model"
    --data_dir "$DATA_DIR"
    --save_path "$SAVE_DIR"
    --history_dir "$history_dir"
    --forget_indices_path "$forget_path"
    --forget_ratio "$ratio"
    --save_tag "$tag"
    --seed "$SEED"
    --patience "$PATIENCE"
    --final_eval_only
  )

  echo "[FOUND] ${dataset}/${ratio}: ${forget_path}"
  submit_once "$tag" "$history_dir" "${command[@]}"
}

echo "============================================================"
echo "Small-forget Retrain oracles"
echo "repo       : ${REPO_DIR}"
echo "indices    : ${INDEX_DIR}"
echo "output     : ${OUTPUT_ROOT}"
echo "summary    : ${SUMMARY_PATH}"
echo "datasets   : ${DATASETS[*]}"
echo "ratios     : ${FORGET_RATIOS[*]}"
echo "GPUs       : ${GPUS[*]}"
echo "============================================================"

PREFLIGHT_FAILED=0
for dataset in "${DATASETS[@]}"; do
  if [ -z "${DS_MODEL[$dataset]+x}" ]; then
    echo "[ERROR] Unsupported dataset: ${dataset}" >&2
    PREFLIGHT_FAILED=1
    continue
  fi
  model="${DS_MODEL[$dataset]}"
  for ratio in "${FORGET_RATIOS[@]}"; do
    ratio_token="$(fmt_token "$ratio")"
    forget_path="${INDEX_DIR}/forget_indices_${dataset}_${model}_seed${SEED}_fr${ratio_token}.pt"
    if [ ! -f "$forget_path" ]; then
      echo "[ERROR] Missing forget indices: ${forget_path}" >&2
      PREFLIGHT_FAILED=1
    else
      echo "[CHECK] ${dataset}/${ratio}: ${forget_path}"
    fi
  done
done

if [ "$PREFLIGHT_FAILED" -ne 0 ]; then
  echo "[ERROR] Preflight failed; no Retrain job was launched." >&2
  exit 2
fi
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "[DONE] Preflight completed; no Retrain job was launched."
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  model="${DS_MODEL[$dataset]}"
  for ratio in "${FORGET_RATIOS[@]}"; do
    run_one "$dataset" "$model" "$ratio"
  done
done

wait_all

if [ "$JOB_FAILED" -ne 0 ]; then
  echo "[ERROR] One or more Retrain jobs failed. Check ${LOG_DIR}." >&2
  exit 1
fi

"$PYTHON_BIN" - \
  "$RUN_DIR" \
  "$SUMMARY_PATH" \
  "$SEED" \
  "${DATASETS[*]}" \
  "${FORGET_RATIOS[*]}" <<'PY'
import csv
import os
import sys

run_dir, output_path, seed, datasets_raw, ratios_raw = sys.argv[1:]
datasets = set(datasets_raw.split())
ratios = {float(value) for value in ratios_raw.split()}
expected = len(datasets) * len(ratios)
rows_by_key = {}
fieldnames = []

for root, _, files in os.walk(run_dir):
    if "summary_retrain_ga.csv" not in files:
        continue
    path = os.path.join(root, "summary_retrain_ga.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("unlearning") != "retrain":
                continue
            row_ratio = float(row.get("forget_ratio") or -1.0)
            if (
                row.get("dataset") not in datasets
                or int(row.get("seed") or -1) != int(seed)
                or row_ratio not in ratios
            ):
                continue
            row["result_path"] = path
            key = (
                row["dataset"],
                row["model"],
                int(row["seed"]),
                row_ratio,
            )
            if key in rows_by_key:
                raise SystemExit(
                    f"[ERROR] Duplicate Retrain result for {key}: "
                    f"{rows_by_key[key]['result_path']} and {path}"
                )
            rows_by_key[key] = row
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

rows = list(rows_by_key.values())
rows.sort(key=lambda row: (
    row.get("dataset", ""),
    -float(row.get("forget_ratio") or 0.0),
))

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"[SAVE] Combined {len(rows)} Retrain rows -> {output_path}")
if len(rows) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} rows, but found {len(rows)}")
PY

echo "[DONE] All small-forget Retrain oracles completed."
echo "       Summary: ${SUMMARY_PATH}"
