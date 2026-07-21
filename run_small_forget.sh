#!/usr/bin/env bash
set -euo pipefail

# Standalone small-forget experiment runner.
# Run from any directory with: bash /path/to/ViLUn/run_small_forget.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/vilun_runs}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${BASE_ROOT_DIR}/saved_models}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"

SMALL_ROOT_DIR="${SMALL_ROOT_DIR:-${BASE_ROOT_DIR}/small_forget}"
SAVE_DIR="${SMALL_SAVE_DIR:-${SMALL_ROOT_DIR}/saved_models}"
HISTORY_DIR="${SMALL_HISTORY_DIR:-${SMALL_ROOT_DIR}/history}"
LOG_DIR="${SMALL_LOG_DIR:-${SMALL_ROOT_DIR}/logs}"
DONE_DIR="${SMALL_DONE_DIR:-${SMALL_ROOT_DIR}/.done}"
INDEX_DIR="${SMALL_INDEX_DIR:-${HISTORY_DIR}/small_forget_indices}"
mkdir -p "$SAVE_DIR" "$HISTORY_DIR" "$LOG_DIR" "$DONE_DIR" "$INDEX_DIR"

SEED="${SEED:-42}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
EXPERT_EPOCHS="${EXPERT_EPOCHS:-50}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-50}"
UNLEARN_LR="${UNLEARN_LR:-0.0001}"
ALPHA="${SMALL_ALPHA:-${DEFAULT_ALPHA:-1}}"
BETA="${SMALL_BETA:-8}"
FEATURE_MARGIN="${FEATURE_MARGIN:--0.2}"
HELDOUT_RATIO="${SMALL_HELDOUT_RATIO:-${DEFAULT_HELDOUT_RATIO:-0.05}}"
RETAIN_RATIO="${SMALL_RETAIN_RATIO:-${DEFAULT_RETAIN_RATIO:-0.05}}"
EXPERT_MODEL="${SMALL_EXPERT_MODEL:-rnn}"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
read -r -a DATASETS <<< "${DATASETS:-cifar10 cifar100 tinyimagenet}"
read -r -a FORGET_RATIOS <<< "${SMALL_FORGET_RATIOS:-0.01 0.005 0.001 0.0005 0.0001}"

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

fmt_token() {
  local raw="$1"
  echo "${raw//./p}"
}

generate_forget_indices() {
  local dataset="$1" model="$2" ratio="$3" out_path="$4"
  "$PYTHON_BIN" - "$dataset" "$model" "$DATA_DIR" "$out_path" "$ratio" "$SEED" <<'PY'
import csv
import os
import random
import sys

import torch

from pretrain_models import load_dataset

dataset, model, data_dir, out_path, ratio_raw, seed_raw = sys.argv[1:]
ratio = float(ratio_raw)
seed = int(seed_raw)
if not 0 < ratio <= 1:
    raise ValueError(f"forget ratio must be in (0, 1], got {ratio}")

train_set, _, _, _, _ = load_dataset(dataset, data_dir, train_augment=False)
n_total = len(train_set)
n_forget = max(1, min(n_total, round(ratio * n_total)))
os.makedirs(os.path.dirname(out_path), exist_ok=True)

if os.path.exists(out_path):
    existing = torch.load(out_path, map_location="cpu").long()
    if len(existing) != n_forget:
        raise ValueError(
            f"Existing split has {len(existing)} samples, expected {n_forget}: {out_path}"
        )
    print(f"[SKIP] Existing forget indices: {out_path} ({len(existing)} samples)")
    raise SystemExit(0)

rng = random.Random(seed)
forget_indices = torch.tensor(rng.sample(range(n_total), n_forget), dtype=torch.long)
torch.save(forget_indices, out_path)
print(f"[SAVE] {out_path} ({n_forget}/{n_total}, ratio={ratio:g})")

summary_path = os.path.join(os.path.dirname(out_path), "small_forget_indices_summary.csv")
write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
with open(summary_path, "a", newline="") as handle:
    writer = csv.writer(handle)
    if write_header:
        writer.writerow(["Dataset", "Model", "Seed", "Ratio", "N_Total", "N_Forget", "Path"])
    writer.writerow([dataset, model, seed, f"{ratio:g}", n_total, n_forget, out_path])
PY
}

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
  shift
  local done_file="${DONE_DIR}/${tag}.done"
  if [ -f "$done_file" ]; then
    echo "[SKIP] ${tag}"
    return
  fi

  find_free_gpu
  local gpu="$FREE_GPU"
  local logfile="${LOG_DIR}/${tag}.log"
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
  if [ "$JOB_FAILED" -ne 0 ]; then
    exit 1
  fi
}

run_pair() {
  local dataset="$1" original_model="$2" ratio="$3"
  local ratio_token
  ratio_token="$(fmt_token "$ratio")"
  local index_path="${INDEX_DIR}/forget_indices_${dataset}_${original_model}_seed${SEED}_fr${ratio_token}.pt"

  generate_forget_indices "$dataset" "$original_model" "$ratio" "$index_path"

  local original_args=()
  local base_original="${BASE_SAVE_DIR}/original_${dataset}_${original_model}_seed${SEED}.pth"
  if [ -f "$base_original" ]; then
    original_args=(--original "$base_original")
  else
    echo "[WARN] Original checkpoint not found; the pipeline will prepare it: ${base_original}"
  fi

  local common_args=(
    --dataset "$dataset" --model "$original_model" --expert_model "$EXPERT_MODEL"
    --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR"
    --seed "$SEED"
    --train_epochs "$TRAIN_EPOCHS" --expert_epochs "$EXPERT_EPOCHS"
    --unlearn_epochs "$UNLEARN_EPOCHS" --unlearn_lr "$UNLEARN_LR"
    --alpha "$ALPHA" --beta "$BETA" --feature_margin "$FEATURE_MARGIN"
    --heldout_ratio "$HELDOUT_RATIO" --retain_ratio "$RETAIN_RATIO"
    --forget_ratio "$ratio" --forget_indices_path "$index_path"
    --no_reuse_tuned_params --freeze_beta
  )

  local tag="smallforget_vilun_${dataset}_${original_model}_${EXPERT_MODEL}_fr${ratio_token}_a$(fmt_token "$ALPHA")_b$(fmt_token "$BETA")_seed${SEED}"
  submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun.py" \
    "${common_args[@]}" --save_tag "$tag" "${original_args[@]}"

  tag="smallforget_vilunf_${dataset}_${original_model}_${EXPERT_MODEL}_fr${ratio_token}_a$(fmt_token "$ALPHA")_b$(fmt_token "$BETA")_seed${SEED}"
  submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun_f.py" \
    "${common_args[@]}" --save_tag "$tag" "${original_args[@]}"
}

echo "============================================================"
echo "Small-forget ViLUn experiments"
echo "repo         : ${REPO_DIR}"
echo "output       : ${SMALL_ROOT_DIR}"
echo "base models  : ${BASE_SAVE_DIR}"
echo "datasets     : ${DATASETS[*]}"
echo "ratios       : ${FORGET_RATIOS[*]}"
echo "GPUs         : ${GPUS[*]}"
echo "expert model : ${EXPERT_MODEL}"
echo "============================================================"

for dataset in "${DATASETS[@]}"; do
  if [ -z "${DS_MODEL[$dataset]+x}" ]; then
    echo "[ERROR] Unsupported dataset: ${dataset}" >&2
    exit 2
  fi
  original_model="${DS_MODEL[$dataset]}"
  for ratio in "${FORGET_RATIOS[@]}"; do
    run_pair "$dataset" "$original_model" "$ratio"
  done
done

wait_all
echo "[DONE] All small-forget experiments completed."
