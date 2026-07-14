#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-./vilun_final}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./villain_mia_runs_rnn}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
EXPERT_EPOCHS="${EXPERT_EPOCHS:-50}"
LR="${LR:-0.001}"
PATIENCE="${PATIENCE:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_one() {
  local dataset="$1"
  local orig_model="$2"
  local forget_path="${SOURCE_ROOT}/history/forget_indices_${dataset}_${orig_model}_seed${SEED}.pt"

  echo "============================================================"
  echo "[Villain MIA RNN] dataset=${dataset} | forget=${forget_path}"
  echo "============================================================"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_villain_mia_only.py" \
    --dataset "${dataset}" \
    --expert_model rnn \
    --forget_indices_path "${forget_path}" \
    --data_dir "${DATA_DIR}" \
    --output_root "${OUTPUT_ROOT}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --expert_epochs "${EXPERT_EPOCHS}" \
    --lr "${LR}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}"
}

run_one cifar10 resnet18
run_one cifar100 resnet50
run_one tinyimagenet resnet50

echo

echo "[Done] RNN villain MIA runs completed for CIFAR-10, CIFAR-100, and TinyImageNet."
echo "[Done] Output root: ${OUTPUT_ROOT}"

