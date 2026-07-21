#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_config

PYTHON_BIN="$PYTHON_BIN" \
SOURCE_ROOT="$ROOT_DIR" \
DATA_DIR="$DATA_DIR" \
SEED="$SEED" \
GPUS="${GPUS[*]}" \
DATASETS="${DATASETS[*]}" \
BASELINE_MIA_OUTPUT_ROOT="${ROOT_DIR}/baseline_mia_metrics" \
bash "${REPO_DIR}/run_existing_baseline_mia_metrics.sh"

PYTHON_BIN="$PYTHON_BIN" \
SOURCE_ROOT="$ROOT_DIR" \
DATA_DIR="$DATA_DIR" \
SEED="$SEED" \
GPUS="${GPUS[*]}" \
DATASETS="${DATASETS[*]}" \
DEFAULT_ALPHA="$DEFAULT_ALPHA" \
DEFAULT_BETA="$DEFAULT_BETA" \
BASELINE_SUMMARY="${ROOT_DIR}/baseline_mia_metrics/summary_baseline_mia_metrics.csv" \
UNLEARNED_MIA_OUTPUT_ROOT="${ROOT_DIR}/unlearned_mia_metrics" \
bash "${REPO_DIR}/run_existing_unlearned_mia_metrics.sh"

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  vm="rnn"
  fi_path="${HISTORY_DIR}/forget_indices_${ds}_${om}_seed${SEED}.pt"
  if [ -f "$fi_path" ]; then
    submit_once "villain_mia_${ds}_${vm}" "$PYTHON_BIN" "${REPO_DIR}/run_villain_mia_only.py" \
      --dataset "$ds" --expert_model "$vm" --data_dir "$DATA_DIR" \
      --forget_indices_path "$fi_path" --output_root "${ROOT_DIR}/villain_mia" \
      --seed "$SEED"
  fi
done
wait_all

"$PYTHON_BIN" "${REPO_DIR}/summarize_villain_mia_runs.py" \
  --output_root "${ROOT_DIR}/villain_mia" \
  --expert_model rnn \
  --seed "$SEED"
