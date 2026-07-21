#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_config

PRETRAIN_ARGS=()
if [ "${PRETRAIN_MAIN_ONLY:-0}" = "1" ]; then
  PRETRAIN_ARGS+=(--main_only)
fi
if [ "${FORCE_EXPERTS:-0}" = "1" ]; then
  PRETRAIN_ARGS+=(--force_experts)
fi

"$PYTHON_BIN" "${REPO_DIR}/pretrain_models.py" \
  --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
  --seed "$SEED" --gpus "${GPUS[@]}" "${PRETRAIN_ARGS[@]}"

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  submit_once "ref_retrain_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py" \
    --unlearning retrain --dataset "$ds" --model "$om" \
    --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" --seed "$SEED"
done
wait_all
