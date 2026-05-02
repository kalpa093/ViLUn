#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_config

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  submit_once "baseline_ga_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py" \
    --unlearning gradient_ascent --dataset "$ds" --model "$om" \
    --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
    --seed "$SEED" --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
  submit_once "baseline_sisa_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/sisa_train.py" \
    --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
    --save_dir "${ROOT_DIR}/sisa_models" --seed "$SEED" --unlearn
  submit_once "baseline_salun_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/salun.py" \
    --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
    --save_dir "${ROOT_DIR}/salun_models" --seed "$SEED" \
    --load_path "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
  submit_once "baseline_ps_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/ps.py" \
    --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
    --save_dir "${ROOT_DIR}/ps_models" --seed "$SEED" \
    --load_path "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
  submit_once "baseline_delete_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/delete.py" \
    --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
    --save_path "$SAVE_DIR" --seed "$SEED" \
    --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
done
wait_all
