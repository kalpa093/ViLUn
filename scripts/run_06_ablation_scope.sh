#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
RUN_RANDOM_VILLAIN="${RUN_RANDOM_VILLAIN:-1}"
RUN_CLASS_LEVEL="${RUN_CLASS_LEVEL:-1}"
print_config

if [ "$RUN_RANDOM_VILLAIN" = "1" ]; then
  for ds in "${DATASETS[@]}"; do
    om="${DS_MODEL[$ds]}"
    vm="rnn"
    run_vilun "ablation_random_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO" "$DEFAULT_RETAIN_RATIO" "--random_expert"
  done
  wait_all
fi

if [ "$RUN_CLASS_LEVEL" = "1" ]; then
  CLASS_HISTORY_DIR="${CLASS_HISTORY_DIR:-${ROOT_DIR}/history_class}"
  mkdir -p "$CLASS_HISTORY_DIR"
  old_history="$HISTORY_DIR"
  HISTORY_DIR="$CLASS_HISTORY_DIR"
  for ds in "${CLASS_DATASETS[@]}"; do
    om="resnet18"
    vm="rnn"
    "$PYTHON_BIN" "${REPO_DIR}/make_class_forget_indices.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --history_dir "$HISTORY_DIR" --target_class "$CLASS_TARGET_ID" --seed "$SEED" \
      --also_write_baseline_names
    submit_once "class_retrain_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py" \
      --unlearning retrain --dataset "$ds" --model "$om" \
      --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" --seed "$SEED"
    submit_once "class_ga_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py" \
      --unlearning gradient_ascent --dataset "$ds" --model "$om" \
      --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
      --seed "$SEED" --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
    submit_once "class_delete_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/delete.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --save_path "$SAVE_DIR" --seed "$SEED" \
      --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
    run_vilun "class_vilun_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO"
    run_vilun_f "class_vilunf_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO"
  done
  wait_all
  HISTORY_DIR="$old_history"
fi
