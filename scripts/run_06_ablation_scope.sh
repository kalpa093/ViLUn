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
  CLASS_ROOT="${CLASS_ROOT:-${ROOT_DIR}/class_level}"
  CLASS_HISTORY_DIR="${CLASS_HISTORY_DIR:-${CLASS_ROOT}/history}"
  CLASS_SAVE_DIR="${CLASS_SAVE_DIR:-${CLASS_ROOT}/saved_models}"
  mkdir -p "$CLASS_HISTORY_DIR" "$CLASS_SAVE_DIR"
  old_history="$HISTORY_DIR"

  "$PYTHON_BIN" "${REPO_DIR}/pretrain_models.py" \
    --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" --history_dir "$old_history" \
    --seed "$SEED" --gpus "${GPUS[@]}" --main_only \
    --extra_original cifar100:resnet18

  HISTORY_DIR="$CLASS_HISTORY_DIR"

  # Generate class-specific indices, then run ViLUn first so any missing
  # ResNet18 original checkpoint is prepared before the baselines start.
  for ds in "${CLASS_DATASETS[@]}"; do
    om="resnet18"
    vm="rnn"
    "$PYTHON_BIN" "${REPO_DIR}/make_class_forget_indices.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --history_dir "$HISTORY_DIR" --target_class "$CLASS_TARGET_ID" --seed "$SEED" \
      --also_write_baseline_names
    run_vilun "class_vilun_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" \
      "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO" \
      "$DEFAULT_RETAIN_RATIO" --expert_path ""
  done
  wait_all

  for ds in "${CLASS_DATASETS[@]}"; do
    om="resnet18"
    vm="rnn"
    submit_once "class_retrain_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py" \
      --unlearning retrain --dataset "$ds" --model "$om" \
      --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" --seed "$SEED"
    submit_once "class_ga_${ds}_${om}" "$PYTHON_BIN" "${REPO_DIR}/retrain_ga.py" \
      --unlearning gradient_ascent --dataset "$ds" --model "$om" \
      --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
      --seed "$SEED" --unlearn_lr "${GA_LR[$ds]}" \
      --unlearn_epochs "${BASELINE_EPOCHS[$ds]}" \
      --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"

    submit_from_dir "class_sisa_${ds}_${om}" "$CLASS_ROOT" "$PYTHON_BIN" "${REPO_DIR}/sisa_train.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --save_dir "${CLASS_ROOT}/sisa_models" --seed "$SEED" --unlearn \
      --shards 5 --slices 3 --epochs_per_slice "${SISA_EPOCHS[$ds]}"
    submit_from_dir "class_salun_${ds}_${om}" "$CLASS_ROOT" "$PYTHON_BIN" "${REPO_DIR}/salun.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --save_dir "${CLASS_ROOT}/salun_models" --seed "$SEED" \
      --unlearn_lr "${SALUN_LR[$ds]}" --unlearn_epochs "${BASELINE_EPOCHS[$ds]}" \
      --load_path "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
    submit_from_dir "class_ps_${ds}_${om}" "$CLASS_ROOT" "$PYTHON_BIN" "${REPO_DIR}/ps.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --save_dir "${CLASS_ROOT}/ps_models" --seed "$SEED" \
      --lr "${PS_LR[$ds]}" --epochs "${BASELINE_EPOCHS[$ds]}" \
      --load_path "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
    submit_from_dir "class_delete_${ds}_${om}" "$CLASS_ROOT" "$PYTHON_BIN" "${REPO_DIR}/delete.py" \
      --dataset "$ds" --model "$om" --data_dir "$DATA_DIR" \
      --save_path "$CLASS_SAVE_DIR" --seed "$SEED" \
      --unlearn_lr 0.00001 --unlearn_epochs "${BASELINE_EPOCHS[$ds]}" \
      --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth"
    run_vilun_f "class_vilunf_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" \
      "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO" \
      "$DEFAULT_RETAIN_RATIO" --expert_path ""
  done
  wait_all
  HISTORY_DIR="$old_history"
fi
