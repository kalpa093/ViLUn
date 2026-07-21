#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_LLM_RETRAIN="${RUN_LLM_RETRAIN:-1}"
LLM_LAYERS="${LLM_LAYERS:-8}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-16}"
LLM_TRAIN_EPOCHS="${LLM_TRAIN_EPOCHS:-10}"
LLM_UNLEARN_EPOCHS="${LLM_UNLEARN_EPOCHS:-100}"
LLM_NAME="${LLM_ORIG_MODEL##*/}"
LLM_ORIGINAL="${SAVE_DIR}/original_llm_${LLM_NAME}_cifar10_seed${SEED}.pt"
LLM_FORGET_INDICES="${HISTORY_DIR}/forget_indices_vilun_llm_cifar10_${LLM_NAME}_cnn_seed${SEED}.pt"
print_config

# Prepare one shared original checkpoint before launching the architecture grid.
submit_once "llm_original_cifar10_seed${SEED}" "$PYTHON_BIN" "${REPO_DIR}/vilun_llm.py" \
  --mode heldout --orig_model "$LLM_ORIG_MODEL" --expert_model cnn \
  --data_dir "$DATA_DIR" --seed "$SEED" \
  --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
  --train_epochs "$LLM_TRAIN_EPOCHS" --unlearn_epochs 0 \
  --llm_layers "$LLM_LAYERS" --batch_size "$LLM_BATCH_SIZE" \
  --device_orig cuda:0 --device_expert cuda:0
wait_all

if [ ! -f "$LLM_ORIGINAL" ]; then
  echo "[ERROR] Missing prepared LLM checkpoint: $LLM_ORIGINAL" >&2
  exit 2
fi

if [ "$RUN_LLM_RETRAIN" = "1" ]; then
  submit_once "llm_retrain_cifar10_seed${SEED}" "$PYTHON_BIN" "${REPO_DIR}/run_qwen3b_retrain.py" \
    --orig_model "$LLM_ORIG_MODEL" --data_dir "$DATA_DIR" \
    --history_dir "$HISTORY_DIR" --save_dir "$SAVE_DIR" \
    --save_tag "retrain_llm_cifar10_seed${SEED}" \
    --forget_indices_path "$LLM_FORGET_INDICES" --seed "$SEED" \
    --llm_layers "$LLM_LAYERS" --batch_size "$LLM_BATCH_SIZE" \
    --device cuda:0
fi

for vm in "${LLM_EXPERTS[@]}"; do
  tag="llm_vilun_cifar10_$(fmt_token "${vm}")_seed${SEED}"
  submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun_llm.py" \
    --mode heldout --orig_model "$LLM_ORIG_MODEL" --expert_model "$vm" \
    --original "$LLM_ORIGINAL" --data_dir "$DATA_DIR" --seed "$SEED" \
    --alpha "$DEFAULT_ALPHA" --beta "$DEFAULT_BETA" \
    --heldout_ratio "$DEFAULT_HELDOUT_RATIO" \
    --unlearn_epochs "$LLM_UNLEARN_EPOCHS" \
    --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
    --save_tag "$tag" --save_model \
    --llm_layers "$LLM_LAYERS" --batch_size "$LLM_BATCH_SIZE" \
    --device_orig cuda:0 --device_expert cuda:0

  tag="llm_vilunf_cifar10_$(fmt_token "${vm}")_seed${SEED}"
  submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun_llm.py" \
    --mode standard --orig_model "$LLM_ORIG_MODEL" --expert_model "$vm" \
    --original "$LLM_ORIGINAL" --data_dir "$DATA_DIR" --seed "$SEED" \
    --alpha "$DEFAULT_ALPHA" --beta "$DEFAULT_BETA" \
    --heldout_ratio "$DEFAULT_HELDOUT_RATIO" \
    --unlearn_epochs "$LLM_UNLEARN_EPOCHS" \
    --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
    --save_tag "$tag" --save_model \
    --llm_layers "$LLM_LAYERS" --batch_size "$LLM_BATCH_SIZE" \
    --device_orig cuda:0 --device_expert cuda:0
done
wait_all

echo "[DONE] LLM original, Retrain, ViLUn, and ViLUn_f experiments completed."
