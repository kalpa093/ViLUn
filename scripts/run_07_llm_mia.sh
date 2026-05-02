#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
RUN_LLM_UNLEARN="${RUN_LLM_UNLEARN:-1}"
RUN_LLM_MIA="${RUN_LLM_MIA:-1}"
LLM_LAYERS="${LLM_LAYERS:-8}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-16}"
print_config

if [ "$RUN_LLM_UNLEARN" = "1" ]; then
  for vm in "${LLM_EXPERTS[@]}"; do
    tag="llm_vilun_cifar10_$(fmt_token "${vm}")_seed${SEED}"
    submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun_llm.py" \
      --mode heldout --orig_model "$LLM_ORIG_MODEL" --expert_model "$vm" \
      --data_dir "$DATA_DIR" --seed "$SEED" \
      --alpha "$DEFAULT_ALPHA" --beta "$DEFAULT_BETA" \
      --heldout_ratio "$DEFAULT_HELDOUT_RATIO" \
      --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
      --save_tag "$tag" --save_model \
      --llm_layers "$LLM_LAYERS" --batch_size "$LLM_BATCH_SIZE" \
      --device_orig cuda:0 --device_expert cuda:0
  done
  wait_all
fi

if [ "$RUN_LLM_MIA" = "1" ]; then
  for vm in "${LLM_EXPERTS[@]}"; do
    tag="llm_vilun_cifar10_$(fmt_token "${vm}")_seed${SEED}"
    submit_once "mia_${tag}" "$PYTHON_BIN" "${REPO_DIR}/run_llm_mia.py" \
      --orig_model "$LLM_ORIG_MODEL" --expert_model "$vm" \
      --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
      --save_tag "$tag" --seed "$SEED" \
      --llm_layers "$LLM_LAYERS" --batch_size "$LLM_BATCH_SIZE" \
      --device cuda:0
  done
  wait_all
fi
