#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_config

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  vm="rnn"
  main_h="main_vilun_${ds}_${om}_${vm}_a$(fmt_token "$DEFAULT_ALPHA")_b$(fmt_token "$DEFAULT_BETA")_seed${SEED}"
  main_f="main_vilunf_${ds}_${om}_${vm}_a$(fmt_token "$DEFAULT_ALPHA")_b$(fmt_token "$DEFAULT_BETA")_seed${SEED}"
  run_mia "vilun_heldout" "$ds" "$om" "${SAVE_DIR}/unlearned_vilun_heldout_${main_h}.pth" "$main_h"
  run_mia "vilun" "$ds" "$om" "${SAVE_DIR}/vilun_${main_f}.pth" "$main_f"
done
wait_all

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
