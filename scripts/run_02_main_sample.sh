#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_config

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  vm="rnn"
  run_vilun "main_vilun_${ds}_${om}_${vm}_a$(fmt_token "$DEFAULT_ALPHA")_b$(fmt_token "$DEFAULT_BETA")_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO"
  run_vilun_f "main_vilunf_${ds}_${om}_${vm}_a$(fmt_token "$DEFAULT_ALPHA")_b$(fmt_token "$DEFAULT_BETA")_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO"
done
wait_all
