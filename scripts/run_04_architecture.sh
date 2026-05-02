#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
RUN_IMAGE_ARCH="${RUN_IMAGE_ARCH:-1}"
print_config

if [ "$RUN_IMAGE_ARCH" = "1" ]; then
  for ds in "${ARCH_DATASETS[@]}"; do
    default_om="${DS_MODEL[$ds]}"
    for om in "${MODELS[@]}"; do
      for vm in "${MODELS[@]}"; do
        if [ "$om" = "$default_om" ] && [ "$vm" = "rnn" ]; then continue; fi
        run_vilun "arch_vilun_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO"
        run_vilun_f "arch_vilunf_${ds}_${om}_${vm}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO"
      done
    done
  done
  wait_all
fi
