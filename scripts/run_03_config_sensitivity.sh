#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
print_config

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  vm="rnn"
  for alpha in "${ALPHA_VALUES[@]}"; do
    atag="$(fmt_token "$alpha")"
    for beta in "${BETA_VALUES[@]}"; do
      btag="$(fmt_token "$beta")"
      run_vilun "hparam_vilun_${ds}_${om}_${vm}_a${atag}_b${btag}_seed${SEED}" "$ds" "$om" "$vm" "$alpha" "$beta" "$DEFAULT_HELDOUT_RATIO"
      run_vilun_f "hparam_vilunf_${ds}_${om}_${vm}_a${atag}_b${btag}_seed${SEED}" "$ds" "$om" "$vm" "$alpha" "$beta" "$DEFAULT_HELDOUT_RATIO"
    done
  done
done
wait_all

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  vm="rnn"
  for hr in "${HELDOUT_RATIOS[@]}"; do
    hrtag="$(fmt_token "$hr")"
    run_vilun "heldout_vilun_${ds}_${om}_${vm}_hr${hrtag}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$hr"
  done
done
wait_all

for ds in "${DATASETS[@]}"; do
  om="${DS_MODEL[$ds]}"
  vm="rnn"
  for rr in "${RETAIN_RATIOS[@]}"; do
    rrtag="$(fmt_token "$rr")"
    run_vilun "retain_vilun_${ds}_${om}_${vm}_rr${rrtag}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO" "$rr"
    run_vilun_f "retain_vilunf_${ds}_${om}_${vm}_rr${rrtag}_seed${SEED}" "$ds" "$om" "$vm" "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$DEFAULT_HELDOUT_RATIO" "$rr"
  done
done
wait_all
