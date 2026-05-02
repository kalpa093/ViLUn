#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_00_prepare.sh"
"${SCRIPT_DIR}/run_01_baselines.sh"
"${SCRIPT_DIR}/run_02_main_sample.sh"
"${SCRIPT_DIR}/run_03_config_sensitivity.sh"
"${SCRIPT_DIR}/run_04_architecture.sh"
"${SCRIPT_DIR}/run_05_privacy_mia.sh"
"${SCRIPT_DIR}/run_06_ablation_scope.sh"
"${SCRIPT_DIR}/run_07_llm_mia.sh"
