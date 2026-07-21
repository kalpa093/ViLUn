#!/usr/bin/env bash
set -euo pipefail

# Evaluate every checkpoint generated for one multi-seed output root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:?ROOT_DIR must point to one seed output directory}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
SEED="${SEED:?SEED is required}"
GPUS="${GPUS:-0 1 2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

env \
  PYTHON_BIN="$PYTHON_BIN" \
  SOURCE_ROOT="$ROOT_DIR" \
  DATA_DIR="$DATA_DIR" \
  SEED="$SEED" \
  GPUS="$GPUS" \
  BASELINE_MIA_OUTPUT_ROOT="${ROOT_DIR}/baseline_mia_metrics" \
  bash "${REPO_DIR}/run_existing_baseline_mia_metrics.sh"

env \
  PYTHON_BIN="$PYTHON_BIN" \
  SOURCE_ROOT="$ROOT_DIR" \
  DATA_DIR="$DATA_DIR" \
  SEED="$SEED" \
  GPUS="$GPUS" \
  DEFAULT_ALPHA="${DEFAULT_ALPHA:-1}" \
  DEFAULT_BETA="${DEFAULT_BETA:-2}" \
  BASELINE_SUMMARY="${ROOT_DIR}/baseline_mia_metrics/summary_baseline_mia_metrics.csv" \
  UNLEARNED_MIA_OUTPUT_ROOT="${ROOT_DIR}/unlearned_mia_metrics" \
  bash "${REPO_DIR}/run_existing_unlearned_mia_metrics.sh"
