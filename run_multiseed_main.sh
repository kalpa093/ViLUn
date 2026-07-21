#!/usr/bin/env bash
set -euo pipefail

# Repeat the main image-domain experiments with paired random seeds.
# Every seed gets an isolated output root so summaries and checkpoints cannot
# overwrite one another. Run this file with `bash run_multiseed_main.sh`.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
MULTISEED_ROOT="${MULTISEED_ROOT:-${REPO_DIR}/vilun_runs/multiseed}"

read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46}"
read -r -a STAGE_LIST <<< "${STAGES:-prepare baselines main mia}"
GPUS="${GPUS:-0 1 2}"

# Paper settings are explicit here so changes to common.sh defaults do not
# silently change the statistical replication runs.
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-50}"
UNLEARN_LR="${UNLEARN_LR:-0.0001}"
DEFAULT_ALPHA="${DEFAULT_ALPHA:-1}"
DEFAULT_BETA="${DEFAULT_BETA:-2}"
FEATURE_MARGIN="${FEATURE_MARGIN:--0.2}"
DEFAULT_HELDOUT_RATIO="${DEFAULT_HELDOUT_RATIO:-0.05}"
DEFAULT_RETAIN_RATIO="${DEFAULT_RETAIN_RATIO:-0.05}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
FORCE_STAGES="${FORCE_STAGES:-0}"

stage_script() {
  case "$1" in
    prepare)   echo "${REPO_DIR}/scripts/run_00_prepare.sh" ;;
    baselines) echo "${REPO_DIR}/scripts/run_01_baselines.sh" ;;
    main)      echo "${REPO_DIR}/scripts/run_02_main_sample.sh" ;;
    mia)       echo "${REPO_DIR}/scripts/run_08_multiseed_mia.sh" ;;
    *)
      echo "[ERROR] Unknown stage: $1" >&2
      return 2
      ;;
  esac
}

mkdir -p "$MULTISEED_ROOT"

echo "============================================================"
echo "Multi-seed main evaluation"
echo "repo       : $REPO_DIR"
echo "output     : $MULTISEED_ROOT"
echo "data       : $DATA_DIR"
echo "seeds      : ${SEED_LIST[*]}"
echo "stages     : ${STAGE_LIST[*]}"
echo "gpus       : $GPUS"
echo "============================================================"

for seed in "${SEED_LIST[@]}"; do
  seed_root="${MULTISEED_ROOT}/seed${seed}"
  mkdir -p "$seed_root"

  for stage in "${STAGE_LIST[@]}"; do
    script="$(stage_script "$stage")"
    driver_log="${seed_root}/driver_${stage}.log"
    stage_done="${seed_root}/.driver_${stage}.done"
    if [ "$FORCE_STAGES" != "1" ] && [ -f "$stage_done" ]; then
      echo "[SKIP] seed=${seed} stage=${stage}"
      continue
    fi
    echo "[$(date '+%F %T')] seed=${seed} stage=${stage}"
    (
      cd "$seed_root"
      env \
        PYTHON_BIN="$PYTHON_BIN" \
        ROOT_DIR="$seed_root" \
        DATA_DIR="$DATA_DIR" \
        SEED="$seed" \
        GPUS="$GPUS" \
        TRAIN_EPOCHS="$TRAIN_EPOCHS" \
        UNLEARN_EPOCHS="$UNLEARN_EPOCHS" \
        UNLEARN_LR="$UNLEARN_LR" \
        DEFAULT_ALPHA="$DEFAULT_ALPHA" \
        DEFAULT_BETA="$DEFAULT_BETA" \
        FEATURE_MARGIN="$FEATURE_MARGIN" \
        DEFAULT_HELDOUT_RATIO="$DEFAULT_HELDOUT_RATIO" \
        DEFAULT_RETAIN_RATIO="$DEFAULT_RETAIN_RATIO" \
        PRETRAIN_MAIN_ONLY=1 \
        bash "$script"
    ) 2>&1 | tee "$driver_log"
    touch "$stage_done"
  done
done

if [ "$RUN_ANALYSIS" = "1" ]; then
  "$PYTHON_BIN" "${REPO_DIR}/analyze_multiseed_main.py" \
    --root "$MULTISEED_ROOT" \
    --seeds "${SEED_LIST[@]}"
fi

echo "[DONE] Multi-seed main evaluation completed."
