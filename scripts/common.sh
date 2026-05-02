#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/vilun_runs}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/saved_models}"
HISTORY_DIR="${HISTORY_DIR:-${ROOT_DIR}/history}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
DONE_DIR="${DONE_DIR:-${ROOT_DIR}/.done}"

mkdir -p "$ROOT_DIR" "$SAVE_DIR" "$HISTORY_DIR" "$LOG_DIR" "$DONE_DIR"

SEED="${SEED:-42}"
GPUS=(${GPUS:-0 1 2})

TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-50}"
UNLEARN_LR="${UNLEARN_LR:-0.00005}"
DEFAULT_ALPHA="${DEFAULT_ALPHA:-1}"
DEFAULT_BETA="${DEFAULT_BETA:-2}"
FEATURE_MARGIN="${FEATURE_MARGIN:--0.3}"
DEFAULT_HELDOUT_RATIO="${DEFAULT_HELDOUT_RATIO:-0.1}"
DEFAULT_RETAIN_RATIO="${DEFAULT_RETAIN_RATIO:-0.05}"

ALPHA_VALUES=(${ALPHA_VALUES:-0.5 1 2 4})
BETA_VALUES=(${BETA_VALUES:-0.5 1 2 4})
HELDOUT_RATIOS=(${HELDOUT_RATIOS:-0.05 0.1 0.2 0.5})
RETAIN_RATIOS=(${RETAIN_RATIOS:-0.05 0.1 0.5})
MODELS=(${MODELS:-cnn mlp rnn resnet18 resnet50})
DATASETS=(${DATASETS:-cifar10 cifar100 tinyimagenet})
ARCH_DATASETS=(${ARCH_DATASETS:-cifar10 cifar100 tinyimagenet})
CLASS_DATASETS=(${CLASS_DATASETS:-cifar10 cifar100})
CLASS_TARGET_ID="${CLASS_TARGET_ID:-0}"
LLM_EXPERTS=(${LLM_EXPERTS:-cnn rnn mlp})
LLM_ORIG_MODEL="${LLM_ORIG_MODEL:-Qwen/Qwen2.5-3B}"

declare -A DS_MODEL=(
  [cifar10]=resnet18
  [cifar100]=resnet50
  [tinyimagenet]=resnet50
)

declare -A GPU_PID
for g in "${GPUS[@]}"; do GPU_PID[$g]=""; done

fmt_token() {
  local raw="$1"
  echo "${raw//./p}"
}

gpu_is_free() {
  local pid="${GPU_PID[$1]}"
  [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null
}

find_free_gpu() {
  while true; do
    for g in "${GPUS[@]}"; do
      if gpu_is_free "$g"; then
        echo "$g"
        return
      fi
    done
    sleep 3
  done
}

submit_once() {
  local tag="$1"; shift
  local done_file="${DONE_DIR}/${tag}.done"
  if [ -f "$done_file" ]; then
    echo "[SKIP] ${tag}"
    return
  fi
  local g
  g="$(find_free_gpu)"
  local logfile="${LOG_DIR}/${tag}.log"
  echo "[$(date '+%H:%M:%S')] START GPU${g} ${tag}"
  (
    set -e
    CUDA_VISIBLE_DEVICES="$g" PYTHONUNBUFFERED=1 "$@" 2>&1 |
      sed --unbuffered "s/^/[GPU${g}|${tag}] /" | tee "$logfile"
    touch "$done_file"
  ) &
  GPU_PID[$g]=$!
}

wait_all() {
  local failed=0
  for g in "${GPUS[@]}"; do
    local pid="${GPU_PID[$g]}"
    if [ -n "$pid" ]; then
      if ! wait "$pid"; then
        echo "[ERROR] job on GPU${g} failed (pid=${pid})" >&2
        failed=1
      fi
    fi
    GPU_PID[$g]=""
  done
  if [ "$failed" -ne 0 ]; then exit 1; fi
}

run_vilun() {
  local tag="$1" ds="$2" om="$3" vm="$4" alpha="$5" beta="$6" heldout_ratio="$7" retain_ratio="${8:-$DEFAULT_RETAIN_RATIO}"
  shift 8 || true
  submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun.py" \
    --dataset "$ds" --model "$om" --expert_model "$vm" \
    --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
    --seed "$SEED" \
    --train_epochs "$TRAIN_EPOCHS" --unlearn_epochs "$UNLEARN_EPOCHS" --unlearn_lr "$UNLEARN_LR" \
    --alpha "$alpha" --beta "$beta" \
    --feature_margin "$FEATURE_MARGIN" \
    --heldout_ratio "$heldout_ratio" \
    --retain_ratio "$retain_ratio" \
    --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth" \
    --expert_path "${SAVE_DIR}/expert_${ds}_${vm}_seed${SEED}.pth" \
    --no_reuse_tuned_params --freeze_beta \
    --save_tag "$tag" "$@"
}

run_vilun_f() {
  local tag="$1" ds="$2" om="$3" vm="$4" alpha="$5" beta="$6" heldout_ratio="$7" retain_ratio="${8:-$DEFAULT_RETAIN_RATIO}"
  submit_once "$tag" "$PYTHON_BIN" "${REPO_DIR}/vilun_f.py" \
    --dataset "$ds" --model "$om" --expert_model "$vm" \
    --data_dir "$DATA_DIR" --save_path "$SAVE_DIR" --history_dir "$HISTORY_DIR" \
    --seed "$SEED" \
    --train_epochs "$TRAIN_EPOCHS" --unlearn_epochs "$UNLEARN_EPOCHS" --unlearn_lr "$UNLEARN_LR" \
    --alpha "$alpha" --beta "$beta" \
    --feature_margin "$FEATURE_MARGIN" \
    --heldout_ratio "$heldout_ratio" \
    --retain_ratio "$retain_ratio" \
    --original "${SAVE_DIR}/original_${ds}_${om}_seed${SEED}.pth" \
    --expert_path "${SAVE_DIR}/expert_${ds}_${vm}_seed${SEED}.pth" \
    --no_reuse_tuned_params --freeze_beta \
    --save_tag "$tag"
}

run_mia() {
  local method="$1" ds="$2" model="$3" model_path="$4" tag="$5"
  local fi_path="${HISTORY_DIR}/forget_indices_${ds}_${model}_seed${SEED}.pt"
  if [ ! -f "$model_path" ] || [ ! -f "$fi_path" ]; then
    echo "[SKIP] MIA missing input for ${tag}"
    return
  fi
  submit_once "mia_${tag}" "$PYTHON_BIN" "${REPO_DIR}/evaluate_mia.py" \
    --method "$method" --dataset "$ds" --model "$model" \
    --data_dir "$DATA_DIR" --history_dir "$HISTORY_DIR" \
    --seed "$SEED" \
    --model_path "$model_path" \
    --forget_indices_path "$fi_path"
}

print_config() {
  echo "============================================================"
  echo "ViLUn artifact"
  echo "repo    : ${REPO_DIR}"
  echo "root    : ${ROOT_DIR}"
  echo "data    : ${DATA_DIR}"
  echo "history : ${HISTORY_DIR}"
  echo "logs    : ${LOG_DIR}"
  echo "gpus    : ${GPUS[*]}"
  echo "============================================================"
}
