#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Defaults. model.name is resolved from CONFIG using the current Python environment.
# =============================================================================
MODEL=""
EXP_NAME=""
CONFIG="configs/models/onetrans/onetrans.yaml"
BATCH=512
LR=""
MODE="all"
DATASET=""
RESUME=false
RESUME_FROM=""
WEIGHT=""
INIT_FROM=""

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
DATA_ROOT="${DATA_ROOT:-}"
TRAIN_DAY_RANGE="${TRAIN_DAY_RANGE:-}"
EVAL_DAY_RANGE="${EVAL_DAY_RANGE:-}"
DATA_CONFIG_RELATIVE=""
DATA_CONFIG_OVERRIDE=""
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/results}"
DATA_MANIFEST="${DATA_MANIFEST:-}"

NUM_WORKERS="${NUM_WORKERS:-8}"

usage() {
  cat <<'EOF'
Usage:
  bash run.sh [options]

Options:
  --mode MODE                   train, eval, or all (default)
  --dataset NAME                Required: small, large, or hotuser
  --exp EXP_NAME                Required. Output directory name shared by all ranks.
  --data-root PATH              Data directory (or set DATA_ROOT).
  --train-days START-END        Small/Large training range (default: 01-07 / 01-34).
  --eval-days START-END         Small/Large evaluation range (default: 08-08 / 35-35).
  --weight PATH                 Optional eval checkpoint override. Pass the
                                complete model checkpoint path ending in /last.
  --init-from PATH              Initialize train/all from model weights only.
                                Cannot be combined with --resume.
  --config PATH                 Model YAML path relative to PROJECT_ROOT
  --data-config PATH            RequestData YAML relative to PROJECT_ROOT;
                                default follows --dataset
  --batch N                     Per-process train/eval batch size
  --lr VALUE                    Override dense optimizer LR for train/all.
                                Omit to use the training YAML value.
  --resume PATH                 Resume train/all from an explicit full step_N
                                checkpoint. Bare --resume is not supported.
  -h, --help                    Show this help

Uses the current Python environment. Small/Large require dt=DAY01/*.parquet partitions.
HotUser uses train/ for training and valid/ for evaluation; test/ is not selected.
Set DATA_ROOT (or --data-root) and WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT.
OUTPUT_BASE defaults to PROJECT_ROOT/results.

Actions execute in train -> eval order. Examples:
  --mode train --dataset small --exp baseline --data-root /path/to/data
  --mode train --dataset large --exp exp_004_smoe_l_r2
  --mode train --dataset small --exp exp_005_lr1e-4_s_r1 --lr 1e-4
  --mode eval --dataset large --exp exp_001_smoe_l_r2
  --mode eval --dataset small --exp exp_009_eval_s_r2 --weight /path/to/checkpoints/last
  --mode all --dataset small --exp exp_006_smoe_s_r1
  --mode all --dataset hotuser --exp exp_100_adamw_s_r2
  --mode train --dataset large --exp exp_007_smoe_l_r2 --resume /path/to/checkpoints/step_5000
  --mode all --dataset small --exp exp_008_posttrain_s_r1 --init-from /path/to/pretrain/checkpoints/last
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "${value}" && "${value}" != --* ]] || die "${option} requires a value"
}

while (($# > 0)); do
  case "$1" in
    --mode)
      require_value "$1" "${2:-}"
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --data-root)
      require_value "$1" "${2:-}"
      DATA_ROOT="$2"
      shift 2
      ;;
    --data-root=*)
      require_value "--data-root" "${1#*=}"
      DATA_ROOT="${1#*=}"
      shift
      ;;
    --train-days|--eval-days)
      require_value "$1" "${2:-}"
      if [[ "$1" == --train-days ]]; then TRAIN_DAY_RANGE="$2"; else EVAL_DAY_RANGE="$2"; fi
      shift 2
      ;;
    --train-days=*)
      require_value "--train-days" "${1#*=}"
      TRAIN_DAY_RANGE="${1#*=}"
      shift
      ;;
    --eval-days=*)
      require_value "--eval-days" "${1#*=}"
      EVAL_DAY_RANGE="${1#*=}"
      shift
      ;;
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --exp)
      require_value "$1" "${2:-}"
      EXP_NAME="$2"
      shift 2
      ;;
    --exp=*)
      EXP_NAME="${1#*=}"
      shift
      ;;
    --weight)
      require_value "$1" "${2:-}"
      WEIGHT="$2"
      shift 2
      ;;
    --weight=*)
      require_value "--weight" "${1#*=}"
      WEIGHT="${1#*=}"
      shift
      ;;
    --init-from)
      require_value "$1" "${2:-}"
      INIT_FROM="$2"
      shift 2
      ;;
    --init-from=*)
      require_value "--init-from" "${1#*=}"
      INIT_FROM="${1#*=}"
      shift
      ;;
    --config)
      require_value "$1" "${2:-}"
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --data-config)
      require_value "$1" "${2:-}"
      DATA_CONFIG_OVERRIDE="$2"
      shift 2
      ;;
    --data-config=*)
      require_value "--data-config" "${1#*=}"
      DATA_CONFIG_OVERRIDE="${1#*=}"
      shift
      ;;
    --batch)
      require_value "$1" "${2:-}"
      BATCH="$2"
      shift 2
      ;;
    --batch=*)
      BATCH="${1#*=}"
      shift
      ;;
    --lr)
      require_value "$1" "${2:-}"
      LR="$2"
      shift 2
      ;;
    --lr=*)
      require_value "--lr" "${1#*=}"
      LR="${1#*=}"
      shift
      ;;
    --resume)
      if (($# < 2)) || [[ "${2}" == --* ]]; then
        die "--resume requires an explicit step_N checkpoint path"
      fi
      RESUME=true
      RESUME_FROM="$2"
      shift 2
      ;;
    --resume=*)
      require_value "--resume" "${1#*=}"
      RESUME=true
      RESUME_FROM="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

case "${DATASET}" in
  small)
    DATA_CONFIG_RELATIVE="configs/data/small.yaml"
    TRAIN_DAY_RANGE="${TRAIN_DAY_RANGE:-01-07}"
    EVAL_DAY_RANGE="${EVAL_DAY_RANGE:-08-08}"
    ;;
  large)
    DATA_CONFIG_RELATIVE="configs/data/large.yaml"
    TRAIN_DAY_RANGE="${TRAIN_DAY_RANGE:-01-34}"
    EVAL_DAY_RANGE="${EVAL_DAY_RANGE:-35-35}"
    ;;
  hotuser)
    [[ -z "${TRAIN_DAY_RANGE}" && -z "${EVAL_DAY_RANGE}" ]] || \
      die "HotUser uses train/valid splits; day ranges are not supported"
    DATA_CONFIG_RELATIVE="configs/data/hotuser.yaml"
    ;;
  *)
    die "--dataset is required and must be small, large, or hotuser"
    ;;
esac
if [[ -n "${DATA_CONFIG_OVERRIDE}" ]]; then
  [[ "${DATA_CONFIG_OVERRIDE}" != /* ]] || die "--data-config must be relative"
  DATA_CONFIG_RELATIVE="${DATA_CONFIG_OVERRIDE}"
fi
case "${MODE}" in
  train)
    DO_TRAIN=1
    DO_EVAL=0
    ;;
  eval)
    DO_TRAIN=0
    DO_EVAL=1
    ;;
  all)
    DO_TRAIN=1
    DO_EVAL=1
    ;;
  *)
    die "Unsupported mode: ${MODE}; expected train, eval, or all"
    ;;
esac

if [[ "${RESUME}" == true && "${DO_TRAIN}" != 1 ]]; then
  die "--resume requires --mode train or --mode all"
fi
if [[ -n "${INIT_FROM}" && "${DO_TRAIN}" != 1 ]]; then
  die "--init-from requires --mode train or --mode all"
fi
if [[ -n "${LR}" && "${DO_TRAIN}" != 1 ]]; then
  die "--lr requires --mode train or --mode all"
fi
if [[ -n "${WEIGHT}" && "${DO_EVAL}" != 1 ]]; then
  die "--weight requires --mode eval or --mode all"
fi
if [[ "${RESUME}" == true && -n "${INIT_FROM}" ]]; then
  die "--resume and --init-from cannot be combined"
fi
[[ -n "${EXP_NAME}" ]] || die "--mode ${MODE} requires --exp EXP_NAME"
if [[ -n "${WEIGHT}" && "${WEIGHT}" != */last ]]; then
  die "--weight must point to a model checkpoint directory ending in /last"
fi

[[ "${EXP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  die "--exp must start with a letter or digit and contain only letters, digits, dots, underscores, or hyphens"
[[ "${CONFIG}" != /* ]] || die "--config must be relative"
[[ "${BATCH}" =~ ^[1-9][0-9]*$ ]] || die "--batch must be a positive integer"
if [[ -n "${LR}" ]]; then
  [[ "${LR}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
    die "--lr must be a positive finite number"
  awk -v value="${LR}" 'BEGIN { exit !(value > 0) }' || \
    die "--lr must be a positive finite number"
fi
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be a non-negative integer"

# =============================================================================
# Current Python environment and derived paths.
# =============================================================================

MODEL_CONFIG="${PROJECT_ROOT}/${CONFIG}"
DATA_CONFIG="${PROJECT_ROOT}/${DATA_CONFIG_RELATIVE}"
MODEL_OUTPUT_DIR=""
OUTPUT_DIR=""

export LOGURU_LEVEL="${LOGURU_LEVEL:-ERROR}"

require_distributed_env() {
  : "${WORLD_SIZE:?WORLD_SIZE is required for train/eval actions}"
  : "${RANK:?RANK is required for train/eval actions}"
  : "${MASTER_ADDR:?MASTER_ADDR is required for train/eval actions}"
  : "${MASTER_PORT:?MASTER_PORT is required for train/eval actions}"

  [[ "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "WORLD_SIZE must be positive"
  [[ "${RANK}" =~ ^[0-9]+$ ]] || die "RANK must be non-negative"
}

prepare_runtime() {
  [[ -d "${PROJECT_ROOT}" ]] || die "Project root does not exist: ${PROJECT_ROOT}"
  [[ -f "${MODEL_CONFIG}" ]] || die "Model config does not exist: ${MODEL_CONFIG}"
  [[ -f "${DATA_CONFIG}" ]] || die "Data config does not exist: ${DATA_CONFIG}"
  [[ -n "${DATA_ROOT}" && -d "${DATA_ROOT}" ]] || die "Set --data-root or DATA_ROOT to an existing data directory"
  DATA_ROOT="$(cd "${DATA_ROOT}" && pwd -P)"
  cd "${PROJECT_ROOT}"
  command -v python >/dev/null 2>&1 || die "python is not available; activate your Python environment first"
}

resolve_model_name() {
  MODEL="$(python - "${MODEL_CONFIG}" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit(f"Model config must contain a YAML mapping: {path}")
model = payload.get("model")
if not isinstance(model, dict):
    raise SystemExit(f"Model config must contain a model mapping: {path}")
name = model.get("name")
if not isinstance(name, str) or not name.strip():
    raise SystemExit(f"Model config model.name must be a non-empty string: {path}")
print(name.strip())
PY
)"
  [[ "${MODEL}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "Invalid model.name in ${MODEL_CONFIG}: ${MODEL}"
  MODEL_OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
}

torchrun_prefix() {
  python -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes "${WORLD_SIZE}" \
    --node_rank "${RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "$@"
}

resolve_experiment_paths() {
  OUTPUT_DIR="${MODEL_OUTPUT_DIR}/${EXP_NAME}"
}

append_day_data_paths() {
  local -n target_args="$1"
  local range="$2"
  local label="$3"
  [[ "${range}" =~ ^([0-9]{1,2})-([0-9]{1,2})$ ]] || \
    die "${label} must use a day range such as 01-07: ${range}"
  local start=$((10#${BASH_REMATCH[1]}))
  local end=$((10#${BASH_REMATCH[2]}))
  ((start >= 1 && end >= start)) || die "Invalid ${label}: ${range}"

  local day day_name directory
  for ((day = start; day <= end; day++)); do
    printf -v day_name 'DAY%02d' "${day}"
    directory="${DATA_ROOT}/dt=${day_name}"
    [[ -d "${directory}" ]] || die "Missing partition: ${directory}"
    compgen -G "${directory}/*.parquet" > /dev/null || \
      die "No Parquet files found in ${directory}"
    target_args+=(--data-path "${directory}/*.parquet")
  done
}

preflight_day_partitions() {
  [[ "${DATASET}" != hotuser ]] || return 0
  local -a checked_paths=()
  if ((DO_TRAIN == 1)); then
    append_day_data_paths checked_paths "${TRAIN_DAY_RANGE}" TRAIN_DAY_RANGE
  fi
  if ((DO_EVAL == 1)); then
    append_day_data_paths checked_paths "${EVAL_DAY_RANGE}" EVAL_DAY_RANGE
  fi
}

run_train() {
  local resume_checkpoint="${1:-}"
  local -a train_args=(
    scripts/train.py
    --device cuda
    --num-workers "${NUM_WORKERS}"
    --config "${MODEL_CONFIG}"
    --data-config "${DATA_CONFIG}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH}"
  )
  if [[ "${DATASET}" == hotuser ]]; then
    train_args+=(--data-root "${DATA_ROOT}" --split train)
    if [[ -n "${DATA_MANIFEST}" ]]; then
      train_args+=(--data-manifest "${DATA_MANIFEST}")
    fi
  else
    append_day_data_paths train_args "${TRAIN_DAY_RANGE}" TRAIN_DAY_RANGE
  fi
  if [[ -n "${resume_checkpoint}" ]]; then
    [[ -d "${resume_checkpoint}" ]] || die \
      "Resume checkpoint does not exist: ${resume_checkpoint}"
    [[ "$(basename "${resume_checkpoint}")" =~ ^step_[0-9]+$ ]] || die \
      "--resume must point to a full step_N checkpoint: ${resume_checkpoint}"
    train_args+=(--resume-from "${resume_checkpoint}")
  fi
  if [[ -n "${INIT_FROM}" ]]; then
    [[ -d "${INIT_FROM}" ]] || die \
      "Initialization checkpoint does not exist: ${INIT_FROM}"
    train_args+=(--init-from "${INIT_FROM}")
  fi
  if [[ -n "${LR}" ]]; then
    train_args+=(--lr "${LR}")
  fi
  torchrun_prefix "${train_args[@]}"
}

run_eval() {
  local checkpoint="${WEIGHT:-${OUTPUT_DIR}/checkpoints/last}"
  [[ -d "${checkpoint}" ]] || die "Completed checkpoint is missing: ${checkpoint}"

  local -a eval_args=(
    scripts/eval.py
    --device cuda
    --num-workers "${NUM_WORKERS}"
    --config "${MODEL_CONFIG}"
    --data-config "${DATA_CONFIG}"
    --checkpoint "${checkpoint}"
    --output-dir "${OUTPUT_DIR}"
    --global-eval
    --batch-size "${BATCH}"
  )
  if [[ "${DATASET}" == hotuser ]]; then
    eval_args+=(--data-root "${DATA_ROOT}" --split valid)
    if [[ -n "${DATA_MANIFEST}" ]]; then
      eval_args+=(--data-manifest "${DATA_MANIFEST}")
    fi
  else
    append_day_data_paths eval_args "${EVAL_DAY_RANGE}" EVAL_DAY_RANGE
  fi
  torchrun_prefix "${eval_args[@]}"
}

print_run_config() {
  local visible_train_day_range=none
  local visible_eval_day_range=none
  if [[ "${DATASET}" != hotuser ]]; then
    visible_train_day_range="${TRAIN_DAY_RANGE}"
    visible_eval_day_range="${EVAL_DAY_RANGE}"
  fi
  cat <<EOF
experiment mode=${MODE}
dataset=${DATASET}
resume=${RESUME}
resume_from=${RESUME_FROM:-none}
init_from=${INIT_FROM:-none}
model=${MODEL}
experiment_name=${EXP_NAME}
config=${MODEL_CONFIG}
data_config=${DATA_CONFIG}
data_root=${DATA_ROOT}
train_day_range=${visible_train_day_range}
eval_day_range=${visible_eval_day_range}
data_manifest=${DATA_MANIFEST:-none}
batch=${BATCH}
lr_override=${LR:-none}
weight_override=${WEIGHT:-none}
output_dir=${OUTPUT_DIR}
EOF
}

require_distributed_env
prepare_runtime
resolve_model_name
resolve_experiment_paths
preflight_day_partitions

print_run_config

if ((DO_TRAIN == 1)); then
  if [[ "${RESUME}" == true ]]; then
    run_train "${RESUME_FROM}"
  else
    run_train
  fi
fi
if ((DO_EVAL == 1)); then
  run_eval
fi
