#!/usr/bin/env bash
#
# Train PatchCore and evaluate per MVTec class. Run from the repository root,
# or this script will cd there automatically.
#
# Parameters (CLI flags override environment defaults):
#   EXP_NAME         -- log_project name and evaluated output folder basename
#   TARGET_EMBED     -- --target_embed_dimension
#   CORESET_P        -- coreset sampling percentage (-p for approx_greedy_coreset)
#   DATASET_PATH     -- path to MVTec root directory
#   CLASS_NAMES      -- space-separated class list if not using --classes
#   GPU, SEED        -- passed through to both Python entrypoints
#   LOG_GROUP        -- defaults to anomaly_challenge (matches your template)
#
# Examples:
#   ./train_eval_mvtec_classes.sh \
#     --exp-name my_run --target-embed 1024 --coreset-p 0.1 \
#     --dataset-path /data/mvtec --classes bottle,cable,zipper
#
#   CLASS_NAMES="bottle screw" DATASET_PATH=/data/mvtec \
#     ./train_eval_mvtec_classes.sh --exp-name run_a --target-embed 512 --coreset-p 0.05
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Defaults (override with flags or env)
: "${EXP_NAME:=my_experiment}"
: "${TARGET_EMBED:=1024}"
: "${CORESET_P:=0.1}"
: "${DATASET_PATH:=/path/to/mvtec}"
: "${GPU:=0}"
: "${SEED:=0}"
: "${LOG_GROUP:=anomaly_challenge}"

EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-}"
TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Train and evaluate PatchCore for each MVTec class (one train + one eval per class).

Options:
  --exp-name NAME        Experiment / log_project name (default: ${EXP_NAME})
  --target-embed N       Target embedding dimension (default: ${TARGET_EMBED})
  --coreset-p P          Coreset percentage, e.g. 0.1 (default: ${CORESET_P})
  --dataset-path PATH    MVTec dataset root (default: ${DATASET_PATH})
  --classes LIST         Comma-separated class names, e.g. bottle,cable,zipper
  --gpu ID               GPU id (default: ${GPU})
  --seed N               Random seed (default: ${SEED})
  --log-group NAME       Logger group folder (default: ${LOG_GROUP})
  --eval-dir PATH        Evaluation output root (default: evaluated_results/<exp_name>)
  --train-results PATH   Training results root first arg (default: results)
  -h, --help             Show this help

If --classes is omitted, classes are taken from CLASS_NAMES (space-separated).

MVTec class names include: bottle cable capsule carpet grid hazelnut leather
metal_nut pill screw tile toothbrush transistor wood zipper
EOF
}

CLASSES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    --target-embed)
      TARGET_EMBED="$2"
      shift 2
      ;;
    --coreset-p)
      CORESET_P="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --classes)
      IFS=',' read -ra CLASSES <<< "$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --log-group)
      LOG_GROUP="$2"
      shift 2
      ;;
    --eval-dir)
      EVAL_RESULTS_DIR="$2"
      shift 2
      ;;
    --train-results)
      TRAIN_RESULTS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#CLASSES[@]} -eq 0 ]]; then
  if [[ -n "${CLASS_NAMES:-}" ]]; then
    # shellcheck disable=SC2206
    CLASSES=($CLASS_NAMES)
  fi
fi

if [[ ${#CLASSES[@]} -eq 0 ]]; then
  echo "error: no classes specified. Use --classes a,b,c or set CLASS_NAMES." >&2
  exit 1
fi

[[ -z "${TRAIN_RESULTS_DIR}" ]] && TRAIN_RESULTS_DIR="results"
[[ -z "${EVAL_RESULTS_DIR}" ]] && EVAL_RESULTS_DIR="evaluated_results/${EXP_NAME}"

NUM_CLASSES=${#CLASSES[@]}
IDX=0

print_banner() {
  local title=$1
  echo ""
  echo "================================================================================"
  echo "  ${title}"
  echo "================================================================================"
  echo ""
}

print_banner "PatchCore MVTec sweep"
echo "  exp_name           = ${EXP_NAME}"
echo "  target_embed       = ${TARGET_EMBED}"
echo "  coreset_p          = ${CORESET_P}"
echo "  dataset_path       = ${DATASET_PATH}"
echo "  log_group          = ${LOG_GROUP}"
echo "  train_results_dir  = ${TRAIN_RESULTS_DIR}"
echo "  eval_results_dir   = ${EVAL_RESULTS_DIR}"
echo "  gpu / seed         = ${GPU} / ${SEED}"
echo "  classes (${NUM_CLASSES}): ${CLASSES[*]}"
echo ""

for class_name in "${CLASSES[@]}"; do
  IDX=$((IDX + 1))

  print_banner "Class ${IDX}/${NUM_CLASSES}: ${class_name} — TRAINING"
  echo "Starting run_patchcore.py for class '${class_name}'..."

  python bin/run_patchcore.py \
    --gpu "${GPU}" --seed "${SEED}" \
    --save_patchcore_model \
    --log_group "${LOG_GROUP}" \
    --log_project "${EXP_NAME}" \
    "${TRAIN_RESULTS_DIR}" \
    patch_core \
      -b wideresnet50 \
      -le layer2 -le layer3 \
      --pretrain_embed_dimension 1024 \
      --target_embed_dimension "${TARGET_EMBED}" \
      --anomaly_scorer_num_nn 1 \
      --patchsize 3 \
    sampler \
      -p "${CORESET_P}" approx_greedy_coreset \
    dataset \
      --resize 224 --imagesize 224 \
      -d "${class_name}" \
      mvtec "${DATASET_PATH}"

  echo ""
  echo ">>> Finished training for class '${class_name}' (${IDX}/${NUM_CLASSES})."
  echo ""

  print_banner "Class ${IDX}/${NUM_CLASSES}: ${class_name} — EVALUATION"
  echo "Starting load_and_evaluate_patchcore.py for class '${class_name}'..."
  echo "Model path: ${TRAIN_RESULTS_DIR}/${LOG_GROUP}/${EXP_NAME}/models/mvtec_${class_name}"

  python bin/load_and_evaluate_patchcore.py \
    --gpu "${GPU}" --seed "${SEED}" \
    --save_segmentation_images \
    "${EVAL_RESULTS_DIR}" \
    patch_core_loader \
      -p "${TRAIN_RESULTS_DIR}/${LOG_GROUP}/${EXP_NAME}/models/mvtec_${class_name}" \
    dataset \
      --resize 224 --imagesize 224 \
      -d "${class_name}" \
      mvtec "${DATASET_PATH}"

  echo ""
  echo ">>> Finished evaluation for class '${class_name}' (${IDX}/${NUM_CLASSES})."
  echo ""
done

print_banner "All done"
echo "Processed ${NUM_CLASSES} class(es). Evaluations saved under: ${EVAL_RESULTS_DIR}"
echo ""
