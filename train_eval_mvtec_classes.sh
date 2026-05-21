#!/usr/bin/env bash
#
# Train PatchCore and evaluate per MVTec class. Run from the repository root,
# or this script will cd there automatically.
#
# Storage layout follows run_patchcore.py:create_storage_folder (mode=iterate):
#   {train_results_dir}/{log_project}/{log_group}/{class}/...
# Each class uses LOG_GROUP/<classname> so iterate mode does not bump the folder
# to log_group_0 on the second class (which would save models where eval cannot find them).
# Example: results/anomaly_challenge/my_run/bottle/models/mvtec_bottle/
#
# Evaluation outputs (load_and_evaluate_patchcore): results.csv at {eval_dir}/;
#   per-split artifacts under {eval_dir}/test/{class}/ and optional {eval_dir}/validation/{class}/.
# Parameters (CLI flags override environment defaults):
#   LOG_PROJECT      -- outer folder under results (default: anomaly_challenge)
#   LOG_GROUP        -- inner run folder; set via --log-group or env LOG_GROUP (required)
#   TARGET_EMBED     -- --target_embed_dimension
#   CORESET_P        -- coreset sampling percentage (-p for approx_greedy_coreset)
#   DATASET_PATH     -- path to MVTec root directory
#   CLASS_NAMES      -- space-separated class list if not using --classes
#   NUM_WORKERS      -- PyTorch DataLoader workers for train/eval (default: 8)
#   FAISS_NUM_WORKERS -- FAISS thread count for nearest-neighbor search (default: 8)
#   GPU, SEED        -- passed through to both Python entrypoints
#   PUSHBULLET_API_KEY -- access token (required if you pass --notify)
#
# Examples:
#   ./train_eval_mvtec_classes.sh \
#     --log-group my_run --target-embed 1024 --coreset-p 0.1 \
#     --dataset-path /data/mvtec --classes bottle,cable,zipper
#
#   LOG_GROUP=run_a CLASS_NAMES="bottle screw" DATASET_PATH=/data/mvtec \
#     ./train_eval_mvtec_classes.sh --target-embed 512 --coreset-p 0.05
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Defaults (override with flags or env). LOG_GROUP is required (see below).
: "${LOG_PROJECT:=anomaly_challenge}"
: "${TARGET_EMBED:=1024}"
: "${CORESET_P:=0.1}"
: "${DATASET_PATH:=/path/to/mvtec}"
: "${GPU:=0}"
: "${SEED:=0}"
: "${NUM_WORKERS:=8}"
: "${FAISS_NUM_WORKERS:=8}"
: "${EVAL_SPLITS:=both}"

EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-}"
TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-}"

NOTIFY_PUSHBULLET=0
NOTIFY_NOTE_TITLE="PatchCore"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Train and evaluate PatchCore for each MVTec class (one train + one eval per class).

Options:
  --log-group NAME       Run folder inside log_project (required unless LOG_GROUP env is set)
  --log-project NAME     Outer folder under train results (default: ${LOG_PROJECT})
  --target-embed N       Target embedding dimension (default: ${TARGET_EMBED})
  --coreset-p P          Coreset percentage, e.g. 0.1 (default: ${CORESET_P})
  --dataset-path PATH    MVTec dataset root (default: ${DATASET_PATH})
  --classes LIST         Comma-separated class names, e.g. bottle,cable,zipper
  --gpu ID               GPU id (default: ${GPU})
  --seed N               Random seed (default: ${SEED})
  --num-workers N        DataLoader worker processes (default: ${NUM_WORKERS})
  --faiss-num-workers N  FAISS threads for NN search (default: ${FAISS_NUM_WORKERS})
  --eval-splits SPLITS   Which splits to evaluate: test, val, or both (default: ${EVAL_SPLITS})
  --eval-dir PATH        Evaluation root; outputs go under PATH/<class>/ (default: evaluated_results/<log_group>)
  --train-results PATH   Training results root first arg (default: results)
  --notify               Send Pushbullet notes for each phase (needs PUSHBULLET_API_KEY)
  --notify-title TEXT    Push note title when using --notify (default: PatchCore)
  -h, --help             Show this help

If --classes is omitted, classes are taken from CLASS_NAMES (space-separated).

MVTec class names include: bottle cable capsule carpet grid hazelnut leather
metal_nut pill screw tile toothbrush transistor wood zipper
EOF
}

CLASSES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-group)
      LOG_GROUP="$2"
      shift 2
      ;;
    --log-project)
      LOG_PROJECT="$2"
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
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --faiss-num-workers)
      FAISS_NUM_WORKERS="$2"
      shift 2
      ;;
    --eval-splits)
      EVAL_SPLITS="$2"
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
    --notify)
      NOTIFY_PUSHBULLET=1
      shift
      ;;
    --notify-title)
      NOTIFY_NOTE_TITLE="$2"
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

if [[ -z "${LOG_GROUP:-}" ]]; then
  echo "error: set LOG_GROUP (environment) or pass --log-group (inner run folder under .../results/<log_project>/)." >&2
  exit 1
fi

if [[ "${EVAL_SPLITS}" != "test" && "${EVAL_SPLITS}" != "val" && "${EVAL_SPLITS}" != "both" ]]; then
  echo "error: --eval-splits must be one of: test, val, both (got '${EVAL_SPLITS}')." >&2
  exit 1
fi

if [[ "${NOTIFY_PUSHBULLET}" -eq 1 ]]; then
  if [[ -z "${PUSHBULLET_API_KEY:-}" ]]; then
    echo "error: --notify requires PUSHBULLET_API_KEY in the environment." >&2
    exit 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "error: --notify requires curl on PATH." >&2
    exit 1
  fi
fi

[[ -z "${TRAIN_RESULTS_DIR}" ]] && TRAIN_RESULTS_DIR="results"
[[ -z "${EVAL_RESULTS_DIR}" ]] && EVAL_RESULTS_DIR="evaluated_results/${LOG_GROUP}"

NUM_CLASSES=${#CLASSES[@]}
IDX=0

# Send a Pushbullet note (no extra Python deps; uses curl). Skipped unless --notify.
_pushbullet_note() {
  local title=$1
  local body=$2
  [[ "${NOTIFY_PUSHBULLET}" -eq 1 ]] || return 0
  local json
  json="$(
    NOTIFY_JSON_TITLE="$title" NOTIFY_JSON_BODY="$body" python3 -c '
import json, os
print(json.dumps({
    "type": "note",
    "title": os.environ["NOTIFY_JSON_TITLE"],
    "body": os.environ["NOTIFY_JSON_BODY"],
}))
'
  )" || {
    echo "warning: could not build Pushbullet JSON payload" >&2
    return 0
  }
  if ! curl -sfS --connect-timeout 15 \
    -u "${PUSHBULLET_API_KEY}:" \
    -H "Content-Type: application/json" \
    -X POST "https://api.pushbullet.com/v2/pushes" \
    -d "${json}" >/dev/null; then
    echo "warning: Pushbullet request failed (title: ${title})" >&2
  fi
}

# On exit, report success or failure (so notifications still fire if a python step fails under set -e).
_pushbullet_exit_trap() {
  local ec=$?
  if [[ "${NOTIFY_PUSHBULLET}" -eq 1 ]]; then
    if [[ "${ec}" -eq 0 ]]; then
      _pushbullet_note "${NOTIFY_NOTE_TITLE}" "Training finished successfully"
    else
      _pushbullet_note "${NOTIFY_NOTE_TITLE}" "Training failed with code ${ec}"
    fi
  fi
  exit "${ec}"
}
trap "_pushbullet_exit_trap" EXIT

print_banner() {
  local title=$1
  echo ""
  echo "================================================================================"
  echo "  ${title}"
  echo "================================================================================"
  echo ""
}

print_banner "PatchCore MVTec sweep"
echo "  log_project        = ${LOG_PROJECT}"
echo "  log_group          = ${LOG_GROUP}"
echo "  target_embed       = ${TARGET_EMBED}"
echo "  coreset_p          = ${CORESET_P}"
echo "  dataset_path       = ${DATASET_PATH}"
echo "  train_save_pattern = ${TRAIN_RESULTS_DIR}/${LOG_PROJECT}/${LOG_GROUP}/<class>/"
echo "  train_results_dir  = ${TRAIN_RESULTS_DIR}"
echo "  eval_save_pattern  = ${EVAL_RESULTS_DIR}/<class>/"
echo "  gpu / seed         = ${GPU} / ${SEED}"
echo "  num_workers        = ${NUM_WORKERS}"
echo "  faiss_num_workers  = ${FAISS_NUM_WORKERS}"
echo "  eval_splits        = ${EVAL_SPLITS}"
echo "  classes (${NUM_CLASSES}): ${CLASSES[*]}"
echo ""

if [[ "${NOTIFY_PUSHBULLET}" -eq 1 ]]; then
  echo "  pushbullet notify   = on (title: ${NOTIFY_NOTE_TITLE})"
  echo ""
  _pushbullet_note "${NOTIFY_NOTE_TITLE}" \
    "PatchCore sweep started (log_group=${LOG_GROUP}, classes=${NUM_CLASSES})"
fi

for class_name in "${CLASSES[@]}"; do
  IDX=$((IDX + 1))
  # Nested path: iterate avoids collision when the parent experiment folder exists.
  CLASS_LOG_GROUP="${LOG_GROUP}/${class_name}"

  _TRAIN_PHASE="Class ${IDX}/${NUM_CLASSES}: ${class_name} — TRAINING"
  print_banner "${_TRAIN_PHASE}"
  _pushbullet_note "${NOTIFY_NOTE_TITLE}" "${_TRAIN_PHASE}"
  echo "Starting run_patchcore.py for class '${class_name}'..."

  python bin/run_patchcore.py \
    --gpu "${GPU}" --seed "${SEED}" \
    --save_patchcore_model \
    --log_group "${CLASS_LOG_GROUP}" \
    --log_project "${LOG_PROJECT}" \
    "${TRAIN_RESULTS_DIR}" \
    patch_core \
      -b wideresnet50 \
      -le layer2 -le layer3 \
      --pretrain_embed_dimension 1024 \
      --target_embed_dimension "${TARGET_EMBED}" \
      --anomaly_scorer_num_nn 1 \
      --patchsize 3 \
      --faiss_num_workers "${FAISS_NUM_WORKERS}" \
    sampler \
      -p "${CORESET_P}" approx_greedy_coreset \
    dataset \
      --resize 224 --imagesize 224 \
      --num_workers "${NUM_WORKERS}" \
      -d "${class_name}" \
      mvtec "${DATASET_PATH}"

  echo ""
  echo ">>> Finished training for class '${class_name}' (${IDX}/${NUM_CLASSES})."
  echo ""

  CLASS_EVAL_DIR="${EVAL_RESULTS_DIR}/${class_name}"

  _EVAL_PHASE="Class ${IDX}/${NUM_CLASSES}: ${class_name} — EVALUATION"
  print_banner "${_EVAL_PHASE}"
  _pushbullet_note "${NOTIFY_NOTE_TITLE}" "${_EVAL_PHASE}"
  echo "Starting load_and_evaluate_patchcore.py for class '${class_name}'..."
  echo "Model path: ${TRAIN_RESULTS_DIR}/${LOG_PROJECT}/${CLASS_LOG_GROUP}/models/mvtec_${class_name}"
  echo "Eval output dir: ${CLASS_EVAL_DIR}"

  python bin/load_and_evaluate_patchcore.py \
    --gpu "${GPU}" --seed "${SEED}" \
    --save_segmentation_images \
    --eval_splits "${EVAL_SPLITS}" \
    "${CLASS_EVAL_DIR}" \
    patch_core_loader \
      -p "${TRAIN_RESULTS_DIR}/${LOG_PROJECT}/${CLASS_LOG_GROUP}/models/mvtec_${class_name}" \
      --faiss_num_workers "${FAISS_NUM_WORKERS}" \
    dataset \
      --resize 224 --imagesize 224 \
      --num_workers "${NUM_WORKERS}" \
      -d "${class_name}" \
      mvtec "${DATASET_PATH}"

  echo ""
  echo ">>> Finished evaluation for class '${class_name}' (${IDX}/${NUM_CLASSES})."
  echo ""
done

print_banner "All done"
echo "Processed ${NUM_CLASSES} class(es). Evaluations saved under: ${EVAL_RESULTS_DIR}/<class>/"
echo ""
