#!/bin/bash
set -euo pipefail

# -------------------------
# User-editable parameters
# -------------------------
# Path to checkpoint, relative to REPO_ROOT (or absolute).
# Example: data/outputs/vision_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt
CHECKPOINT=${CHECKPOINT:-}

# Where to write eval videos and eval_log.json, relative to REPO_ROOT.
# Defaults to data/eval_output/{model_name}/{timestamp} derived from CHECKPOINT.
OUTPUT_DIR=${OUTPUT_DIR:-}

# Hydra config name used during training (same --config-name as run_local.sh).
CFG_NAME=${CFG_NAME:-train_diffusion_workspace.yaml}

# Number of parallel envs to evaluate (leave empty to use the value from the task yaml).
N_TEST=${N_TEST:-}

# Number of envs to record video for (leave empty to use the value from the task yaml).
N_TEST_VIS=${N_TEST_VIS:-}

# Set to isaacgym_config_gui.yaml to open the interactive viewer (1 env, needs a display).
# Leave empty to reuse the isaacgym config from the training run.
ISAACGYM_CONFIG=${ISAACGYM_CONFIG:-}

# cuda device
DEVICE=${DEVICE:-cuda:0}

# Container file
CONTAINER_FILE=${CONTAINER_FILE:-manifeel.sif}

# -------------------------
# Validation
# -------------------------
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT must be set."
  echo "  Example: CHECKPOINT=data/outputs/vision_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt bash scripts/eval_local.sh"
  exit 1
fi

# -------------------------
# Repo root resolution
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Make checkpoint path absolute if it isn't already.
if [[ "${CHECKPOINT}" != /* ]]; then
  CHECKPOINT="${REPO_ROOT}/${CHECKPOINT}"
fi

# Derive model name from the outputs/{model_name}/... portion of the checkpoint path.
MODEL_NAME=$(echo "${CHECKPOINT}" | sed 's|.*/outputs/\([^/]*\)/.*|\1|')
OUTPUT_DIR=${OUTPUT_DIR:-data/eval_output/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)}

echo "Checkpoint : ${CHECKPOINT}"
echo "Output dir : ${REPO_ROOT}/${OUTPUT_DIR}"
echo "Config     : ${CFG_NAME}"
echo ""

# -------------------------
# Run inside Apptainer
# -------------------------
apptainer exec --nv --cleanenv \
  --env LD_PRELOAD= \
  --env DISPLAY="${DISPLAY:-:0}" \
  --env XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}" \
  --bind /tmp/.X11-unix \
  "${REPO_ROOT}/${CONTAINER_FILE}" bash -ic "
  set -e
  conda activate manifeel
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  export PYTHONIOENCODING=utf-8
  cd '${REPO_ROOT}'
  EVAL_ARGS=(
    --checkpoint '${CHECKPOINT}'
    --output_dir '${OUTPUT_DIR}'
    --cfg_name '${CFG_NAME}'
    --device '${DEVICE}'
  )
  [[ -n '${N_TEST}' ]]         && EVAL_ARGS+=(--n_test '${N_TEST}')
  [[ -n '${N_TEST_VIS}' ]]     && EVAL_ARGS+=(--n_test_vis '${N_TEST_VIS}')
  [[ -n '${ISAACGYM_CONFIG}' ]] && EVAL_ARGS+=(--isaacgym_cfg '${ISAACGYM_CONFIG}')
  python eval.py "\${EVAL_ARGS[@]}"
"
