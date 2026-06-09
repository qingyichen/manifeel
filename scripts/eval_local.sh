#!/bin/bash
set -euo pipefail

# -------------------------
# User-editable parameters
# -------------------------
# Path to checkpoint, relative to REPO_ROOT (or absolute).
# Example: data/outputs/vision_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt
CHECKPOINT=${CHECKPOINT:-}

# Where to write eval videos and eval_log.json, relative to REPO_ROOT.
OUTPUT_DIR=${OUTPUT_DIR:-data/eval_output}

# Hydra config name used during training (same --config-name as run_local.sh).
CFG_NAME=${CFG_NAME:-train_diffusion_workspace.yaml}

# Task config, must match what was used during training.
TASK_NAME=${TASK_NAME:-vision_wrist}  # vision_wrist | vistac_wrist | visff_wrist | ...

# Number of parallel envs to evaluate (overrides task yaml's n_test).
N_TEST=${N_TEST:-22}

# Number of envs to record video for.
N_TEST_VIS=${N_TEST_VIS:-6}

# Set to isaacgym_config_gui.yaml to open the interactive viewer (1 env, needs a display).
ISAACGYM_CONFIG=${ISAACGYM_CONFIG:-isaacgym_config_usb.yaml}

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

echo "Checkpoint : ${CHECKPOINT}"
echo "Output dir : ${REPO_ROOT}/${OUTPUT_DIR}"
echo "Config     : ${CFG_NAME}  task=${TASK_NAME}"
echo "IsaacGym   : ${ISAACGYM_CONFIG}  (headless=$(grep -m1 'headless' "${REPO_ROOT}/manifeel/config/${ISAACGYM_CONFIG}" | awk '{print $2}'))"
echo "n_test=${N_TEST}  n_test_vis=${N_TEST_VIS}"
echo ""

# -------------------------
# Run inside Apptainer
# -------------------------
apptainer exec --nv --cleanenv --env LD_PRELOAD= "${REPO_ROOT}/${CONTAINER_FILE}" bash -ic "
  set -e
  conda activate manifeel
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  export PYTHONIOENCODING=utf-8
  cd '${REPO_ROOT}'
  python eval.py \
    --checkpoint '${CHECKPOINT}' \
    --output_dir '${OUTPUT_DIR}' \
    --cfg_name '${CFG_NAME}' \
    --device '${DEVICE}' \
    --n_test '${N_TEST}' \
    --n_test_vis '${N_TEST_VIS}' \
    --isaacgym_cfg '${ISAACGYM_CONFIG}'
"
