#!/bin/bash
set -euo pipefail

# -------------------------
# User-editable parameters
# -------------------------
SEED=${SEED:-0}
NUM_DEMOS=${NUM_DEMOS:-50}
NUM_EPOCH=${NUM_EPOCH:-1000}

DATASET_PATH=${DATASET_PATH:-data/usb_quan_Aug05}
ISAACGYM_CONFIG=${ISAACGYM_CONFIG:-isaacgym_config_usb.yaml}

# Workspace config. Default routes TacFF through the ResNet image encoder.
# Set to train_tacff_diffusion.yaml to route TacFF through the ff_head MLP instead.
CONFIG_NAME=${CONFIG_NAME:-train_diffusion_workspace.yaml}

TASK_NAME=${TASK_NAME:-vision_wrist}  # vision_wrist | vistac_wrist | visff_wrist | vision_front | vistac_front | visff_front
INPUT_TYPE=${INPUT_TYPE:-vision}   # vision | vistac | tacff
ENV_TAG=${ENV_TAG:-usb_wrist_0805}
LOG_NAME=${LOG_NAME:-dp_usb_tacff}

# Optional Hydra overrides
IMAGENET_NORM=${IMAGENET_NORM:-false}   # set true for tacff runs if needed
ACTION_SHAPE=${ACTION_SHAPE:-}          # set to 7 for gripper tasks, leave empty otherwise
USE_CONV_FF=${USE_CONV_FF:-}            # legacy: route TacFF through the CNN ff_head (train_tacff_diffusion.yaml only)
# TacFF obs-encoder architecture for the default image workspace (train_diffusion_workspace.yaml
# + TASK_NAME=visff_wrist). Selects the small dedicated tactile encoder; TacFF is per-channel
# normalized and skips ResNet/imagenet_norm. Values: cnn | mlp. Leave empty to use the config default (cnn).
FF_ARCH=${FF_ARCH:-}

# Container file (default assumes you run from repo root and built it there)
CONTAINER_FILE=${CONTAINER_FILE:-manifeel.sif}

# -------------------------
# Derived names
# -------------------------
# Task group for grouping outputs (e.g. pih, gear, bulb). Defaults to the leading
# token of ENV_TAG (pih_resnet_0629 -> pih); override with TASK_TAG if needed.
TASK_TAG=${TASK_TAG:-${ENV_TAG%%_*}}
EXP_NAME="${INPUT_TYPE}_${ENV_TAG}_${NUM_DEMOS}"
OUT_DIR="data/outputs/${TASK_TAG}/${EXP_NAME}/${SEED}"
# wandb run name (logging.name = ${now}_${exp_name}). Drop the trailing date token
# from ENV_TAG (pih_ffcnn_0629 -> pih_ffcnn) and the demo count so runs group cleanly
# in wandb; OUT_DIR still uses the full EXP_NAME to keep runs distinct on disk.
RUN_NAME="${INPUT_TYPE}_${ENV_TAG%_[0-9]*}"

# Repo root resolution (script works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------
# Build Hydra args
# -------------------------
HYDRA_ARGS=(
  "--config-name=${CONFIG_NAME}"
  "task=${TASK_NAME}"
  "exp_name=${RUN_NAME}"
  "dataset_path=${DATASET_PATH}"
  "isaacgym_cfg_name=${ISAACGYM_CONFIG}"
  "training.seed=${SEED}"
  "training.num_epochs=${NUM_EPOCH}"
  "task.dataset.max_train_episodes=${NUM_DEMOS}"
  "hydra.run.dir=${OUT_DIR}"
  "logging.project=${LOG_NAME}"
)

if [[ "${IMAGENET_NORM}" == "true" ]]; then
  HYDRA_ARGS+=("policy.obs_encoder.imagenet_norm=True")
fi

if [[ -n "${ACTION_SHAPE}" ]]; then
  HYDRA_ARGS+=("task.shape_meta.action.shape=[${ACTION_SHAPE}]")
fi

if [[ -n "${USE_CONV_FF}" ]]; then
  HYDRA_ARGS+=("policy.use_conv_ff=${USE_CONV_FF}")
fi

if [[ -n "${FF_ARCH}" ]]; then
  # tactile encoder architecture: cnn | mlp (train_diffusion_workspace.yaml only)
  HYDRA_ARGS+=("policy.obs_encoder.tactile_arch=${FF_ARCH}")
fi

# Append the TacFF encoder architecture to task.name so wandb run names distinguish
# the route: *_resnet tasks push TacFF through the ResNet18 image path with the
# correct per-channel normalizer -> "resnet"; *_rgb keeps the legacy dead [0,1]
# image normalizer (kept on purpose as the faulty-baseline ablation) ->
# "resnet_deadnorm"; otherwise the dedicated tactile head is used -> tactile_arch
# (FF_ARCH, config default cnn). e.g. train_diffusion_unet_image_vision_tacff_cnn.
# Vision-only / vistac runs are untouched. The base name is read from the task yaml
# (visff_wrist and visff_front differ), so this stays correct if task names change.
if [[ "${INPUT_TYPE}" == "tacff" ]]; then
  if [[ "${TASK_NAME}" == *_resnet ]]; then
    ARCH_TAG="resnet"
  elif [[ "${TASK_NAME}" == *_rgb ]]; then
    ARCH_TAG="resnet_deadnorm"
  else
    ARCH_TAG="${FF_ARCH:-cnn}"
  fi
  TASK_BASE_NAME=$(awk '/^name:/{print $2; exit}' "${REPO_ROOT}/manifeel/config/task/${TASK_NAME}.yaml")
  HYDRA_ARGS+=("task.name=${TASK_BASE_NAME}_${ARCH_TAG}")
fi

# -------------------------
# Run inside Apptainer
# -------------------------
# NB: non-interactive shell (bash -c, not -ic). An interactive shell turns on job
# control and tries to grab the terminal foreground group; as a nested descendant
# it instead gets SIGTTOU and the job auto-stops. conda is sourced explicitly from
# the bind-mounted host install (the container itself has no conda).
apptainer exec --nv --cleanenv --env LD_PRELOAD= "${REPO_ROOT}/${CONTAINER_FILE}" bash -c "
  set -e
  source /home/qc/miniconda3/etc/profile.d/conda.sh
  conda activate manifeel
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  export PYTHONIOENCODING=utf-8
  cd '${REPO_ROOT}'
  python train.py ${HYDRA_ARGS[*]}
" < /dev/null