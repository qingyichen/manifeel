#!/bin/bash
# Re-run campaign 2026-07-02: bulb / pih / usb x {vision, tacff_cnn, tacff_resnet}.
#
# Why re-run: (1) top-k checkpointing by rollout success (test_mean_score) was only
# just enabled — old runs kept only the last, most-overfit epochs; (2) the widened
# 512-d ConvPoolingHead replaces the undersized 128-d tactile head; (3) the old usb
# tacff run trained on the dead legacy TacFF normalizer. 200 epochs (peaks were all
# <= ep200), seed 0, 50 demos, sequential on the single local GPU.
#
# INCLUDE_DEADNORM=1 adds the faulty-normalizer ResNet ablation (task visff_wrist_rgb,
# tagged resnet_deadnorm) per task: 12 runs instead of 9.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE_DEADNORM=${INCLUDE_DEADNORM:-0}
DATE_TAG=0702
NUM_EPOCH=${NUM_EPOCH:-200}

# task_key | dataset | isaacgym cfg | action shape ('' = task default 6) | wandb project
TASKS=(
  "pih|data/pih_quan_June06|isaacgym_config.yaml||dp_pih"
  "bulb|data/bulb_quan_Sep19|isaacgym_config_bulb.yaml|7|dp_bulb"
  "usb|data/usb_quan_Aug05|isaacgym_config_usb.yaml||dp_usb"
)

# input_type | task_name | env-tag arch suffix ('' for vision)
ARMS=(
  "vision|vision_wrist|"
  "tacff|visff_wrist|cnn"
  "tacff|visff_wrist_resnet|resnet"
)
if [[ "${INCLUDE_DEADNORM}" == "1" ]]; then
  ARMS+=("tacff|visff_wrist_rgb|resnet_deadnorm")
fi

for task_spec in "${TASKS[@]}"; do
  IFS='|' read -r TKEY DSET IGCFG ASHAPE PROJ <<< "${task_spec}"
  for arm_spec in "${ARMS[@]}"; do
    IFS='|' read -r ITYPE TNAME ASUFFIX <<< "${arm_spec}"
    ENV_TAG="${TKEY}${ASUFFIX:+_${ASUFFIX}}_${DATE_TAG}"
    echo "=============================================================="
    echo ">>> $(date '+%F %T')  ${TKEY} / ${ITYPE}${ASUFFIX:+ (${ASUFFIX})}  epochs=${NUM_EPOCH}"
    echo "=============================================================="
    DATASET_PATH="${DSET}" \
      ISAACGYM_CONFIG="${IGCFG}" \
      NUM_EPOCH="${NUM_EPOCH}" \
      ENV_TAG="${ENV_TAG}" \
      TASK_NAME="${TNAME}" \
      INPUT_TYPE="${ITYPE}" \
      ACTION_SHAPE="${ASHAPE}" \
      LOG_NAME="${PROJ}" \
      bash "${SCRIPT_DIR}/run_local.sh"
  done
done
echo ">>> $(date '+%F %T')  campaign complete"
