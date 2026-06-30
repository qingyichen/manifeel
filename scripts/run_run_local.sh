# DATASET_PATH=data/bulb_quan_Sep19 \
#     ISAACGYM_CONFIG=isaacgym_config_bulb.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=bulb_0610 \
#     TASK_NAME=vision_wrist \
#     INPUT_TYPE=vision \
#     ACTION_SHAPE=7 \
#     LOG_NAME=dp_bulb \
#     bash scripts/run_local.sh

# DATASET_PATH=data/bulb_quan_Sep19 \
#     ISAACGYM_CONFIG=isaacgym_config_bulb.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=bulb_0610 \
#     TASK_NAME=visff_wrist \
#     INPUT_TYPE=tacff \
#     ACTION_SHAPE=7 \
#     LOG_NAME=dp_bulb \
#     bash scripts/run_local.sh

# DATASET_PATH=data/bulb_quan_Sep19 \
#     ISAACGYM_CONFIG=isaacgym_config_bulb.yaml \
#     CONFIG_NAME=train_tacff_diffusion.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=bulb_ffhead_0610 \
#     TASK_NAME=visff_wrist \
#     INPUT_TYPE=tacff \
#     ACTION_SHAPE=7 \
#     LOG_NAME=dp_bulb \
#     bash scripts/run_local.sh

# ── gear assembly, all three input variants ──────────────────────────────────
# NB: gear numActions=6, so NO ACTION_SHAPE override (matches the *_wrist default [6]).

# 1) vision only (baseline)
# DATASET_PATH=data/gear_quan_Sep15 \
#     ISAACGYM_CONFIG=isaacgym_config_gear.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=gear_0628 \
#     TASK_NAME=vision_wrist \
#     INPUT_TYPE=vision \
#     LOG_NAME=dp_gear \
#     bash scripts/run_local.sh

# # 2) vision + tactile RGB (taxim image through ResNet)
# DATASET_PATH=data/gear_quan_Sep15 \
#     ISAACGYM_CONFIG=isaacgym_config_gear.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=gear_resnet_0628 \
#     TASK_NAME=visff_wrist \
#     INPUT_TYPE=tacff \
#     LOG_NAME=dp_gear \
#     bash scripts/run_local.sh

# # 3) vision + TacFF via ff_head MLP (per-channel min-max norm, no ResNet/imagenet_norm on TacFF)
# DATASET_PATH=data/gear_quan_Sep15 \
#     ISAACGYM_CONFIG=isaacgym_config_gear.yaml \
#     CONFIG_NAME=train_tacff_diffusion.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=gear_ffhead_0628 \
#     TASK_NAME=visff_wrist \
#     INPUT_TYPE=tacff \
#     LOG_NAME=dp_gear \
#     bash scripts/run_local.sh

# ── peg insertion, all three input variants ──────────────────────────────────
# pih dataset action dim = 6 (TacSLTaskInsertion via default isaacgym_config.yaml),
# so NO ACTION_SHAPE override.

# 1) vision only (baseline)
DATASET_PATH=data/pih_quan_June06 \
    ISAACGYM_CONFIG=isaacgym_config.yaml \
    NUM_EPOCH=300 \
    ENV_TAG=pih_0629 \
    TASK_NAME=vision_wrist \
    INPUT_TYPE=vision \
    LOG_NAME=dp_pih \
    bash scripts/run_local.sh

# 2) vision + TacFF through ResNet image encoder
DATASET_PATH=data/pih_quan_June06 \
    ISAACGYM_CONFIG=isaacgym_config.yaml \
    NUM_EPOCH=300 \
    ENV_TAG=pih_resnet_0629 \
    TASK_NAME=visff_wrist \
    INPUT_TYPE=tacff \
    LOG_NAME=dp_pih \
    bash scripts/run_local.sh

# 3) vision + TacFF via ff_head MLP (per-channel min-max norm, no ResNet/imagenet_norm on TacFF)
DATASET_PATH=data/pih_quan_June06 \
    ISAACGYM_CONFIG=isaacgym_config.yaml \
    CONFIG_NAME=train_tacff_diffusion.yaml \
    NUM_EPOCH=300 \
    ENV_TAG=pih_ffhead_0629 \
    TASK_NAME=visff_wrist \
    INPUT_TYPE=tacff \
    LOG_NAME=dp_pih \
    bash scripts/run_local.sh