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

# ── peg insertion benchmark matrix ───────────────────────────────────────────
# pih dataset action dim = 6 (TacSLTaskInsertion via default isaacgym_config.yaml),
# so NO ACTION_SHAPE override. Each block below is a full run; uncomment the ones you
# want (multiple uncommented blocks run sequentially, one after another).
#
#   ours     : TacFF fused in the shared obs encoder (type: tactile, small net,
#              per-channel norm, no ResNet/imagenet)         -> variants 5, 6
#   baseline : naive TacFF-as-image through ResNet18+imagenet -> variant 2
#   legacy   : separate ff_head via DiffusionTacffPolicy      -> variants 3, 4

# 1) vision only (baseline)
# DATASET_PATH=data/pih_quan_June06 \
#     ISAACGYM_CONFIG=isaacgym_config.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=pih_vision_0701 \
#     TASK_NAME=vision_wrist \
#     INPUT_TYPE=vision \
#     LOG_NAME=dp_pih \
#     bash scripts/run_local.sh

# 2) [baseline] faulty TacFF-as-image: task visff_wrist_rgb reproduces BOTH original
#    faults -- dead [0,1] image normalization (norm: image) + ResNet18/imagenet.
# DATASET_PATH=data/pih_quan_June06 \
#     ISAACGYM_CONFIG=isaacgym_config.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=pih_resnet_0701 \
#     TASK_NAME=visff_wrist_rgb \
#     INPUT_TYPE=tacff \
#     LOG_NAME=dp_pih \
#     bash scripts/run_local.sh

# 3) [legacy] separate ff_head MLP (DiffusionTacffPolicy)
# DATASET_PATH=data/pih_quan_June06 \
#     ISAACGYM_CONFIG=isaacgym_config.yaml \
#     CONFIG_NAME=train_tacff_diffusion.yaml \
#     NUM_EPOCH=300 \
#     ENV_TAG=pih_ffhead_0701 \
#     TASK_NAME=visff_wrist \
#     INPUT_TYPE=tacff \
#     LOG_NAME=dp_pih \
#     bash scripts/run_local.sh

# 4) [legacy] separate ff_head CNN (DiffusionTacffPolicy)
# DATASET_PATH=data/pih_quan_June06 \
#     ISAACGYM_CONFIG=isaacgym_config.yaml \
#     CONFIG_NAME=train_tacff_diffusion.yaml \
#     USE_CONV_FF=true \
#     NUM_EPOCH=300 \
#     ENV_TAG=pih_ffcnn_0701 \
#     TASK_NAME=visff_wrist \
#     INPUT_TYPE=tacff \
#     LOG_NAME=dp_pih \
#     bash scripts/run_local.sh

# 5) [ours] TacFF via obs-encoder CNN
DATASET_PATH=data/pih_quan_June06 \
    ISAACGYM_CONFIG=isaacgym_config.yaml \
    NUM_EPOCH=300 \
    ENV_TAG=pih_ffenc_cnn \
    TASK_NAME=visff_wrist \
    INPUT_TYPE=tacff \
    FF_ARCH=cnn \
    LOG_NAME=dp_pih \
    bash scripts/run_local.sh

# 6) [ours] TacFF via obs-encoder MLP
DATASET_PATH=data/pih_quan_June06 \
    ISAACGYM_CONFIG=isaacgym_config.yaml \
    NUM_EPOCH=300 \
    ENV_TAG=pih_ffenc_mlp \
    TASK_NAME=visff_wrist \
    INPUT_TYPE=tacff \
    FF_ARCH=mlp \
    LOG_NAME=dp_pih \
    bash scripts/run_local.sh