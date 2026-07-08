# 1) vision only (baseline)
DATASET_PATH=data/pih_quan_June06 \
    ISAACGYM_CONFIG=isaacgym_config.yaml \
    NUM_EPOCH=300 \
    ENV_TAG=pih_vision_0701 \
    TASK_NAME=vision_wrist \
    INPUT_TYPE=vision \
    LOG_NAME=dp_pih \
    bash scripts/run_local.sh
