CHECKPOINT=${CHECKPOINT:-data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt}
TASK=${TASK:-usb}
LOSS=${LOSS:-mse}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-4}
FREEZE_IMG=${FREEZE_IMG:-True}

apptainer exec --nv --cleanenv --env LD_PRELOAD= manifeel.sif bash -ic "
  conda activate manifeel
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  python alignment/training/train_alignment.py \
    --checkpoint ${CHECKPOINT} \
    --task       ${TASK} \
    --loss       ${LOSS} \
    --epochs     ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr         ${LR} \
    --freeze_img_encoder ${FREEZE_IMG}
"
