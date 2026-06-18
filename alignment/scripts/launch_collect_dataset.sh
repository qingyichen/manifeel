apptainer exec --nv --cleanenv --env LD_PRELOAD= manifeel.sif bash -ic "
  conda activate manifeel
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  python alignment/collect_dataset.py \
  -c data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \
  --num_envs 20 --n_rounds 5 --max_steps 400 \
  --output alignment/data/usb.zarr
"
