apptainer exec --nv --cleanenv --env LD_PRELOAD= manifeel.sif bash -ic "
  conda activate manifeel
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
  python alignment/collect_dataset.py \
  -c data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \
  --num_envs 4 --n_rounds 1 --max_steps 100
"
