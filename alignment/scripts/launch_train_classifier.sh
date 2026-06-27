# vision only (baseline)
# python alignment/training/train_classifier.py \
#   --task usb \
#   --obs_keys wrist \
#   --output_dir alignment/models/usb/classifiers/vision_only

# vision + external torque (tau_ext = J_linᵀ·contact_force; dof_force reads ~0 in this build)
python alignment/training/train_classifier.py \
  --task usb \
  --obs_keys wrist,tau_ext,state \
  --output_dir alignment/models/usb/classifiers/vision_torque

# vision + tactile
python alignment/training/train_classifier.py \
  --task usb \
  --obs_keys wrist,tactile_force_field_right \
  --output_dir alignment/models/usb/classifiers/vision_tactile
