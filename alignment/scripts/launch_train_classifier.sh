# vision only (baseline)
python alignment/training/train_classifier.py \
  --task usb \
  --obs_keys wrist \
  --output_dir models/usb/classifiers/vision_only

# vision + torque
python alignment/training/train_classifier.py \
  --task usb \
  --obs_keys wrist,dof_torque_cmd,state \
  --output_dir models/usb/classifiers/vision_torque

# vision + tactile
python alignment/training/train_classifier.py \
  --task usb \
  --obs_keys wrist,tactile_force_field_right \
  --output_dir models/usb/classifiers/vision_tactile
