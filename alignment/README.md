# Alignment

Work on (1) aligning visuo-tactile features (TacFF) with robot torque, and
(2) classifying task-success patterns. First step: collect rollout datasets that
contain everything both directions need.

## Dataset collection

`collect_dataset.py` rolls out a trained policy checkpoint in the TacSL sim and
records **every** env step (no MultiStep aggregation) to a zarr dataset with the
same layout as the ManiFeel demo datasets, so `ManifeelDataset` can read it by
listing the keys in `shape_meta`.

```bash
# inside the apptainer env (see scripts/run_local.sh for the exec incantation)
python alignment/collect_dataset.py \
  -c data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \
  -o data/alignment/usb_rollouts.zarr \
  --num_envs 16 --n_rounds 4
```

The task, isaacgym config, and shape_meta are replayed from the checkpoint's
`.hydra/overrides.yaml`, exactly like `eval.py`.

### Recorded channels (per step, all sampled at the same instant)

| key | shape | meaning |
|---|---|---|
| `front`, `side`, `wrist`, ... | (H, W, 3) | whatever camera streams the env config enables |
| `right_tactile_camera_taxim` | (H, W, 3) | tactile RGB (stored at native size) |
| `tactile_force_field_right` | (10, 14, 3) | TacFF: (normal, shear_x, shear_y) per taxel |
| `state` | (7,) | ee_pos + ee_quat (what the policy consumes) |
| `action` | (6,) or (7,) | action executed at this step |
| `dof_pos`, `dof_vel` | (9,) | joint positions / velocities |
| `dof_torque_cmd` | (9,) | commanded joint torque (task-space impedance controller) |
| `dof_force` | (9,) | measured joint torque (Isaac Gym DOF force sensors) |
| `success` | (1,) | per-step `_check_success()` flag |

Only real-hardware-available signals are recorded: a real Franka has joint torque
sensors (`dof_force` ≈ libfranka `tau_J`), but no finger force sensors — so the
simulator's privileged contact forces are deliberately left out.

The policies only consume the right tactile sensor, so the left-side streams and
the viz-only `client` camera are skipped by default (`--skip_keys`). The two
GelSight RGB streams dominate storage (~250 KB/step each compressed); add
`right_tactile_camera_taxim` to `--skip_keys` if you don't need tactile RGB at all.

Per-episode metadata: `meta/episode_ends` (standard), `meta/episode_success`,
`meta/episode_seed`.

## Visualization

`data_vis.py` renders a collected zarr to a rerun recording — images, rendered
TacFF, per-joint commanded-vs-measured torque, TacFF summary magnitudes, and
success flags on a shared timeline (the quick eyeball check for TacFF/torque
correlation):

```bash
python alignment/data_vis.py data/alignment/usb_rollouts.zarr --episodes 0,1,2
rerun data/alignment/usb_rollouts.zarr/debug.rrd
```

### Notes / gotchas

- **Test-set hygiene**: the default `--seed` is 31415926, far away from the
  designated eval seed (`test_start_seed`, 100000 in the task configs). Keep
  collection seeds away from it so the test set stays hidden from training.
- Step semantics: row *t* holds the state **before** `action[t]` is applied.
  TacFF, torque, and success in the same row are synchronized.
- `dof_torque_cmd` is zeros in each episode's first row (no controller step has
  run yet after reset).
- Env steps run at 15 Hz (physics dt 1/60 s x `controlFrequencyInv` 4); the
  default 500-step round is ~33 s of sim time.
- Envs are not reset on success — a successful env just holds position while the
  rest of the round finishes. Each episode is therefore trimmed at write time to
  its first success plus `--post_success_steps` (default 15, ~1 s), so the success
  transition is kept but the redundant holding frames are not.
- Failures come for free: rollouts of an imperfect policy yield both labels for
  the success classifier (see `meta/episode_success`). If the success rate is too
  lopsided, collect extra rounds from weaker/earlier checkpoints.
