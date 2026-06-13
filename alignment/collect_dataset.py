"""
Collect a rollout dataset for TacFF/torque alignment and success classification.

Rolls out a trained policy checkpoint in the TacSL sim (same machinery as eval.py,
but without the MultiStep/Video wrappers so every single env step is recorded) and
writes a zarr dataset in the same layout as the ManiFeel demo datasets:

  data/
    <obs streams the env produces>        # front, side, wrist, right tactile streams,
                                          # state, ... (per step; see --skip_keys)
    action                                # action executed at this step
    dof_pos, dof_vel                      # joint state (9 dofs: 7 arm + 2 gripper)
    dof_torque_cmd                        # commanded joint torque (task-space impedance ctrl)
    dof_force                             # measured joint torque (Isaac Gym DOF force sensors)
    success                               # per-step success flag from task._check_success()
  meta/
    episode_ends                          # standard ReplayBuffer episode boundaries
    episode_success                       # 1 if success was ever reached in the episode
    episode_seed                          # RNG seed used for the episode's round

Only signals that exist on real hardware are recorded (a real Franka has joint
torque sensors but no finger force sensors), so the dataset transfers to real-robot
work. All per-step channels are sampled at the same instant (right after the physics
step), so TacFF and torque are synchronized frame by frame.

Test-set hygiene: the default seed (31415926) is far away from the designated eval
seed (test_start_seed, 100000 in the task configs); keep it that way.

Usage:
  python alignment/collect_dataset.py \
    -c data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \
    -o data/alignment/usb_rollouts.zarr \
    --num_envs 16 --n_rounds 4
"""

import isaacgym  # noqa: F401  (must be imported before torch)

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')

import os
import collections
import pathlib

import click
import cv2
import dill
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from manifeel.envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper
from manifeel.gym_util.multistep_wrapper import stack_last_n_obs

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Keys from the training overrides that are not meaningful at collection time.
_TRAINING_ONLY_KEYS = {
    'training.seed', 'training.num_epochs', 'logging.project',
    'hydra.run.dir', 'exp_name',
}


def _load_training_overrides(checkpoint_path: str):
    """Read <run_dir>/.hydra/overrides.yaml from the checkpoint's grandparent dir."""
    overrides_file = pathlib.Path(checkpoint_path).parent.parent / '.hydra' / 'overrides.yaml'
    if not overrides_file.exists():
        return []
    with open(overrides_file) as f:
        raw = yaml.safe_load(f) or []
    filtered = []
    for entry in raw:
        key = entry.split('=')[0].lstrip('~+')
        if key not in _TRAINING_ONLY_KEYS:
            filtered.append(entry)
    return filtered


def _compose_isaacgym_cfg(isaacgym_cfg_name, shape_meta, num_envs):
    """Compose the isaacgym env config the same way ManifeelRunner does."""
    isaacgym_cfg = hydra.compose(config_name=isaacgym_cfg_name)
    OmegaConf.set_struct(isaacgym_cfg, False)
    if 'light_factor' not in isaacgym_cfg:
        print(f"[collect] hydra.compose returned incomplete isaacgym config "
              f"(keys: {sorted(isaacgym_cfg.keys())}); "
              f"loading {isaacgym_cfg_name} from yaml directly")
        config_dir = pathlib.Path(__file__).parent.parent / 'manifeel' / 'config'
        base = OmegaConf.load(config_dir / isaacgym_cfg_name)
        OmegaConf.set_struct(base, False)
        if 'task' in isaacgym_cfg:
            base.task = isaacgym_cfg.task
        isaacgym_cfg = base
    isaacgym_cfg.shape_meta = OmegaConf.create(shape_meta)
    isaacgym_cfg.num_envs = num_envs
    return isaacgym_cfg


def _sample_extras(task):
    """Read joint-state/torque/success tensors off the task at the current instant.

    Only signals measurable on a real Franka (joint sensors + own commands);
    simulator-privileged contact forces are deliberately not recorded.
    Returns dict of (num_envs, ...) float32/uint8 numpy arrays.
    """
    extras = {
        'dof_pos': task.dof_pos,
        'dof_vel': task.dof_vel,
        'dof_torque_cmd': task.dof_torque,   # commanded (impedance controller)
        'dof_force': task.dof_force_view,    # measured (DOF force sensors)
    }
    extras = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in extras.items()}
    success = task._check_success()
    extras['success'] = success.detach().cpu().numpy().astype(np.uint8).reshape(-1, 1)
    return extras


def _resize_taxim(np_obs_dict, tactile_size):
    """Resize tactile RGB camera obs to the size the policy was trained with
    (same logic as ManifeelRunner)."""
    for key in ['left_tactile_camera_taxim', 'right_tactile_camera_taxim']:
        if key not in np_obs_dict:
            continue
        data = np_obs_dict[key]
        B, T, C, H, W = data.shape
        resized = np.zeros((B, T, C, tactile_size[0], tactile_size[1]), dtype=data.dtype)
        for b in range(B):
            for t in range(T):
                for c in range(C):
                    resized[b, t, c] = cv2.resize(
                        data[b, t, c], (tactile_size[1], tactile_size[0]),
                        interpolation=cv2.INTER_LINEAR)
        np_obs_dict[key] = resized
    return np_obs_dict


def _to_storage_layout(value):
    """Convert a recorded per-step array to the on-disk layout of the demo datasets:
    image-like channels go back to channel-last (H, W, 3); float64 is downcast."""
    if value.ndim >= 3 and value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if value.dtype == np.float64:
        value = value.astype(np.float32)
    return value


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output', required=True, help='Output zarr directory')
@click.option('-n', '--cfg_name', default='train_diffusion_workspace.yaml')
@click.option('-d', '--device', default='cuda:0')
@click.option('--isaacgym_cfg', default=None, help='Override isaacgym config yaml')
@click.option('--num_envs', default=16, type=int, help='Parallel envs per round (= episodes per round)')
@click.option('--n_rounds', default=4, type=int, help='Rounds; each round reseeds and resets all envs')
@click.option('--max_steps', default=None, type=int, help='Max env steps per round (default: runner max_steps)')
@click.option('--seed', default=31415926, type=int,
              help='Base collection seed; round r uses seed + r. '
                   'Default is far from the designated test seed (100000).')
@click.option('--stop_on_all_done', default=True, type=bool,
              help='End a round early once every env has reached success')
@click.option('--post_success_steps', default=15, type=int,
              help='Steps kept after an episode first reaches success')
@click.option('--skip_keys',
              default='tactile_force_field_left,left_tactile_camera_taxim,tactile_depth_left,client',
              help='Comma-separated obs streams not to record. The policies only use the '
                   'right tactile sensor, so the left streams (and the viz-only client cam) '
                   'are skipped by default.')
@click.option('--append', is_flag=True, default=False, help='Append to an existing zarr dataset')
def main(checkpoint, output, cfg_name, device, isaacgym_cfg, num_envs, n_rounds,
         max_steps, seed, stop_on_all_done, post_success_steps, skip_keys, append):
    if os.path.exists(output) and not append:
        raise click.ClickException(
            f"Output path {output} already exists. Pass --append to add episodes to it.")
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    skip_keys = {k.strip() for k in skip_keys.split(',') if k.strip()}

    # ----- compose configs the same way eval.py does -----
    config_dir = pathlib.Path(__file__).resolve().parent.parent / 'manifeel' / 'config'
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)

    overrides = _load_training_overrides(checkpoint)
    if overrides:
        print(f"[collect] training overrides from checkpoint: {overrides}")
    if isaacgym_cfg is not None:
        overrides.append(f'isaacgym_cfg_name={isaacgym_cfg}')
    cfg = hydra.compose(config_name=cfg_name, overrides=overrides)

    with open_dict(cfg):
        cfg.task.env_runner.shape_meta = OmegaConf.to_container(
            cfg.task.shape_meta, resolve=True)

    runner_cfg = cfg.task.env_runner
    n_obs_steps = cfg.n_obs_steps
    if max_steps is None:
        max_steps = int(runner_cfg.max_steps)
    tactile_size = list(runner_cfg.get('tactile_size', [256, 256]))
    round_seeds = [seed + r for r in range(n_rounds)]
    print(f"[collect] {n_rounds} rounds x {num_envs} envs, "
          f"max {max_steps} steps/round, seeds {round_seeds[0]}..{round_seeds[-1]}")

    # ----- load policy -----
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=str(pathlib.Path(output).parent))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # ----- build env (no MultiStep/Video wrappers: we record every step ourselves) -----
    isaacgym_env_cfg = _compose_isaacgym_cfg(
        runner_cfg.isaacgym_cfg_name, cfg.task.env_runner.shape_meta, num_envs)
    env = MultipleIsaacEnvWrapper(isaacgym_env_cfg)
    task = env.envs

    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')
    episode_success, episode_seed = [], []
    if append and 'episode_success' in replay_buffer.meta:
        episode_success = list(np.asarray(replay_buffer.meta['episode_success']))
        episode_seed = list(np.asarray(replay_buffer.meta['episode_seed']))

    # ----- rollout -----
    for round_idx, round_seed in enumerate(round_seeds):
        env.seed(round_seed)
        policy.reset()
        policy_obs = env.reset()
        obs_history = collections.deque([policy_obs], maxlen=n_obs_steps)

        # records[key] is a list over time of (num_envs, ...) arrays
        records = collections.defaultdict(list)
        done_mask = np.zeros(num_envs, dtype=bool)
        t = 0
        while t < max_steps:
            stacked = {
                key: stack_last_n_obs([o[key] for o in obs_history], n_obs_steps)
                for key in policy_obs.keys()
            }
            stacked = _resize_taxim(stacked, tactile_size)
            obs_dict = dict_apply(stacked, lambda x: torch.from_numpy(x).to(device=device))
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            actions = action_dict['action'].detach().cpu().numpy()  # (B, n_action_steps, Da)

            for k in range(actions.shape[1]):
                # snapshot of time t: full obs (all env streams) + extras, sampled together
                for key, value in env.render_cache.items():
                    if key not in skip_keys:
                        records[key].append(value)
                for key, value in _sample_extras(task).items():
                    records[key].append(value)
                records['action'].append(actions[:, k].astype(np.float32))

                policy_obs, _, done, _ = env.step(actions[:, k])
                obs_history.append(policy_obs)
                done_mask |= done.astype(bool)
                t += 1
                if t >= max_steps:
                    break
            if stop_on_all_done and done_mask.all():
                break

        # ----- write one episode per env -----
        # Per-step snapshots are taken *before* each action, so the state after the
        # final action is not in `records`; sample success once more for the label.
        final_success = _sample_extras(task)['success'][:, 0]   # (num_envs,)
        success_per_step = np.array(records['success'])[:, :, 0]  # (T, num_envs)
        ep_success_all = np.maximum(success_per_step.max(axis=0), final_success)
        for i in range(num_envs):
            # successful envs just hold position afterwards; keep only a short tail
            end = len(success_per_step)
            hits = np.flatnonzero(success_per_step[:, i])
            if len(hits) > 0:
                end = min(end, hits[0] + 1 + post_success_steps)
            episode = {
                key: np.stack([_to_storage_layout(step[i]) for step in steps[:end]])
                for key, steps in records.items()
            }
            replay_buffer.add_episode(episode, compressors='disk')
            episode_success.append(int(ep_success_all[i]))
            episode_seed.append(round_seed)

        n_success = int(ep_success_all.sum())
        print(f"[collect] round {round_idx + 1}/{n_rounds} (seed {round_seed}): "
              f"{t} steps, {n_success}/{num_envs} successes")

    replay_buffer.update_meta({
        'episode_success': np.array(episode_success, dtype=np.uint8),
        'episode_seed': np.array(episode_seed, dtype=np.int64),
    })

    total = len(episode_success)
    n_success = int(np.sum(episode_success))
    print(f"[collect] done: {total} episodes written to {output}")
    print(f"[collect] success rate: {n_success}/{total} ({n_success / total:.1%}) "
          f"-- sanity-check against the policy's eval score")
    print(f"[collect] data keys: {list(replay_buffer.data.keys())}")


if __name__ == '__main__':
    main()
