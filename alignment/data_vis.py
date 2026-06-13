"""
Visualize a dataset collected by collect_dataset.py in rerun.

Logs camera/tactile images, the rendered TacFF, and time-series for action,
torque (commanded vs measured per joint), TacFF summary magnitudes, and the
success flag — so TacFF/torque correlation can be eyeballed on a shared timeline.

Usage:
  python alignment/data_vis.py data/alignment/usb_rollouts.zarr
  python alignment/data_vis.py data/alignment/usb_rollouts.zarr --episodes 0,1,2
Then open the .rrd (saved next to the zarr) with `rerun <path>.rrd`.
"""

import pathlib
import sys

import click
import numpy as np
import rerun as rr
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from manifeel.utils.shear_tactile_viz_utils import visualize_tactile_shear_image

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

AXES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

TACFF_KEYS = ('tactile_force_field_left', 'tactile_force_field_right')
TORQUE_KEYS = ('dof_torque_cmd', 'dof_force')


def render_tacff(raw_force_field):
    img = visualize_tactile_shear_image(
        raw_force_field[..., 0],
        raw_force_field[..., 1:],
        normal_force_threshold=0.004,
        shear_force_threshold=0.0010,
        resolution=25)
    return np.moveaxis(img[:, :, ::-1], 1, 0)


def log_step(episode, i):
    for key, buffer in episode.items():
        value = buffer[i]
        if key in TACFF_KEYS:
            side = key.rsplit('_', 1)[-1]
            rr.log(f"image/{key}", rr.Image(render_tacff(value)))
            # summary magnitudes: directly comparable against the torque curves
            rr.log(f"tacff/{side}/normal_sum", rr.Scalar(float(np.abs(value[..., 0]).sum())))
            rr.log(f"tacff/{side}/shear_sum",
                   rr.Scalar(float(np.linalg.norm(value[..., 1:], axis=-1).sum())))
        elif value.ndim == 3:  # (H, W, C) image
            rr.log(f"image/{key}", rr.Image(value))
        elif key in TORQUE_KEYS:
            # commanded and measured of the same joint share a plot path
            for j in range(value.shape[0]):
                rr.log(f"torque/joint{j}/{key}", rr.Scalar(float(value[j])))
        elif key == 'action':
            for j in range(value.shape[0]):
                rr.log(f"action/{AXES[j]}", rr.Scalar(float(value[j])))
        elif key == 'success':
            rr.log("success", rr.Scalar(float(np.ravel(value)[0])))
        elif value.ndim == 1:  # state, dof_pos, dof_vel, ...
            for j in range(value.shape[0]):
                rr.log(f"{key}/{j}", rr.Scalar(float(value[j])))


@click.command()
@click.argument('zarr_path', type=click.Path(exists=True))
@click.option('--episodes', default=None, help='Comma-separated episode indices (default: all)')
@click.option('-o', '--output', default=None, help='Output .rrd path (default: <zarr>/debug.rrd)')
def main(zarr_path, episodes, output):
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')

    if output is None:
        output = pathlib.Path(zarr_path) / "debug.rrd"
    output = pathlib.Path(output)
    if output.exists():
        output.unlink()

    rr.init("alignment_data_vis", spawn=False)
    rr.save(str(output))

    if episodes is not None:
        episode_idxs = [int(e.strip()) for e in episodes.split(",")]
    else:
        episode_idxs = list(range(replay_buffer.n_episodes))

    meta_success = replay_buffer.meta.get('episode_success', None)
    meta_seed = replay_buffer.meta.get('episode_seed', None)

    step = 0
    for episode_idx in tqdm(episode_idxs, "Loading episodes"):
        episode = replay_buffer.get_episode(episode_idx)
        size = len(next(iter(episode.values())))
        for i in range(size):
            rr.set_time_sequence("step", step)
            log_step(episode, i)
            rr.log("episode", rr.Scalar(episode_idx))
            if meta_success is not None:
                rr.log("episode_success", rr.Scalar(float(meta_success[episode_idx])))
            step += 1
        if meta_success is not None:
            seed = int(meta_seed[episode_idx]) if meta_seed is not None else '?'
            print(f"episode {episode_idx}: {size} steps, "
                  f"success={int(meta_success[episode_idx])}, seed={seed}")

    rr.disconnect()
    print(f"Saved at {output}!")


if __name__ == "__main__":
    main()
