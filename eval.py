"""
Usage:
  # Headless eval - reads task/overrides from the checkpoint's .hydra/overrides.yaml automatically
  python eval.py \
    -c data/outputs/vision_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \
    -o data/eval_output \
    -n train_diffusion_workspace.yaml

  # Interactive viewer (1 env, needs a display)
  python eval.py \
    -c data/outputs/vision_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \
    -o data/eval_output \
    -n train_diffusion_workspace.yaml \
    --isaacgym_cfg isaacgym_config_gui.yaml \
    --n_test 1 --n_test_vis 1
"""

import isaacgym

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8')

import os
import pathlib
import click
import hydra
import torch
import numpy as np
import dill
import wandb
import json
import yaml
from omegaconf import OmegaConf, open_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Keys from the training overrides that are not meaningful at eval time.
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


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-n', '--cfg_name', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_test', default=None, type=int, help='Number of parallel eval envs (overrides task yaml)')
@click.option('--n_test_vis', default=None, type=int, help='Number of envs to record video for')
@click.option('--isaacgym_cfg', default=None, help='Override isaacgym config (e.g. isaacgym_config_gui.yaml for viewer)')
def main(checkpoint, output_dir, cfg_name, device, n_test, n_test_vis, isaacgym_cfg):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize Hydra
    hydra.initialize(config_path=str(pathlib.Path(__file__).parent.joinpath(
        './manifeel','config')), version_base=None)

    # Replay the overrides used at training time (task=, dataset_path=, etc.)
    # so Hydra resolves the same config tree without needing manual flags.
    overrides = _load_training_overrides(checkpoint)
    if overrides:
        print(f"Loaded training overrides from checkpoint: {overrides}")

    # Eval-specific overrides come last and win over training values.
    if n_test is not None:
        overrides.append(f'task.env_runner.n_test={n_test}')
    if n_test_vis is not None:
        overrides.append(f'task.env_runner.n_test_vis={n_test_vis}')
    if isaacgym_cfg is not None:
        overrides.append(f'isaacgym_cfg_name={isaacgym_cfg}')
    cfg = hydra.compose(config_name=str(cfg_name), overrides=overrides)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    # cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # Pre-resolve shape_meta so the ${shape_meta} interpolation in env_runner doesn't
    # fail when hydra.compose() is used instead of @hydra.main (which has a stricter
    # resolution context inside Apptainer containers).
    with open_dict(cfg):
        cfg.task.env_runner.shape_meta = OmegaConf.to_container(
            cfg.task.shape_meta, resolve=True)

    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            if isinstance(value, np.integer):
                value = int(value)
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
