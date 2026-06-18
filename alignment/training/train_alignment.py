"""
Offline alignment: proxy encoder (wrist image + joint torque + state) → TacFF feature space.

Teacher: frozen ff_head from a DiffusionTacffPolicy checkpoint.
Student: ProxyEncoder(wrist + dof_force + dof_pos + state).
Loss:    cosine distance between z_proxy and z_tacff.

The proxy can later substitute the tactile encoder at inference time on hardware
that has a wrist camera and joint torque sensors but no tactile finger.

Usage:
  # infer paths from task name (zarr → alignment/data/<task>.zarr,
  #                              output → alignment/data/<task>_proxy.ckpt)
  python alignment/training/train_alignment.py \\
    --checkpoint data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \\
    --task usb

  # or specify paths explicitly
  python alignment/training/train_alignment.py \\
    --checkpoint data/outputs/tacff_usb_wrist_0805_50/0/checkpoints/latest_epoch999.ckpt \\
    --zarr_path  alignment/data/usb.zarr \\
    --output     alignment/data/usb_proxy.ckpt
"""

import pathlib

import click
import dill
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import zarr
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)

_TRAINING_ONLY_KEYS = {
    'training.seed', 'training.num_epochs', 'logging.project',
    'hydra.run.dir', 'exp_name',
}


# Dataset and model definitions
class AlignmentDataset(Dataset):
    """Per-step dataset: (wrist_img, lowdim, tacff) from a rollout zarr."""

    def __init__(self, zarr_path: str, img_transform=None):
        z = zarr.open(zarr_path, 'r')['data']
        # (N, H, W, 3) float32 in [0, 1]
        self.wrist   = torch.from_numpy(z['wrist'][:])
        # (N, 7) + (N, 9) + (N, 9) → concatenated later
        self.state   = torch.from_numpy(z['state'][:])
        self.dof_pos = torch.from_numpy(z['dof_pos'][:])
        self.force   = torch.from_numpy(z['dof_force'][:])
        # (N, 10, 14, 3) float32 — channel-last as stored by collect_dataset
        self.tacff   = torch.from_numpy(z['tactile_force_field_right'][:])

        self.img_transform = img_transform

        # compute per-feature mean/std for low-dim normalization from the dataset
        lowdim = torch.cat([self.state, self.dof_pos, self.force], dim=1)
        self.lowdim_mean = lowdim.mean(0)
        self.lowdim_std  = lowdim.std(0).clamp(min=1e-6)

    def __len__(self):
        return len(self.wrist)

    def __getitem__(self, idx):
        # (H,W,3) → (3,H,W)
        img = self.wrist[idx].permute(2, 0, 1)
        if self.img_transform is not None:
            img = self.img_transform(img)

        lowdim = torch.cat([self.state[idx], self.dof_pos[idx], self.force[idx]])
        lowdim = (lowdim - self.lowdim_mean) / self.lowdim_std

        # (10,14,3) → (3,10,14)
        tacff = self.tacff[idx].permute(2, 0, 1)

        return img, lowdim, tacff

class ProxyEncoder(nn.Module):
    """(wrist image + low-dim) → feat_dim embedding."""

    def __init__(self, lowdim_dim: int = 25, feat_dim: int = 512,
                 img_encoder: nn.Module = None):
        super().__init__()
        if img_encoder is not None:
            self.img_encoder = img_encoder
        else:
            resnet = models.resnet18(weights=None)
            self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B,512,1,1)

        self.lowdim_encoder = nn.Sequential(
            nn.Linear(lowdim_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.projector = nn.Sequential(
            nn.Linear(512 + 256, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, img: torch.Tensor, lowdim: torch.Tensor) -> torch.Tensor:
        img_feat = self.img_encoder(img).flatten(1)           # (B, 512)
        ld_feat  = self.lowdim_encoder(lowdim)                # (B, 256)
        return self.projector(torch.cat([img_feat, ld_feat], dim=1))  # (B, feat_dim)


# Helper functions for loss computation and checkpoint loading
_LOSS_FNS = {'cosine': cosine_loss, 'mse': mse_loss}
def cosine_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(z1, z2, dim=-1)).mean()

def mse_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z1, z2)

def _load_training_overrides(checkpoint_path: str):
    overrides_file = pathlib.Path(checkpoint_path).parent.parent / '.hydra' / 'overrides.yaml'
    if not overrides_file.exists():
        return []
    with open(overrides_file) as f:
        raw = yaml.safe_load(f) or []
    return [e for e in raw if e.split('=')[0].lstrip('~+') not in _TRAINING_ONLY_KEYS]


def load_tacff_encoder(checkpoint_path: str, cfg_name: str, device: torch.device):
    """Load ff_head, normalizer, and wrist backbone from a DiffusionTacffPolicy checkpoint."""
    config_dir = (
        pathlib.Path(__file__).resolve().parents[2] / 'manifeel' / 'config'
    )
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None)

    overrides = _load_training_overrides(checkpoint_path)
    cfg = hydra.compose(config_name=cfg_name, overrides=overrides)
    with open_dict(cfg):
        cfg.task.env_runner.shape_meta = OmegaConf.to_container(
            cfg.task.shape_meta, resolve=True)

    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location='cpu')
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir='/tmp/align_tmp')
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    wrist_encoder = policy.obs_encoder.key_model_map['wrist']
    return policy.ff_head, policy.normalizer, wrist_encoder


# Training loop with command-line interface
@click.command()
@click.option('--checkpoint', '-c', required=True,
              help='DiffusionTacffPolicy checkpoint path')
@click.option('--task',       '-t', default=None,
              help='Task name; infers zarr_path=alignment/models/<task>.zarr '
                   'and output=alignment/models/<task>_proxy.ckpt')
@click.option('--zarr_path',  '-z', default=None,
              help='Rollout zarr dataset (overrides --task)')
@click.option('--output',     '-o', default=None,
              help='Output checkpoint path (overrides --task)')
@click.option('--cfg_name',   default='train_diffusion_workspace.yaml')
@click.option('--device',     '-d', default='cuda:0')
@click.option('--epochs',     default=100,   type=int)
@click.option('--batch_size', default=256,   type=int)
@click.option('--lr',         default=1e-4,  type=float)
@click.option('--val_ratio',  default=0.1,   type=float)
@click.option('--num_workers',default=4,     type=int)
@click.option('--loss', default='mse', type=click.Choice(['mse', 'cosine']),
              help='Alignment loss function')
@click.option('--freeze_img_encoder', default=True, type=bool,
              help='Freeze the wrist encoder weights from the policy checkpoint')
def main(checkpoint, task, zarr_path, output, cfg_name, device, epochs, batch_size,
         lr, val_ratio, num_workers, loss, freeze_img_encoder):
    if task is not None:
        if zarr_path is None:
            zarr_path = f'alignment/models/{task}.zarr'
        if output is None:
            output = f'alignment/models/{task}_proxy.ckpt'
    if zarr_path is None or output is None:
        raise click.UsageError('Provide --task or both --zarr_path and --output')
    device = torch.device(device)

    # frozen teacher
    print('[align] loading teacher (ff_head) from checkpoint ...')
    ff_head, normalizer, wrist_encoder = load_tacff_encoder(checkpoint, cfg_name, device)
    tacff_normalizer = normalizer['tactile_force_field_right']
    if freeze_img_encoder:
        for p in wrist_encoder.parameters():
            p.requires_grad_(False)
        print('[align] wrist encoder frozen')
    else:
        print('[align] wrist encoder will be fine-tuned')

    # ── dataset ──
    img_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = AlignmentDataset(zarr_path, img_transform=img_transform)

    n_val   = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f'[align] dataset: {len(dataset)} steps  '
          f'train={n_train}  val={n_val}')
    loss_fn = _LOSS_FNS[loss]
    print(f'[align] loss: {loss}')

    # ── proxy encoder ──
    lowdim_dim = 7 + 9 + 9   # state + dof_pos + dof_force
    proxy = ProxyEncoder(lowdim_dim=lowdim_dim, img_encoder=wrist_encoder).to(device)
    optimizer = torch.optim.Adam(proxy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        proxy.train()
        train_loss = 0.0
        for img, lowdim, tacff_raw in tqdm(train_loader,
                                            desc=f'epoch {epoch:3d}/{epochs}',
                                            leave=False):
            img, lowdim, tacff_raw = (
                img.to(device), lowdim.to(device), tacff_raw.to(device))

            # teacher: normalize TacFF → flatten → frozen ff_head
            tacff_norm = tacff_normalizer.normalize(tacff_raw)
            tacff_flat = tacff_norm.reshape(tacff_norm.shape[0], -1)   # (B, 420)
            with torch.no_grad():
                z_tacff = ff_head(tacff_flat)

            z_proxy = proxy(img, lowdim)
            loss = loss_fn(z_proxy, z_tacff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # val
        proxy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, lowdim, tacff_raw in val_loader:
                img, lowdim, tacff_raw = (
                    img.to(device), lowdim.to(device), tacff_raw.to(device))
                tacff_norm = tacff_normalizer.normalize(tacff_raw)
                tacff_flat = tacff_norm.reshape(tacff_norm.shape[0], -1)
                z_tacff = ff_head(tacff_flat)
                z_proxy = proxy(img, lowdim)
                val_loss += loss_fn(z_proxy, z_tacff).item()
        val_loss /= len(val_loader)

        print(f'[align] epoch {epoch:3d}/{epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'proxy_state_dict': proxy.state_dict(),
                'lowdim_mean': dataset.lowdim_mean,
                'lowdim_std':  dataset.lowdim_std,
                'lowdim_dim':  lowdim_dim,
                'val_loss':    val_loss,
            }, output)
            print(f'[align]   ↳ best val={val_loss:.4f}  saved → {output}')

    print(f'[align] done.  best val loss: {best_val:.4f}')


if __name__ == '__main__':
    main()
