"""
Offline success classifier: sequence of observations → binary success.

Dataset:    the zarr produced by collect_dataset.py.
Modalities: any subset of zarr obs keys, e.g.:
              wrist,dof_force,state           (vision + torque)
              wrist,tactile_force_field_right  (vision + tactile)
              wrist,front,dof_force,state      (multi-view + torque)
Sequence:   sliding window of --seq_len steps over each episode;
            label = success flag at the last step of the window.
Output:     models/<task>/classifiers/best.ckpt + train_log.csv + settings.json

Key type is inferred from the stored per-step shape (size-based, not ndim):
  ndim >= 3 and min(H, W) >= 32  → image key  → separate ResNet-18 encoder
  everything else                → lowdim key → shared MLP encoder (flattened,
                                                 all lowdim concat'd, z-scored)
This routes large camera frames to ResNet while small spatial fields like the
10×14×3 tactile force field are flattened + per-feature z-scored (their physical
force units are ~1e-3 N, so the image [0,1]→[-1,1] transform would erase them).

Usage:
  python alignment/training/train_classifier.py \\
    --task usb --obs_keys wrist,dof_force,state --seq_len 8

  python alignment/training/train_classifier.py \\
    --task usb \\
    --zarr_path  alignment/data/usb.zarr \\
    --output_dir alignment/classifiers/usb \\
    --obs_keys   wrist,tactile_force_field_right \\
    --seq_len 4
"""

import copy
import json
import pathlib

import click
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClassifierDataset(Dataset):
    """Sliding-window dataset over rollout zarr episodes.

    Each item: (obs_dict, label) where label = success flag at the last window step.
      obs[img_key] : (seq_len, C, H, W) float32, mapped [0,1]→[-1,1]
      obs['lowdim']: (seq_len, D) float32, z-score normalised (all lowdim keys cat'd)
      label        : scalar float32 in {0, 1}
    """

    def __init__(self, zarr_path, obs_keys, seq_len, val_ratio=0.1, seed=42):
        obs_keys = list(obs_keys)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys + ['success'])
        self.obs_keys = obs_keys
        self.seq_len  = seq_len

        # classify each key by stored per-step shape.
        #   true images (e.g. 256×256×3 camera) → ResNet encoder + [0,1]→[-1,1]
        #   everything else (vectors, and small spatial fields like the 10×14×3
        #   tactile force field) → flattened + per-feature z-score MLP.
        # Routing is size-based, not ndim-based: a 10×14 force field is in physical
        # force units (~1e-3 N), not [0,1], so the image *2-1 norm would crush it,
        # and a tiny field through a 256²-oriented ResNet is wasteful.
        self.img_keys    = []
        self.lowdim_keys = []
        for key in obs_keys:
            per_step_shape = self.replay_buffer[key].shape[1:]  # strip N axis
            is_image = len(per_step_shape) >= 3 and min(per_step_shape[:2]) >= 32
            (self.img_keys if is_image else self.lowdim_keys).append(key)

        val_mask        = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio, seed=seed)
        self.train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=seq_len,
            pad_before=0,
            pad_after=0,
            episode_mask=self.train_mask)

        # lowdim normalisation: per-feature z-score over all stored steps.
        # Each key is flattened to (N, -1) first so spatial fields (e.g. the
        # 10×14×3 tactile force field → 420 features) get per-element stats —
        # important because the 3 channels (normal, shear-x, shear-y) differ in
        # scale by ~4× and a single global scale would under-weight the smallest.
        if self.lowdim_keys:
            parts = [self.replay_buffer[k][:].astype(np.float32).reshape(
                         self.replay_buffer[k].shape[0], -1)
                     for k in self.lowdim_keys]
            all_ld = np.concatenate(parts, axis=-1)  # (N, D)
            self.lowdim_mean = torch.from_numpy(all_ld.mean(0)).float()
            self.lowdim_std  = torch.from_numpy(all_ld.std(0).clip(1e-6)).float()
        else:
            self.lowdim_mean = self.lowdim_std = None

        # image keys are normalised with a fixed x*2-1 (maps [0,1]→[-1,1]);
        # no data-derived stats needed since camera images are always in [0,1].

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=0,
            pad_after=0,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        sample = self.sampler.sample_sequence(idx)
        obs = {}

        for key in self.img_keys:
            # (T, H, W, C) → (T, C, H, W), then [0,1] → [-1,1]
            img = torch.from_numpy(sample[key].astype(np.float32))
            obs[key] = img.permute(0, 3, 1, 2) * 2.0 - 1.0

        if self.lowdim_keys:
            parts = [torch.from_numpy(sample[k].astype(np.float32)).reshape(
                         self.seq_len, -1)
                     for k in self.lowdim_keys]
            ld = torch.cat(parts, dim=-1)  # (T, D)
            obs['lowdim'] = (ld - self.lowdim_mean) / self.lowdim_std

        label = torch.tensor(float(sample['success'][-1, 0]), dtype=torch.float32)
        return obs, label


# ── Model ─────────────────────────────────────────────────────────────────────

class SuccessClassifier(nn.Module):
    """Multi-modal sequence → scalar logit (pass through sigmoid for probability).

    Per-step encoder:
      - Independent ResNet-18 per image key (AdaptiveAvgPool handles any spatial size)
      - 2-layer MLP for all lowdim keys concatenated
    Temporal encoder: GRU over seq_len steps (batch_first)
    Head: 2-layer MLP → scalar
    """

    _IMG_OUT    = 512   # ResNet-18 adaptive pool output
    _LOWDIM_OUT = 256

    def __init__(self, img_keys, lowdim_dim, seq_len, hidden_dim=256):
        super().__init__()
        self.img_keys   = list(img_keys)
        self.lowdim_dim = lowdim_dim
        self.seq_len    = seq_len

        self.img_encoders = nn.ModuleDict()
        for key in img_keys:
            resnet = models.resnet18(weights=None)
            self.img_encoders[key] = nn.Sequential(*list(resnet.children())[:-1])

        per_step_dim = len(img_keys) * self._IMG_OUT
        if lowdim_dim > 0:
            self.lowdim_encoder = nn.Sequential(
                nn.Linear(lowdim_dim, self._LOWDIM_OUT), nn.ReLU(),
                nn.Linear(self._LOWDIM_OUT, self._LOWDIM_OUT), nn.ReLU(),
            )
            per_step_dim += self._LOWDIM_OUT

        if per_step_dim == 0:
            raise ValueError('No input modalities — check obs_keys')

        self.gru  = nn.GRU(per_step_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        first = next(iter(obs.values()))
        B, T  = first.shape[:2]
        parts = []

        for key in self.img_keys:
            x    = obs[key].flatten(0, 1)                       # (B*T, C, H, W)
            feat = self.img_encoders[key](x).flatten(1)         # (B*T, _IMG_OUT)
            parts.append(feat.view(B, T, -1))

        if self.lowdim_dim > 0:
            ld   = obs['lowdim'].flatten(0, 1)                  # (B*T, D)
            parts.append(self.lowdim_encoder(ld).view(B, T, -1))

        x  = torch.cat(parts, dim=-1)                          # (B, T, per_step_dim)
        _, hn = self.gru(x)                                     # hn: (1, B, hidden_dim)
        return self.head(hn.squeeze(0)).squeeze(-1)             # (B,) logit


# ── Training ──────────────────────────────────────────────────────────────────

@click.command()
@click.option('--task',       '-t', default=None,
              help='Task name; infers zarr_path and output_dir automatically')
@click.option('--zarr_path',  '-z', default=None,
              help='Rollout zarr path (overrides --task)')
@click.option('--output_dir', '-o', default=None,
              help='Output directory for checkpoints and logs (overrides --task)')
@click.option('--obs_keys', default='wrist,dof_force,state',
              help='Comma-separated zarr obs keys to use as input')
@click.option('--seq_len',    default=4,    type=int,
              help='Sliding window length in timesteps')
@click.option('--device',     '-d', default='cuda:0')
@click.option('--epochs',     default=100,  type=int)
@click.option('--batch_size', default=64,  type=int)
@click.option('--lr',         default=1e-4, type=float)
@click.option('--val_ratio',  default=0.1,  type=float)
@click.option('--num_workers',default=4,    type=int)
@click.option('--pos_weight', default=None, type=float,
              help='BCEWithLogitsLoss positive-class weight; auto-computed from '
                   'class ratio if omitted')
@click.option('--hidden_dim', default=256,  type=int,
              help='GRU hidden size and classifier head width')
def main(task, zarr_path, output_dir, obs_keys, seq_len, device, epochs,
         batch_size, lr, val_ratio, num_workers, pos_weight, hidden_dim):
    if task is not None:
        zarr_path  = zarr_path  or f'alignment/data/{task}.zarr'
        output_dir = output_dir or f'models/{task}/classifiers'
    if zarr_path is None or output_dir is None:
        raise click.UsageError('Provide --task or both --zarr_path and --output_dir')

    obs_keys   = [k.strip() for k in obs_keys.split(',') if k.strip()]
    device     = torch.device(device)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path     = output_dir / 'best.ckpt'
    log_path      = output_dir / 'train_log.csv'
    settings_path = output_dir / 'settings.json'

    print(f'[clf] obs_keys={obs_keys}  seq_len={seq_len}')

    train_ds = ClassifierDataset(
        zarr_path, obs_keys, seq_len, val_ratio=val_ratio, seed=42)
    val_ds   = train_ds.get_validation_dataset()

    all_labels = train_ds.replay_buffer['success'][:, 0].astype(np.float32)
    n_pos = int(all_labels.sum())
    n_neg = len(all_labels) - n_pos
    print(f'[clf] buffer: {len(all_labels)} steps  success={n_pos}  failure={n_neg}')
    print(f'[clf] sequences: train={len(train_ds)}  val={len(val_ds)}')
    print(f'[clf] img_keys={train_ds.img_keys}  lowdim_keys={train_ds.lowdim_keys}')

    if pos_weight is None and n_pos > 0:
        pos_weight = float(n_neg) / n_pos
    if pos_weight is not None:
        print(f'[clf] pos_weight={pos_weight:.2f}')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    lowdim_dim = (
        sum(int(np.prod(train_ds.replay_buffer[k].shape[1:]))
            for k in train_ds.lowdim_keys)
        if train_ds.lowdim_keys else 0
    )
    model = SuccessClassifier(
        img_keys=train_ds.img_keys,
        lowdim_dim=lowdim_dim,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
    ).to(device)

    # Persist the full training configuration alongside the checkpoint so a run
    # is reproducible and the inference side knows exactly how inputs were routed.
    settings = {
        'task':        task,
        'zarr_path':   str(zarr_path),
        'obs_keys':    obs_keys,
        'img_keys':    train_ds.img_keys,
        'lowdim_keys': train_ds.lowdim_keys,
        'lowdim_dim':  lowdim_dim,
        'seq_len':     seq_len,
        'hidden_dim':  hidden_dim,
        'batch_size':  batch_size,
        'lr':          lr,
        'epochs':      epochs,
        'val_ratio':   val_ratio,
        'pos_weight':  pos_weight,
        'n_pos':       n_pos,
        'n_neg':       n_neg,
    }
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    print(f'[clf] settings → {settings_path}')

    pw        = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_acc,val_prec,val_rec,val_f1\n')

    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'epoch {epoch:3d}/{epochs} [train]', leave=False)
        for step, (obs, labels) in enumerate(pbar, 1):
            obs    = {k: v.to(device) for k, v in obs.items()}
            labels = labels.to(device)
            logits = model(obs)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f'{train_loss / step:.4f}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for obs, labels in tqdm(val_loader, desc=f'epoch {epoch:3d}/{epochs} [val]  ', leave=False):
                obs    = {k: v.to(device) for k, v in obs.items()}
                labels = labels.to(device)
                logits = model(obs)
                val_loss += criterion(logits, labels).item()
                all_preds.append((logits.sigmoid() > 0.5).cpu())
                all_tgts.append(labels.cpu())
        val_loss /= max(len(val_loader), 1)

        if all_preds:
            preds  = torch.cat(all_preds).float()
            tgts   = torch.cat(all_tgts).float()
            acc    = (preds == tgts).float().mean().item()
            tp = ((preds == 1) & (tgts == 1)).sum().item()
            fp = ((preds == 1) & (tgts == 0)).sum().item()
            fn = ((preds == 0) & (tgts == 1)).sum().item()
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        else:
            acc = prec = rec = f1 = float('nan')

        print(f'[clf] epoch {epoch:3d}/{epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                    f'{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'obs_keys':    obs_keys,
                'img_keys':    train_ds.img_keys,
                'lowdim_keys': train_ds.lowdim_keys,
                'lowdim_dim':  lowdim_dim,
                'seq_len':     seq_len,
                'hidden_dim':  hidden_dim,
                'lowdim_mean': train_ds.lowdim_mean,
                'lowdim_std':  train_ds.lowdim_std,
                'val_loss':    val_loss,
                'val_acc':     acc,
                'val_f1':      f1,
            }, ckpt_path)
            print(f'[clf]   ↳ best val={val_loss:.4f}  f1={f1:.3f}  saved → {ckpt_path}')

    print(f'[clf] done.  best val loss: {best_val:.4f}')


if __name__ == '__main__':
    main()
