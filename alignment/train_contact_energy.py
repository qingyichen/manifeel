"""
Reference implementation: a controllable contact-success energy E(z_contact).

Design decisions encoded here (from the proposal):
  - Encoder g_tac: window of tactile/wrench frames -> low-dim latent z (8-32 dims).
  - Energy head: distance-to-prototype, E(z) = softmin_k ||z - c_k||^2.
    This *structurally* gives a bowl whose minimum is a learned success region
    (a set, not a point), instead of hoping a free MLP discovers one.
  - Four losses:
      L_disc : margin separation success vs failure contact (NOT plain BCE -
               BCE only carves a boundary and leaves the interior gradient-free).
      L_ord  : ordinal ranking from the TIME-TO-EVENT surrogate (the only
               monotonicity source available from per-step BINARY labels).
      L_con  : InfoNCE-style contrastive -> compact, nuisance-invariant success region.
      L_lip  : gradient penalty -> smooth landscape, no cliffs (controllability).
  - Monotonicity go/no-go: does descending E predict the binary flip? (checkable
    with exactly the binary labels you have; gates the control phase).

Labels assumed (the realistic case): per-step BINARY success indicator that flips
0->1 at seating. From successful trajectories we derive a time-to-event progress
surrogate; we do NOT assume graded/geometric progress.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- 
# Encoder: window of contact frames -> latent z
# Swap the per-frame encoder for a small CNN (TacFF/TacRGB) or MLP (wrench).
# Here frames are already feature vectors of dim `frame_dim` for generality.
# -----------------------------------------------------------------------------
class ContactEncoder(nn.Module):
    def __init__(self, frame_dim, latent_dim=16, hidden=128, window=8):
        super().__init__()
        self.window = window
        # temporal conv over the window; force *signatures* are sequential
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, latent_dim)

    def forward(self, x):                 # x: [B, W, frame_dim]
        B, W, D = x.shape
        h = self.frame_mlp(x.reshape(B * W, D)).reshape(B, W, -1)
        _, hn = self.temporal(h)          # hn: [1, B, hidden]
        z = self.head(hn.squeeze(0))      # [B, latent_dim]
        return z


# -----------------------------------------------------------------------------
# Energy head: distance to nearest learned prototype (success region = the c_k).
# softmin (logsumexp) keeps it differentiable and smooth.
# E >= 0, low near the success prototypes, rising smoothly away from them.
# -----------------------------------------------------------------------------
class PrototypeEnergy(nn.Module):
    def __init__(self, latent_dim=16, n_proto=4, temp=1.0):
        super().__init__()
        self.protos = nn.Parameter(torch.randn(n_proto, latent_dim) * 0.5)
        self.temp = temp

    def forward(self, z):                 # z: [B, latent_dim] -> E: [B]
        d2 = torch.cdist(z, self.protos) ** 2          # [B, n_proto]
        # softmin over prototypes = -temp * logsumexp(-d2/temp); ~min distance, smooth
        E = -self.temp * torch.logsumexp(-d2 / self.temp, dim=1)
        return E


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def loss_disc(E, success, margin=1.0):
    """Margin separation. Successes pushed to low E, failures above a margin.
    Hinge keeps gradient on the *interior* (unlike BCE which saturates)."""
    E_succ = E[success == 1]
    E_fail = E[success == 0]
    l = E.new_tensor(0.0)
    if E_succ.numel() > 0:
        l = l + E_succ.clamp(min=0).mean()                 # successes -> ~0
    if E_fail.numel() > 0:
        l = l + F.relu(margin - E_fail).mean()             # failures  -> >= margin
    return l


def loss_ord(E, traj_id, t2e, traj_succeeds, margin=0.05):
    """Ordinal ranking from the time-to-event surrogate.

    KEY: the progress signal lives in the frames *leading up to* the flip, which
    carry the per-step label 0. So we rank over all frames of trajectories that
    EVENTUALLY succeed (traj_succeeds==1), using t2e = steps-until-flip (>0 before
    the flip, 0 at/after). A frame closer to the flip must have <= energy than a
    farther one. This is the sole monotonicity source given binary labels."""
    l = E.new_tensor(0.0)
    n = 0
    for tid in traj_id.unique():
        m = (traj_id == tid) & (traj_succeeds == 1)
        if m.sum() < 2:
            continue
        Em, tm = E[m], t2e[m]
        diff_t = tm.unsqueeze(0) - tm.unsqueeze(1)
        diff_E = Em.unsqueeze(0) - Em.unsqueeze(1)
        closer = (diff_t < 0).float()                      # i closer to flip than j
        viol = closer * F.relu(margin + diff_E)            # penalize closer-but-higher-E
        if closer.sum() > 0:
            l = l + viol.sum() / closer.sum()
            n += 1
    return l / max(n, 1)


def loss_con(z, success, temp=0.2):
    """InfoNCE: success frames are positives for each other; failures are negatives.
    Compacts the success region and makes it nuisance-invariant."""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temp                                 # [B,B]
    B = z.shape[0]
    sim.fill_diagonal_(-1e9)
    pos_mask = (success.unsqueeze(0) == success.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)
    # only anchor on success frames (we care about a tight success region)
    anchors = (success == 1)
    if anchors.sum() == 0:
        return z.new_tensor(0.0)
    logp = F.log_softmax(sim, dim=1)
    num_pos = pos_mask[anchors].sum(1).clamp(min=1)
    l = -(pos_mask[anchors] * logp[anchors]).sum(1) / num_pos
    return l.mean()


def loss_lip(energy_fn, z, weight=1.0):
    """Gradient penalty on E w.r.t. z -> bounded slope, smooth descent landscape."""
    z = z.detach().clone().requires_grad_(True)
    E = energy_fn(z)
    g = torch.autograd.grad(E.sum(), z, create_graph=True)[0]
    return weight * (g.norm(dim=1) ** 2).mean()


# -----------------------------------------------------------------------------
# Monotonicity go/no-go check (run after training, BEFORE any control RL).
# With only binary labels we can't plot E vs graded progress, so instead:
# does lower E predict "flip within next k steps"? Report AUROC-like separation.
# -----------------------------------------------------------------------------
@torch.no_grad()
def monotonicity_check(E, t2e, traj_succeeds):
    """Among the APPROACH frames of succeeding trajectories (t2e>0), does E rise
    with distance-to-flip? Returns rank correlation between E and t2e (want
    strongly +: high E <-> far from flip, low E <-> near flip => descendable)."""
    m = (traj_succeeds == 1) & (t2e > 0)
    if m.sum() < 3:
        return float('nan')
    e = E[m]; t = t2e[m]
    er = e.argsort().argsort().float()
    tr = t.argsort().argsort().float()
    er = (er - er.mean()) / (er.std() + 1e-8)
    tr = (tr - tr.mean()) / (tr.std() + 1e-8)
    return (er * tr).mean().item()


# -----------------------------------------------------------------------------
# Demo on synthetic data shaped like the real problem, to verify it runs and
# that the losses produce a descendable, separating energy.
# -----------------------------------------------------------------------------
def _make_synthetic(n_traj=200, T=20, frame_dim=12, window=8, seed=0):
    """Each trajectory: a contact 'signature' drifting toward either a success
    attractor or a failure region. Successful trajectories flip at some t*.
    Returns per-step binary label y (post-flip=1), per-frame trajectory id,
    time-to-event t2e (steps until flip; 0 at/after flip; large for failures),
    and a per-frame flag traj_succeeds (does this frame's trajectory eventually
    flip)."""
    g = torch.Generator().manual_seed(seed)
    Xs, ys, tids, t2es, tsucc = [], [], [], [], []
    succ_center = torch.randn(frame_dim, generator=g)
    fail_center = torch.randn(frame_dim, generator=g) + 3.0
    for i in range(n_traj):
        is_succ = (torch.rand(1, generator=g) < 0.5).item()
        tstar = int(torch.randint(window + 2, T, (1,), generator=g)) if is_succ else None
        target = succ_center if is_succ else fail_center
        start = torch.randn(frame_dim, generator=g) + 1.5
        for t in range(window, T):
            alpha = (t - window) / (T - window)
            cur = (1 - alpha) * start + alpha * target
            frame_seq = cur.unsqueeze(0).repeat(window, 1)
            frame_seq = frame_seq + 0.15 * torch.randn(window, frame_dim, generator=g)
            Xs.append(frame_seq); tids.append(i)
            if is_succ:
                ys.append(int(t >= tstar))
                t2es.append(max(tstar - t, 0))   # >0 on the approach, 0 at/after flip
                tsucc.append(1)
            else:
                ys.append(0)
                t2es.append(float(T))            # failures: effectively "never"
                tsucc.append(0)
    return (torch.stack(Xs), torch.tensor(ys), torch.tensor(tids),
            torch.tensor(t2es).float(), torch.tensor(tsucc))


def main():
    torch.manual_seed(0)
    X, y, tid, t2e, tsucc = _make_synthetic()
    frame_dim, window = X.shape[2], X.shape[1]
    enc = ContactEncoder(frame_dim, latent_dim=16, window=window)
    energy = PrototypeEnergy(latent_dim=16, n_proto=4)
    opt = torch.optim.Adam(list(enc.parameters()) + list(energy.parameters()), lr=2e-3)

    for step in range(400):
        z = enc(X)
        E = energy(z)
        Ld = loss_disc(E, y)
        Lo = loss_ord(E, tid, t2e, tsucc)
        Lc = loss_con(z, y)
        Ll = loss_lip(energy, z, weight=0.1)
        # Ordinal early (before contrastive collapses the approach); contrastive light.
        w_ord = min(1.0, step / 30) * 2.0
        w_con = 0.1
        loss = Ld + w_ord * Lo + w_con * Lc + Ll
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 399:
            with torch.no_grad():
                z = enc(X); E = energy(z)
                sep = E[y == 0].mean() - E[y == 1].mean()
            mono = monotonicity_check(E, t2e, tsucc)
            print(f"step {step:3d} | loss {loss.item():.3f} "
                  f"| Ld {Ld.item():.3f} Lo {Lo.item():.3f} Lc {Lc.item():.3f} Ll {Ll.item():.3f} "
                  f"| E_fail-E_succ {sep.item():+.3f} | mono(E,t2e) {mono:+.3f}")

    print("\nInterpretation:")
    print("  E_fail-E_succ > 0  : energy separates failure (high) from success (low).")
    print("  mono(E,t2e)   > 0  : within successes, E rises with distance-to-flip")
    print("                       => E is DESCENDABLE => control phase is unlocked.")


if __name__ == "__main__":
    main()