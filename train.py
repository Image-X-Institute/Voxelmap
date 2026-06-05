"""
train_irb.py

Self-contained trainer for the IRB ablation. Own config, normalisation,
dataset, masked losses, and deep supervision. The only cross-file reference is
`build_model` (which model to construct) imported from network_irb.

Modes / depth
-------------
  --mode baseline                 initial decoder only (no IRB)
  --mode dvf   --num_irb {1,2,3}  + N shared-weight DVF-only IRBs
  --mode dual  --num_irb {1,2,3}  + N shared-weight dual (DVF + bounded image) IRBs
  --mode bigfly                   deeper one-shot decoder, no recurrence (control)

Losses (all reconstruction terms computed INSIDE the thoracic mask)
-------------------------------------------------------------------
  * Deep-supervision reconstruction: gamma-weighted masked L1 over y_steps,
    later steps weighted more heavily (weight gamma^(T - t)).
  * DVF smoothness: gradient penalty on the final flow (unmasked; smoothness is
    wanted across the boundary too).
  * Bounded image-residual penalty (dual only): masked L1 of each dimg toward 0.

Usage
-----
python train_irb.py --mode baseline               --excl_vol 01 --gpu 0
python train_irb.py --mode dvf   --num_irb 2       --excl_vol 01 --gpu 1
python train_irb.py --mode dual  --num_irb 3       --excl_vol 01 --gpu 2
python train_irb.py --mode bigfly                  --excl_vol 01 --gpu 3
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

from network_irb import build_model

# ============================================================================
# CONFIGURATION
# ============================================================================

IM_DIR          = '/srv/shared/data/pixelprint'
MASK_FILE       = 'sub_CorrectMask_mha.npy'   # in the working directory
IM_SIZE         = 128
INT_STEPS       = 7
ALL_VOLS        = ['01', '02', '03', '04', '05', '06', '07', '08']
SOURCE_PHASE    = '06'
PROJS_PER_PHASE = 397
TRAIN_CONFIG    = dict(epochs=100, lr=1e-5, batch_size=4)

IMG_EPS      = 0.05    # bound on the per-step image residual: eps * tanh(.)
DS_GAMMA     = 0.8     # deep-supervision decay: step t weight = gamma^(T - t)
                       # (RAFT default; step axis uses a fixed ordering prior,
                       #  term axis uses learned uncertainty weighting instead)

# Optional: freeze the dual image head for the first K epochs so the DVF learns
# to do the geometric work before the image residual is allowed to contribute.
IMG_FREEZE_EPOCHS = 5


def tag(mode, num_irb):
    return mode if mode in ('baseline', 'bigfly') else f'{mode}{num_irb}'

def ckpt_path(mode, num_irb, excl_vol):
    return os.path.join('weights', f'{tag(mode, num_irb)}_excl{excl_vol}_best.pth')

def plot_path(mode, num_irb, excl_vol):
    return os.path.join('plots', f'{tag(mode, num_irb)}_excl{excl_vol}_loss.png')

def log_path(mode, num_irb, excl_vol):
    return os.path.join('logs', f'{tag(mode, num_irb)}_excl{excl_vol}.log')


# ============================================================================
# NORMALISATION  +  MASK
# ============================================================================

def compute_global_stats(excl_vol):
    phases = [v for v in ALL_VOLS if v != excl_vol]
    print('Computing global normalisation stats...')
    vol_min, vol_max = np.inf, -np.inf
    prj_min, prj_max = np.inf, -np.inf
    for phase in phases:
        v = np.load(os.path.join(IM_DIR, f'sub_CT_{phase}_mha.npy'))
        vol_min = min(vol_min, v.min()); vol_max = max(vol_max, v.max())
    for phase in phases:
        for n in [1, PROJS_PER_PHASE // 2, PROJS_PER_PHASE]:
            fp = os.path.join(IM_DIR, f'{phase}_proj_{n:05d}_bin.npy')
            if os.path.exists(fp):
                p = np.load(fp)
                prj_min = min(prj_min, p.min()); prj_max = max(prj_max, p.max())
    print(f'  Vol:  [{vol_min:.4f}, {vol_max:.4f}]')
    print(f'  Proj: [{prj_min:.4f}, {prj_max:.4f}]')
    return dict(vol_min=vol_min, vol_max=vol_max, prj_min=prj_min, prj_max=prj_max)

def _norm_vol(x, s):
    return (x - s['vol_min']) / (s['vol_max'] - s['vol_min'] + 1e-7)

def _norm_prj(x, s):
    return (x - s['prj_min']) / (s['prj_max'] - s['prj_min'] + 1e-7)

def _load_raw_proj(phase, proj_num, stats):
    fp = os.path.join(IM_DIR, f'{phase}_proj_{proj_num:05d}_bin.npy')
    return torch.from_numpy(_norm_prj(np.load(fp), stats).astype(np.float32)).unsqueeze(0)

def _load_vol_tensor(phase, stats):
    arr = _norm_vol(np.load(os.path.join(IM_DIR, f'sub_CT_{phase}_mha.npy')), stats)
    return torch.from_numpy(arr.reshape(1, IM_SIZE, IM_SIZE, IM_SIZE).astype(np.float32))

def load_mask_tensor():
    """Binary thoracic mask -> float tensor [1,1,D,H,W]."""
    m = np.load(MASK_FILE).astype(np.float32).reshape(1, 1, IM_SIZE, IM_SIZE, IM_SIZE)
    return torch.from_numpy(m)


# ============================================================================
# DATASET
# ============================================================================

class ProjectionDataset(Dataset):
    def __init__(self, excl_vol, stats):
        self.stats  = stats
        self.phases = [p for p in ALL_VOLS if p != excl_vol and p != SOURCE_PHASE]
        print('Loading source volume and target volumes...')
        self.source_vol  = _load_vol_tensor(SOURCE_PHASE, stats)
        self.target_vols = {p: _load_vol_tensor(p, stats) for p in self.phases}

    def __len__(self):
        return len(self.phases) * PROJS_PER_PHASE

    def __getitem__(self, idx):
        phase    = random.choice(self.phases)
        proj_num = random.randint(1, PROJS_PER_PHASE)
        return {
            'target_proj': _load_raw_proj(phase, proj_num, self.stats),
            'source_vol':  self.source_vol.clone(),
            'target_vol':  self.target_vols[phase].clone(),
        }


# ============================================================================
# MASKED LOSS HELPERS
# ============================================================================

def masked_l1(pred, target, mask):
    """Mean |pred - target| over masked voxels (per-batch safe)."""
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

def masked_l1_to_zero(x, mask):
    return (x.abs() * mask).sum() / mask.sum().clamp_min(1.0)


# ============================================================================
# LOSS
# ============================================================================
#
# Two weighting axes, each with the appropriate tool:
#
#  * STEP axis (deep supervision): fixed gamma^(T - t) prior. Steps are an
#    ordered refinement SEQUENCE that we WANT to improve monotonically, not
#    competing observations -- so a fixed ordering prior is correct here, and
#    UW would wrongly down-weight early steps for having higher (expected) loss.
#
#  * TERM axis (motion vs image): uncertainty weighting via learned log-vars
#    on the model. The clean-motion reconstruction and the motion+bounded-image
#    reconstruction ARE competing observations of the same target, so UW is the
#    right tool and removes the hand-set term weights.
#
# No smoothness term: scaling-and-squaring integration yields diffeomorphic
# (non-folding) fields, so an explicit smoothness penalty is redundant.

def _uw_term(loss, log_var):
    """Uncertainty-weighted term: 0.5 * exp(-lv) * loss + 0.5 * lv."""
    lv = torch.clamp(log_var, min=LV_CLAMP)
    return 0.5 * torch.exp(-lv) * loss + 0.5 * lv, lv

def compute_loss(out, target_vol, mask, mode):
    """
    gamma-weighted deep-supervision reconstruction, with the FINAL output's
    motion vs image terms balanced by uncertainty weighting. Returns
    (total, metrics_dict). All reconstruction L1s are masked.
    """
    y_steps = out['y_steps']
    T = len(y_steps) - 1   # refinement steps (0 for baseline/bigfly)

    # ── Deep supervision over steps: fixed gamma prior (normalised) ────────
    weights = [DS_GAMMA ** (T - t) for t in range(len(y_steps))]
    wsum    = sum(weights)
    recon   = sum(w * masked_l1(y, target_vol, mask)
                  for w, y in zip(weights, y_steps)) / wsum

    metrics = dict(recon=float(recon.item()))
    metrics['recon_final'] = float(masked_l1(out['y_final'], target_vol, mask).item())

    if mode == 'dual':
        # ── UW over the two competing FINAL reconstructions ────────────────
        warp_loss  = masked_l1(out['y_flow_final'], target_vol, mask)  # clean motion
        img_loss   = masked_l1(out['y_final'],      target_vol, mask)  # + bounded image
        uw_w, lv_w = _uw_term(warp_loss, out['log_var_warp'])
        uw_i, lv_i = _uw_term(img_loss,  out['log_var_img'])

        # Deep supervision provides the across-step signal; UW balances the two
        # final-output terms. Sum them.
        total = recon + uw_w + uw_i
        metrics.update(
            warp=float(warp_loss.item()), img=float(img_loss.item()),
            lv_warp=float(lv_w.item()),   lv_img=float(lv_i.item()),
        )
    else:
        # Single reconstruction objective (motion only). UW on one term is
        # degenerate, so use plain deep-supervision recon.
        total = recon

    metrics['total'] = float(total.item())
    return total, metrics


# ============================================================================
# EPOCH RUNNER
# ============================================================================

def run_epoch(mdl, loader, device, mask, mode, optimizer=None, freeze_img=False):
    is_train = optimizer is not None
    mdl.train() if is_train else mdl.eval()

    # Optionally freeze the dual image head (warm-up the DVF first).
    if mode == 'dual' and mdl.irb is not None:
        for p in mdl.irb.img_up.parameters():   p.requires_grad = not freeze_img
        for p in mdl.irb.img_head.parameters(): p.requires_grad = not freeze_img

    accum, n = {}, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            tgt_proj = batch['target_proj'].to(device)
            src_vol  = batch['source_vol'].to(device)
            tgt_vol  = batch['target_vol'].to(device)

            out = mdl(src_vol, tgt_proj)
            loss, metrics = compute_loss(out, tgt_vol, mask, mode)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step()

            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.) + v
            n += 1
    return {k: v / n for k, v in accum.items()}


# ============================================================================
# PLOTTING
# ============================================================================

def _save_plot(history, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    ax = axes[0]
    for key in ['train_recon_final', 'val_recon_final', 'train_recon', 'val_recon']:
        if key in history:
            ax.plot(history[key], label=key, linestyle='--' if 'val' in key else '-')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Masked L1'); ax.set_title(title); ax.legend()

    ax2 = axes[1]
    for key in ['train_lv_warp', 'val_lv_warp', 'train_lv_img', 'val_lv_img']:
        if key in history:
            ax2.plot(history[key], label=key, linestyle='--' if 'val' in key else '-')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('log σ²')
    ax2.set_title('Uncertainty weights (motion vs image)'); ax2.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(mode, num_irb, excl_vol, device):
    cfg = TRAIN_CONFIG
    print(f'\n{"=" * 60}')
    print(f'  Mode: {mode}   num_irb: {num_irb}   Excl: {excl_vol}')
    print(f'  Epochs: {cfg["epochs"]}   LR: {cfg["lr"]}   Device: {device}')
    print(f'{"=" * 60}\n')

    stats   = compute_global_stats(excl_vol)
    dataset = ProjectionDataset(excl_vol, stats)
    mask    = load_mask_tensor().to(device)
    mdl     = build_model(mode, num_irb=num_irb, im_size=IM_SIZE,
                          int_steps=INT_STEPS, img_eps=IMG_EPS).to(device)
    opt     = optim.Adam(mdl.parameters(), lr=cfg['lr'])

    history, best, tic = {}, float('inf'), time.time()

    def _split(ds, frac=0.9):
        n = int(len(ds) * frac); return random_split(ds, [n, len(ds) - n])

    def _loader(ds, shuffle=True):
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    for epoch in range(1, cfg['epochs'] + 1):
        freeze_img = (mode == 'dual') and (epoch <= IMG_FREEZE_EPOCHS)
        tr_set, vl_set = _split(dataset)
        tr = run_epoch(mdl, _loader(tr_set),        device, mask, mode, opt, freeze_img)
        vl = run_epoch(mdl, _loader(vl_set, False), device, mask, mode)

        for k, v in tr.items(): history.setdefault(f'train_{k}', []).append(v)
        for k, v in vl.items(): history.setdefault(f'val_{k}',   []).append(v)

        elapsed = (time.time() - tic) / 3600
        tr_s = ' '.join(f'{k}:{v:.4f}' for k, v in tr.items())
        vl_s = ' '.join(f'{k}:{v:.4f}' for k, v in vl.items())
        fz   = '  [img frozen]' if freeze_img else ''
        print(f'[{elapsed:.2f}h] Epoch {epoch:3d} | {tr_s}  ||  {vl_s}{fz}')
        sys.stdout.flush()

        metric = vl.get('recon_final', vl.get('recon', 0.))
        if metric < best:
            best = metric
            torch.save({'model': mdl.state_dict(),
                        'mode': mode, 'num_irb': num_irb}, ckpt_path(mode, num_irb, excl_vol))

        _save_plot(history, f'{tag(mode, num_irb)} / excl{excl_vol}',
                   plot_path(mode, num_irb, excl_vol))

    print(f'\nDone. Best val recon_final: {best:.4f}')
    print(f'Checkpoint: {ckpt_path(mode, num_irb, excl_vol)}')


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',     required=True,
                        choices=['baseline', 'dvf', 'dual', 'bigfly'])
    parser.add_argument('--num_irb',  type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--excl_vol', required=True, choices=ALL_VOLS)
    parser.add_argument('--gpu',      type=int, default=0)
    args = parser.parse_args()

    if args.mode in ('dvf', 'dual') and args.num_irb == 0:
        parser.error(f'--mode {args.mode} requires --num_irb >= 1')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    for d in ('weights', 'plots', 'logs'):
        os.makedirs(d, exist_ok=True)

    lp = log_path(args.mode, args.num_irb, args.excl_vol)

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data)
        def flush(self):
            for s in self.streams: s.flush()

    log_file   = open(lp, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    try:
        train(args.mode, args.num_irb, args.excl_vol, device)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == '__main__':
    main()
