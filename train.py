"""
train.py

Trains a single variant / excl_vol combination.
Called directly or via launch.py.

Usage
-----
python train.py --variant proj-single --excl_vol 01 --gpu 0
python train.py --variant proj-dual   --excl_vol 03 --gpu 1
python train.py --variant vol-dual    --excl_vol 05 --gpu 2
python train.py --variant vol-dual-z  --excl_vol 07 --gpu 3
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

from utilities.network import build_model

# ============================================================================
# CONFIGURATION
# ============================================================================

IM_DIR          = '/srv/shared/data/pixelprint'
IM_SIZE         = 128
INT_STEPS       = 7
ALL_VOLS        = ['01', '02', '03', '04', '05', '06', '07', '08']
SOURCE_PHASE    = '06'
PROJS_PER_PHASE = 397

STEPS_PER_EPOCH = 3000
TRAIN_CONFIG    = dict(epochs=50, lr=1e-5, batch_size=4)

LV_CLAMP = -3.0
LV_WARN  =  2.0

PROJ_VARIANTS = {'proj-single', 'proj-dual'}
VOL_VARIANTS  = {'vol-dual', 'vol-dual-z'}


def ckpt_path(variant, excl_vol):
    return os.path.join('weights', f'{variant}_excl{excl_vol}_best.pth')

def plot_path(variant, excl_vol):
    return os.path.join('plots', f'{variant}_excl{excl_vol}_loss.png')

def log_path(variant, excl_vol):
    return os.path.join('logs', f'{variant}_excl{excl_vol}.log')


# ============================================================================
# NORMALISATION
# ============================================================================

def compute_global_stats(excl_vol):
    phases = [v for v in ALL_VOLS if v != excl_vol]
    print('Computing global normalisation stats...')
    vol_min, vol_max = np.inf, -np.inf
    prj_min, prj_max = np.inf, -np.inf
    for phase in phases:
        v = np.load(os.path.join(IM_DIR, f'sub_CT_{phase}_mha.npy'))
        vol_min = min(vol_min, v.min())
        vol_max = max(vol_max, v.max())
    for phase in phases:
        for n in [1, PROJS_PER_PHASE // 2, PROJS_PER_PHASE]:
            fp = os.path.join(IM_DIR, f'{phase}_proj_{n:05d}_bin.npy')
            if os.path.exists(fp):
                p = np.load(fp)
                prj_min = min(prj_min, p.min())
                prj_max = max(prj_max, p.max())
    print(f'  Vol:  [{vol_min:.4f}, {vol_max:.4f}]')
    print(f'  Proj: [{prj_min:.4f}, {prj_max:.4f}]')
    return dict(vol_min=vol_min, vol_max=vol_max, prj_min=prj_min, prj_max=prj_max)

def _norm_vol(x, s):
    return (x - s['vol_min']) / (s['vol_max'] - s['vol_min'] + 1e-7)

def _norm_prj(x, s):
    return (x - s['prj_min']) / (s['prj_max'] - s['prj_min'] + 1e-7)

def _load_raw_proj(phase, proj_num, stats):
    fp = os.path.join(IM_DIR, f'{phase}_proj_{proj_num:05d}_bin.npy')
    return torch.from_numpy(
        _norm_prj(np.load(fp), stats).astype(np.float32)
    ).unsqueeze(0)

def _load_vol_tensor(phase, stats):
    arr = _norm_vol(np.load(os.path.join(IM_DIR, f'sub_CT_{phase}_mha.npy')), stats)
    return torch.from_numpy(
        arr.reshape(1, IM_SIZE, IM_SIZE, IM_SIZE).astype(np.float32)
    )


# ============================================================================
# DATASET
# ============================================================================

class ProjectionDataset(Dataset):
    """Samples random (phase, projection) pairs from the training phases."""

    def __init__(self, excl_vol, stats, steps_per_epoch=STEPS_PER_EPOCH):
        self.stats           = stats
        self.steps_per_epoch = steps_per_epoch
        self.phases          = [p for p in ALL_VOLS if p != excl_vol and p != SOURCE_PHASE]

        print('Loading source volume and target volumes...')
        self.source_vol  = _load_vol_tensor(SOURCE_PHASE, stats)
        self.target_vols = {p: _load_vol_tensor(p, stats) for p in self.phases}

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        phase    = random.choice(self.phases)
        proj_num = random.randint(1, PROJS_PER_PHASE)
        return {
            'source_proj': _load_raw_proj(SOURCE_PHASE, proj_num, self.stats),
            'target_proj': _load_raw_proj(phase,        proj_num, self.stats),
            'source_vol':  self.source_vol.clone(),
            'target_vol':  self.target_vols[phase].clone(),
        }


# ============================================================================
# LOSS
# ============================================================================

def compute_loss(variant, mdl, y_source, y_cycle, target_vol):
    """
    proj-single : L1(warped, target)
    proj-dual / vol-dual / vol-dual-z : uncertainty-weighted L1 for both outputs
    """
    warp_loss = F.l1_loss(y_source, target_vol)

    if variant == 'proj-single':
        return warp_loss, dict(total=warp_loss.item(), warp=warp_loss.item())

    cycle_loss = F.l1_loss(y_cycle, target_vol)
    lv_w = torch.clamp(mdl.log_var_dvf, min=LV_CLAMP)
    lv_c = torch.clamp(mdl.log_var_img, min=LV_CLAMP)
    total = (0.5 * torch.exp(-lv_w) * warp_loss + 0.5 * lv_w +
             0.5 * torch.exp(-lv_c) * cycle_loss + 0.5 * lv_c)
    return total, dict(
        total=total.item(), warp=warp_loss.item(), cycle=cycle_loss.item(),
        lv_warp=lv_w.item(), lv_cycle=lv_c.item(),
    )


# ============================================================================
# FORWARD PASS  (unified across proj / vol variants)
# ============================================================================

def forward(variant, mdl, batch, device):
    src_proj = batch['source_proj'].to(device)
    tgt_proj = batch['target_proj'].to(device)
    src_vol  = batch['source_vol'].to(device)
    tgt_vol  = batch['target_vol'].to(device)

    if variant in PROJ_VARIANTS:
        out = mdl(src_proj, tgt_proj, src_vol)
    else:
        out = mdl(src_vol, tgt_proj)

    y_source = out[0]
    y_cycle  = out[2]   # None for proj-single
    return y_source, y_cycle, tgt_vol


# ============================================================================
# EPOCH RUNNER
# ============================================================================

def run_epoch(variant, mdl, loader, device, optimizer=None):
    is_train = optimizer is not None
    mdl.train() if is_train else mdl.eval()
    accum, n = {}, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            y_source, y_cycle, tgt_vol = forward(variant, mdl, batch, device)
            loss, metrics = compute_loss(variant, mdl, y_source, y_cycle, tgt_vol)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
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
    for key in ['train_warp', 'val_warp', 'train_cycle', 'val_cycle']:
        if key in history:
            ax.plot(history[key], label=key, linestyle='--' if 'val' in key else '-')
    ax.set_xlabel('Epoch'); ax.set_ylabel('L1 loss'); ax.set_title(title); ax.legend()

    ax2 = axes[1]
    for key in ['train_lv_warp', 'val_lv_warp', 'train_lv_cycle', 'val_lv_cycle']:
        if key in history:
            ax2.plot(history[key], label=key, linestyle='--' if 'val' in key else '-')
    if any(k.startswith('train_lv') for k in history):
        ax2.axhline(LV_WARN,  color='red',  linestyle=':', linewidth=1, label=f'warn ({LV_WARN})')
        ax2.axhline(LV_CLAMP, color='blue', linestyle=':', linewidth=1, label=f'clamp ({LV_CLAMP})')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('log σ²'); ax2.set_title('Uncertainty weights'); ax2.legend()

    plt.tight_layout(); plt.savefig(path); plt.close()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(variant, excl_vol, device):
    cfg = TRAIN_CONFIG
    print(f'\n{"=" * 60}')
    print(f'  Variant: {variant}   Excl: {excl_vol}')
    print(f'  Epochs: {cfg["epochs"]}   LR: {cfg["lr"]}   Device: {device}')
    print(f'{"=" * 60}\n')

    stats   = compute_global_stats(excl_vol)
    dataset = ProjectionDataset(excl_vol, stats)
    mdl     = build_model(variant, im_size=IM_SIZE, int_steps=INT_STEPS).to(device)
    opt     = optim.Adam(mdl.parameters(), lr=cfg['lr'])

    history = {}
    best    = float('inf')
    tic     = time.time()

    def _split(ds, frac=0.9):
        n = int(len(ds) * frac)
        return random_split(ds, [n, len(ds) - n])

    def _loader(ds, shuffle=True):
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    for epoch in range(1, cfg['epochs'] + 1):
        tr_set, vl_set = _split(dataset)
        tr = run_epoch(variant, mdl, _loader(tr_set),            device, opt)
        vl = run_epoch(variant, mdl, _loader(vl_set, False),     device)

        for k, v in tr.items(): history.setdefault(f'train_{k}', []).append(v)
        for k, v in vl.items(): history.setdefault(f'val_{k}',   []).append(v)

        for k, v in tr.items():
            if k.startswith('lv_') and v > LV_WARN:
                print(f'  WARNING epoch {epoch}: {k}={v:.2f}')

        elapsed = (time.time() - tic) / 3600
        main_metrics = ' '.join(f'{k}:{v:.4f}' for k, v in tr.items() if not k.startswith('lv_'))
        val_metrics  = ' '.join(f'{k}:{v:.4f}' for k, v in vl.items() if not k.startswith('lv_'))
        line = f'[{elapsed:.2f}h] Epoch {epoch:3d} | {main_metrics}  ||  {val_metrics}'
        print(line)
        sys.stdout.flush()

        metric = vl.get('warp', 0.) + vl.get('cycle', 0.)
        if metric < best:
            best = metric
            torch.save({'model': mdl.state_dict()}, ckpt_path(variant, excl_vol))

        _save_plot(history, f'{variant} / excl{excl_vol}', plot_path(variant, excl_vol))

    print(f'\nDone. Best val metric: {best:.4f}')
    print(f'Checkpoint: {ckpt_path(variant, excl_vol)}')


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',  required=True,
                        choices=['proj-single', 'proj-dual', 'vol-dual', 'vol-dual-z'])
    parser.add_argument('--excl_vol', required=True, choices=ALL_VOLS)
    parser.add_argument('--gpu',      type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    for d in ('weights', 'plots', 'logs'):
        os.makedirs(d, exist_ok=True)

    # Redirect stdout/stderr to log file (tee to console via Tee)
    lp = log_path(args.variant, args.excl_vol)

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
        train(args.variant, args.excl_vol, device)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == '__main__':
    main()
