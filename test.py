import argparse
import os
from dataclasses import fields
import numpy as np
from utilities.config import Config
from matplotlib import pyplot as plt

P = argparse.ArgumentParser()
P.add_argument('ckpt', help='checkpoint written by train.py')
P.add_argument('--gpu', type=int, default=0)
P.add_argument('--stride', type=int, default=17, help='evaluate every n-th held-out view')
P.add_argument('--n-fig', type=int, default=3, help='views shown in the figure')
A = P.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(A.gpu)

import torch
import torch.nn.functional as F
from utilities import data as D
from utilities.model import BSplineModel, MotionEnc, base_grid, predict, warp
from utilities.projector import Projector

dev = 'cuda'
ck = torch.load(A.ckpt, map_location=dev, weights_only=False)
names = {f.name for f in fields(Config)}
cfg = Config(**{k: v for k, v in ck['config'].items() if k in names})

# ---- rebuild the scan and the trained networks ---------------------------
stats = D.projection_stats(cfg.case)
pr_full = Projector(D.geometry(), D.VOL_ZYX, D.VOX_MM, D.DET_VU, D.DET_SPACING)
y = D.simulate_arc(cfg.case, pr_full, stats, dev, cfg.n_breaths, cfg.noise_i0, cfg.seed)
zyx = tuple(s // cfg.bin for s in D.VOL_ZYX)


def load(b):
    v = torch.from_numpy(D.load_volume(cfg.case, b))[None, None].to(dev)
    return F.avg_pool3d(v, cfg.bin, cfg.bin)


iref = load(D.REF_BIN)
grid0 = base_grid(zyx, dev)
model = BSplineModel(zyx, D.HALF_MM_XYZ[::-1], cfg.base_grid, cfg.levels).to(dev)
model.load_state_dict(ck['model'])
enc = MotionEnc(D.DET_VU, max_ch=cfg.enc_max_ch, n_comp=cfg.levels).to(dev)
enc.load_state_dict(ck['enc'])

# ---- held-out views: one projection -> volume, against the true phase ----
held = np.setdiff1d(np.arange(D.N_ANGLES), ck['views'])[::A.stride]
vols = {}
err, base, sl = [], [], []
mid = zyx[0] // 2           # coronal slice through the middle of the AP extent
for a in held:
    b = D.phase_of(a, D.N_ANGLES, cfg.n_breaths)
    if b not in vols:
        vols[b] = load(b)
    est = warp(iref, predict(enc, model, y[a:a + 1].to(dev)), grid0)
    err.append(float((est - vols[b]).pow(2).mean().sqrt()))
    base.append(float((iref - vols[b]).pow(2).mean().sqrt()))
    sl.append((a, b, est[0, 0, mid].cpu().numpy(), vols[b][0, 0, mid].cpu().numpy()))
    print(f'view {a:4d}  T{10 * (b - 1):02d}  rmse {err[-1]:.5f}  (static {base[-1]:.5f})', flush=True)

print(f'{len(held)} views: mean rmse {np.mean(err):.5f} vs static reference {np.mean(base):.5f}')

# ---- figure: the views furthest from the reference, one per phase -------
pick, seen = [], set()
for i in np.argsort(base)[::-1]:
    if sl[i][1] not in seen:
        pick.append(i)
        seen.add(sl[i][1])
    if len(pick) == A.n_fig:
        break
ref = iref[0, 0, mid].cpu().numpy()
mm = D.VOX_MM * cfg.bin
ext = [-zyx[2] * mm / 2, zyx[2] * mm / 2, -zyx[1] * mm / 2, zyx[1] * mm / 2]
vmax = float(np.percentile(ref, 99.5))
dmax = max(float(np.abs(ref - sl[i][3]).max()) for i in pick) or 1.0
fig, ax = plt.subplots(len(pick), 5, figsize=(16, 3.6 * len(pick)), squeeze=False)
for r, i in enumerate(pick):
    a, b, est, gt = sl[i]
    ph = f'T{10 * (b - 1):02d}'
    panels = [(ref, f'Reference T{10 * (D.REF_BIN - 1):02d} (unwarped)\nrmse {base[i]:.4f}', 'gray', 0, vmax),
              (est, f'Warped (predicted, view {a})\nrmse {err[i]:.4f}', 'gray', 0, vmax),
              (gt, f'Ground truth {ph}', 'gray', 0, vmax),
              (ref - gt, 'Unwarped - truth', 'RdBu_r', -dmax, dmax),
              (est - gt, 'Warped - truth', 'RdBu_r', -dmax, dmax)]
    for c, (img, title, cmap, lo, hi) in enumerate(panels):
        im = ax[r, c].imshow(img[::-1], cmap=cmap, vmin=lo, vmax=hi, extent=ext)
        ax[r, c].set_title(title, fontsize=9)
        ax[r, c].set_xlabel('LR (mm)')
        if c == 0:
            ax[r, c].set_ylabel('SI (mm)')
    fig.colorbar(im, ax=ax[r, 3:].tolist(), fraction=0.023, label='mu (1/mm)')
    fig.colorbar(ax[r, 2].images[0], ax=ax[r, :3].tolist(), fraction=0.015, label='mu (1/mm)')
fig.suptitle(f'Case {cfg.case}: coronal slice, single-projection motion estimate', fontsize=11)
plt.show()
