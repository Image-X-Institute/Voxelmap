"""Folding of ENCODER-predicted fields at inference (the deployment path), for every
held-out view (i % 3 != 0) of each checkpoint. det J by central differences on the FULL grid
(stride 1), so folds between the stride-2 points the regularizer sees are counted. Regions
by reference intensity only: body (I > 0.01), soft tissue (0.9-1.2 mu_w), parenchyma
(0.05-0.5 mu_w).

    python analysis/encoder_folding.py --gpu 0 --out runs/folding.json runs/hyper_b8.5/case*_s0.pt runs/lsc_b8.5/case*_s0.pt
"""
import argparse
import json
import os
import sys

P = argparse.ArgumentParser()
P.add_argument('--gpu', type=int, default=0)
P.add_argument('--out', required=True)
P.add_argument('ckpts', nargs='+')
A = P.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(A.gpu)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from utilities import data as D
from utilities.config import Config
from utilities.model import MotionEnc, base_grid, build_model, displacement, inference_mode
from utilities.objective import det_jacobian
from utilities.projector import Projector

dev = 'cuda'
ycache, out = {}, {}
for path in A.ckpts:
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = Config(**ck['config'])
    zyx = tuple(s // cfg.bin for s in D.VOL_ZYX)
    g0 = base_grid(zyx, dev)
    if cfg.case not in ycache:
        stats = D.projection_stats(cfg.case)
        pr = Projector(D.geometry(), D.VOL_ZYX, D.VOX_MM, D.DET_VU, D.DET_SPACING)
        y = D.simulate_arc(cfg.case, pr, stats, dev, cfg.n_breaths, cfg.noise_i0, cfg.seed).to(dev)
        iref = F.avg_pool3d(torch.from_numpy(D.load_volume(cfg.case, D.REF_BIN))[None, None].to(dev),
                            cfg.bin, cfg.bin)[0, 0]
        mu = D.MU_WATER
        regions = dict(box=torch.ones_like(iref, dtype=torch.bool), body=iref > D.MASK_THRESH,
                       soft=(iref > 0.9 * mu) & (iref < 1.2 * mu), lung=(iref > 0.05 * mu) & (iref < 0.5 * mu))
        ycache = {cfg.case: (y, regions)}
    y, regions = ycache[cfg.case]
    model = build_model(cfg, zyx, dev)
    model.load_state_dict(ck['model'], strict=False)
    enc = MotionEnc(D.DET_VU, max_ch=cfg.enc_max_ch, n_comp=cfg.levels).to(dev)
    enc.load_state_dict(ck['enc'])
    views = [i for i in range(D.N_ANGLES) if i % 3]
    fold = {k: [] for k in regions}
    pct = {k: [] for k in ('soft', 'lung')}
    with torch.no_grad(), inference_mode(enc):
        for s in range(0, len(views), 8):
            J = det_jacobian(displacement(model, enc(y[views[s:s + 8]]), cfg, g0), stride=1)
            for k, m in regions.items():
                fold[k].append((J[:, m] <= 0).float().mean(1).cpu())
            for k in pct:
                pct[k].append(torch.quantile(J[:, regions[k]][:, ::97].float(),
                                             torch.tensor([0.01, 0.99], device=dev), dim=1).T.cpu())
    rec = {f'fold_{k}': float(torch.cat(v).mean()) for k, v in fold.items()}
    rec.update({f'fold_{k}_max_view': float(torch.cat(v).max()) for k, v in fold.items()})
    rec.update({f'views_with_fold_{k}': float((torch.cat(v) > 0).float().mean()) for k, v in fold.items()})
    for k, v in pct.items():
        q = torch.cat(v).numpy()
        rec[f'J_{k}_p1'], rec[f'J_{k}_p99'] = float(np.median(q[:, 0])), float(np.median(q[:, 1]))
    out[path] = rec
    print(f"{path:<40} fold box {rec['fold_box']:.4%} body {rec['fold_body']:.4%} lung {rec['fold_lung']:.4%} "
          f"| views with any fold {rec['views_with_fold_box']:.0%} | J soft {rec['J_soft_p1']:.2f}/{rec['J_soft_p99']:.2f} "
          f"lung {rec['J_lung_p1']:.2f}/{rec['J_lung_p99']:.2f}", flush=True)
    json.dump(out, open(A.out, 'w'), indent=1)
