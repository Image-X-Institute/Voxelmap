"""What self-consistency (L_SC) measures. On known fields, compare the published image-space L_SC (warp,
then warp back with the Chen inverse: two resamplings) with the composed form (one
resampling of I_ref at the composed coordinates) and with the inverse error itself
(|u(x + v(x)) + v(x)|, mm). Fields: constant SI translations; the case's hyperelastic T50
fields scaled by s; the L_SC baseline's T50 fields.

    python analysis/lsc_decompose.py --gpu 0 --hyper runs/hyper_b8.5 --baseline runs/lsc_b8.5

--hyper and --baseline are directories of train.py checkpoints for cases 8 and 1.
"""
import argparse
import os
import sys

P = argparse.ArgumentParser()
P.add_argument('--gpu', type=int, default=0)
P.add_argument('--hyper', required=True)
P.add_argument('--baseline', required=True)
A = P.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(A.gpu)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from utilities import data as D
from utilities.config import Config
from utilities.model import base_grid, build_model, invert_dvf, warp
from utilities.objective import Ops

dev = 'cuda'
for case in (8, 1):
    cfg = Config(case=case)
    zyx = tuple(s // cfg.bin for s in D.VOL_ZYX)
    g0 = base_grid(zyx, dev)
    iref = F.avg_pool3d(torch.from_numpy(D.load_volume(case, D.REF_BIN))[None, None].to(dev), 2, 2)
    model = build_model(cfg, zyx, dev)
    hm = torch.tensor(model.half_mm_zyx, device=dev).view(1, 3, 1, 1, 1)
    ops = {m: Ops(model, iref, g0, None, None, (1, 1), {'prj_min': 0, 'prj_max': 1},
                  dataclasses.replace(cfg, sc_mode=m)) for m in ('image', 'composed')}

    def row(name, d):
        with torch.no_grad():
            vol = warp(iref, d, g0)
            li = float(ops['image'].sc_term(d, vol).pow(2).mean())
            lc = float(ops['composed'].sc_term(d, vol).pow(2).mean())
            inv = invert_dvf(d, g0, cfg.sc_iters)
            p = g0 + torch.stack([inv[:, 2], inv[:, 1], inv[:, 0]], -1)
            e = F.grid_sample(d, p, mode='bilinear', padding_mode='border', align_corners=True) + inv
            err = float((e * hm).pow(2).sum(1).mean().sqrt())
        print(f'case {case}  {name:<28} L_SC image {li:.3e}  composed {lc:.3e}  '
              f'share not explained by the composed form {1 - lc / li:6.1%}  inverse err {err:.3f} mm', flush=True)

    for a in (0.5, 1, 2, 4):
        d = torch.zeros(1, 3, *zyx, device=dev)
        d[:, 1] = a / hm[0, 1, 0, 0, 0]
        row(f'translation {a} mm SI', d)
    for tag, arm in (('hyper 1e-1', A.hyper), ('L_SC baseline', A.baseline)):
        ck = torch.load(f'{arm}/case{case}_s0.pt', map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        v = [x for x in np.asarray(ck['views']) if D.phase_of(x) == 6][:4]
        with torch.no_grad():
            d0 = model(ck['w'].to(dev)[v])
        for s in ((0.25, 0.5, 1.0, 1.5) if tag.startswith('hyper') else (1.0,)):
            row(f'{tag} T50 x{s}', d0 * s)
