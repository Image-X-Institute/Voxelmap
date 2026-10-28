import argparse
import json
import os
import time
from dataclasses import fields
import numpy as np
from utilities.config import Config

P = argparse.ArgumentParser()
P.add_argument('--gpu', type=int, default=0)
P.add_argument('--out', default='runs')
for f in fields(Config):
    P.add_argument('--' + f.name.replace('_', '-'), type=type(f.default), default=f.default)
A = P.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(A.gpu)

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from utilities import als, data as D, synthesis                 # noqa: E402
from utilities.model import BSplineModel, MotionEnc, base_grid, polynomial_init  # noqa: E402
from utilities.objective import Ops                             # noqa: E402
from utilities.projector import Projector                       # noqa: E402

cfg = Config(**{f.name: getattr(A, f.name) for f in fields(Config)})
dev = 'cuda'
t0 = time.time()
D.set_seed(cfg.seed)

# ---- the two scans -------------------------------------------------------
# reference: T00 stands in for the DIBH CBCT. ventilated arc: simulated from
# the ten phase volumes at full resolution, then solved at `bin` mm.
stats = D.projection_stats(cfg.case)
geo = D.geometry()
zyx = tuple(s // cfg.bin for s in D.VOL_ZYX)
pr_full = Projector(geo, D.VOL_ZYX, D.VOX_MM, D.DET_VU, D.DET_SPACING)
pr = Projector(geo, zyx, D.VOX_MM * cfg.bin, D.DET_VU, D.DET_SPACING)
y = D.simulate_arc(cfg.case, pr_full, stats, dev, cfg.n_breaths, cfg.noise_i0, cfg.seed).to(dev)
iref = F.avg_pool3d(torch.from_numpy(D.load_volume(cfg.case, D.REF_BIN))[None, None].to(dev),
                    cfg.bin, cfg.bin)
views = np.arange(0, D.N_ANGLES, cfg.train_stride)

model = BSplineModel(zyx, D.HALF_MM_XYZ[::-1], cfg.base_grid, cfg.levels).to(dev)
polynomial_init(model)
enc = MotionEnc(D.DET_VU, max_ch=cfg.enc_max_ch, n_comp=cfg.levels).to(dev)
ops = Ops(model, iref, base_grid(zyx, dev), y, pr, D.DET_VU, stats, cfg)
w = torch.zeros(D.N_ANGLES, 3, cfg.levels, device=dev)
print(f'case {cfg.case}: {D.N_ANGLES} views, {len(views)} solved; '
      f'setup {time.time() - t0:.0f} s', flush=True)

# ---- stage 1 -------------------------------------------------------------
print(f"{'outer':>5} {'|r|':>10} {'rms w':>8} {'w-step':>8} {'c-step':>8} {'total':>8}")
t1 = time.time()
w, history = als.run(model, w, views, ops, cfg, seed=cfg.seed,
                     log=lambda m: print(m, flush=True))
t1 = time.time() - t1

# ---- stage 2 -------------------------------------------------------------
t2 = time.time()
synthesis.fit_encoder(enc, w, views, ops, cfg.synth_pairs, cfg.synth_epochs, cfg.batch,
                      cfg.enc_lr, seed=cfg.seed, log=lambda m: print(m, flush=True))
t2 = time.time() - t2
print(f'stage 1 {t1 / 60:.2f} min, stage 2 {t2 / 60:.2f} min, '
      f'total {(time.time() - t0) / 60:.2f} min', flush=True)

os.makedirs(A.out, exist_ok=True)
stem = os.path.join(A.out, f'case{cfg.case}_s{cfg.seed}')
torch.save(dict(model=model.state_dict(), enc=enc.state_dict(), w=w.cpu(),
                views=views, config=cfg.as_dict()), stem + '.pt')
np.savez(stem + '_loss.npz', config=json.dumps(cfg.as_dict()), t_stage1=t1, t_stage2=t2,
         **{k: np.array([h[k] for h in history]) for k in history[0]})
print(f'wrote {stem}.pt and {stem}_loss.npz')
