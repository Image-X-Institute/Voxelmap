"""Evaluation of a trained checkpoint on the arc: landmark TRE and image similarity.

    python test.py --ckpt runs/hyper_b8.5/case2_s0.pt --gpu 0

TRE: the 75-point sets, phases T10-T50, landmarks propagated through the inverse field.
Image: the predicted volume (I_ref warped by the predicted field) against the ground-truth
phase volume on the 2 mm grid, all nine moving phases, within a body mask from the acquired
reference (tissue threshold, holes filled per axial slice so the lungs count as body), and
over the whole box. PSNR uses the case's volume maximum; SSIM is 3-D with a Gaussian window,
sigma 1.5 voxels, C1 = (0.01 L)^2, C2 = (0.03 L)^2. 'identity' is the unwarped reference.
Everything is the mean over phases of the per-phase mean over views; per-phase values are
kept for paired tests. Writes <ckpt>_eval.json.
"""
import argparse
import json
import os

P = argparse.ArgumentParser()
P.add_argument('--ckpt', required=True)
P.add_argument('--gpu', type=int, default=0)
A = P.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(A.gpu)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes

from utilities import data as D
from utilities.config import Config
from utilities.landmarks import landmarks_by_bin, sample_dvf, tre_mm
from utilities.model import (MotionEnc, base_grid, build_model, displacement, inference_mode,
                             inverse_displacement, warp)
from utilities.projector import Projector

dev = 'cuda'
CHUNK = 16
ck = torch.load(A.ckpt, map_location=dev, weights_only=False)
cfg = Config(**ck['config'])
zyx = tuple(s // cfg.bin for s in D.VOL_ZYX)
grid0 = base_grid(zyx, dev)
model = build_model(cfg, zyx, dev)
model.load_state_dict(ck['model'], strict=False)
enc = MotionEnc(D.DET_VU, max_ch=cfg.enc_max_ch, n_comp=cfg.levels).to(dev)
enc.load_state_dict(ck['enc'])

stats = D.projection_stats(cfg.case)
pr = Projector(D.geometry(), D.VOL_ZYX, D.VOX_MM, D.DET_VU, D.DET_SPACING)
y = D.simulate_arc(cfg.case, pr, stats, dev, cfg.n_breaths, cfg.noise_i0, cfg.seed).to(dev)
vols = {b: torch.from_numpy(D.load_volume(cfg.case, b))[None, None] for b in range(1, D.N_BINS + 1)}
L = max(float(v.max()) for v in vols.values())
GT = {b: F.avg_pool3d(v.to(dev), cfg.bin, cfg.bin) for b, v in vols.items()}
iref = GT[D.REF_BIN]

lm = landmarks_by_bin(cfg.case)
lm0 = lm[D.REF_BIN]
bin_of = np.array([D.phase_of(i, D.N_ANGLES, cfg.n_breaths) for i in range(D.N_ANGLES)])
heldout = np.array([i for i in range(D.N_ANGLES) if i % 3])
moving = [b for b in range(1, D.N_BINS + 1) if b != D.REF_BIN]

# body mask from the acquired reference; zyx is (AP, SI, LR), so an axial slice fixes SI
body = iref[0, 0].cpu().numpy() > 0.4 * 0.0195
for s in range(body.shape[1]):
    body[:, s, :] = binary_fill_holes(body[:, s, :])
MASK = torch.from_numpy(body).to(dev)

g = torch.exp(-0.5 * (torch.arange(11, device=dev) - 5.0) ** 2 / 1.5 ** 2)
g = g / g.sum()


def blur(x):
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = 11
        x = F.conv3d(x, g.view(shape), padding=[5 if d == dim else 0 for d in range(3)])
    return x


def image_metrics(pred, ref):
    """(B,1,Z,Y,X) pairs -> per-volume dict of PSNR and SSIM, body and full box."""
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_p, mu_r = blur(pred), blur(ref)
    s_pp, s_rr = blur(pred * pred) - mu_p ** 2, blur(ref * ref) - mu_r ** 2
    s_pr = blur(pred * ref) - mu_p * mu_r
    smap = (2 * mu_p * mu_r + C1) * (2 * s_pr + C2) / ((mu_p ** 2 + mu_r ** 2 + C1) * (s_pp + s_rr + C2))
    out = []
    for i in range(pred.shape[0]):
        e2 = (pred[i, 0] - ref[i, 0]) ** 2
        row = {}
        for name, m in (('body', MASK), ('full', torch.ones_like(MASK))):
            row[f'psnr_{name}'] = float(10 * np.log10(L * L / float(e2[m].mean())))
            row[f'ssim_{name}'] = float(smap[i, 0][m].mean())
        out.append(row)
    return out


def score(proj, b):
    """Encoder prediction for projections of phase b -> (per-view TRE or None, image rows)."""
    w = enc(proj)
    vol = warp(iref, displacement(model, w, cfg, grid0), grid0)
    rows = image_metrics(vol, GT[b].expand(len(w), -1, -1, -1, -1))
    t = None
    if b in lm:
        q = sample_dvf(inverse_displacement(model, w, cfg, grid0), lm0).cpu().numpy()
        t = [float(tre_mm(lm0 + d, lm[b]).mean()) for d in q]
    return t, rows


acc = {'arc': {'tre': {}, 'img': {}}}
with torch.no_grad(), inference_mode(enc):
    for b in moving:
        sel = heldout[bin_of[heldout] == b]
        for s in range(0, len(sel), CHUNK):
            t, rows = score(y[sel[s:s + CHUNK]], b)
            if t:
                acc['arc']['tre'].setdefault(b, []).extend(t)
            acc['arc']['img'].setdefault(b, []).extend(rows)
    ident = {b: image_metrics(iref, GT[b])[0] for b in moving}

KEYS = ('psnr_body', 'ssim_body', 'psnr_full', 'ssim_full')
res = dict(ckpt=A.ckpt)
for r, a in acc.items():
    per = {int(b): float(np.mean(v)) for b, v in sorted(a['tre'].items())}
    res[r] = dict(tre=float(np.mean(list(per.values()))), tre_phase=per)
    for k in KEYS:
        per = {int(b): float(np.mean([m[k] for m in v])) for b, v in sorted(a['img'].items())}
        res[r][k] = float(np.mean(list(per.values())))
        res[r][f'{k}_phase'] = per
res['identity'] = {k: float(np.mean([ident[b][k] for b in moving])) for k in KEYS}
res['identity'].update({f'{k}_phase': {int(b): ident[b][k] for b in moving} for k in KEYS})
res['identity']['tre'] = float(np.mean([tre_mm(lm0, lm[b]).mean() for b in lm if b != D.REF_BIN]))
print(f"case {cfg.case}: arc TRE {res['arc']['tre']:.3f}  "
      f"| SSIM {res['arc']['ssim_body']:.4f}  PSNR {res['arc']['psnr_body']:.2f} dB  "
      f"(identity {res['identity']['ssim_body']:.4f}, {res['identity']['psnr_body']:.2f})")
out = os.path.splitext(A.ckpt)[0] + '_eval.json'
with open(out, 'w') as f:
    json.dump(res, f, indent=1)
print(f'wrote {out}')
