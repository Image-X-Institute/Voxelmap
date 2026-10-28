"""The motion model and the encoder.

DVF(p) = sum_l w_l(p) * e_l : one B-spline basis field per resolution level and
Cartesian axis, so a motion state is nine scores w. The encoder maps one
projection to those nine scores.
"""
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _basis_matrix(n_out, grid):
    """(n_out, grid + 3) cubic B-spline basis over normalised coordinates."""
    u = np.linspace(0.0, 1.0, n_out, dtype=np.float64) * (grid - 1)
    idx = np.clip(np.floor(u), 0, grid - 2).astype(np.int64)
    t = (u - idx)[:, None]
    w = np.concatenate([(1 - 3 * t + 3 * t ** 2 - t ** 3) / 6,
                        (4 - 6 * t ** 2 + 3 * t ** 3) / 6,
                        (1 + 3 * t + 3 * t ** 2 - 3 * t ** 3) / 6,
                        t ** 3 / 6], axis=1)
    B = np.zeros((n_out, grid + 3), dtype=np.float64)
    np.add.at(B, (np.repeat(np.arange(n_out), 4), (idx[:, None] + np.arange(4)).ravel()),
              w.ravel())
    return torch.from_numpy(B.astype(np.float32))


class BSplineLevel(nn.Module):
    """One level: a control lattice per Cartesian axis."""

    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        n = grid + 3
        self.control = nn.Parameter(torch.randn(3, 1, n, n, n) * 1e-2)   # (axis, 1, cz, cy, cx)

    def set_resolution(self, vol_zyx):
        dev = self.control.device
        for nm, s in zip('zyx', vol_zyx):
            self.register_buffer('B' + nm, _basis_matrix(s, self.grid).to(dev), persistent=False)

    def field(self, w):
        """w: (B, 3, 1) -> DVF (B, 3, Z, Y, X)."""
        c = torch.einsum('bki,kipqr->bkpqr', w, self.control)
        c = torch.einsum('Pp,bkpqr->bkPqr', self.Bz, c)
        c = torch.einsum('Qq,bkPqr->bkPQr', self.By, c)
        return torch.einsum('Rr,bkPQr->bkPQR', self.Bx, c)


class BSplineModel(nn.Module):
    """Three levels at lattice spacing base_grid x (1, 2, 4).

    The DVF is in grid_sample's normalised [-1, 1] units, channels (dz, dy, dx);
    `half_mm_zyx` converts to millimetres so that gram() is physical.
    """

    def __init__(self, vol_zyx, half_mm_zyx, base_grid=6, n_levels=3):
        super().__init__()
        self.levels = nn.ModuleList([BSplineLevel(base_grid * 2 ** l) for l in range(n_levels)])
        self.half_mm_zyx = tuple(float(v) for v in half_mm_zyx)
        self.set_resolution(vol_zyx)

    def set_resolution(self, vol_zyx):
        self.vol_zyx = tuple(int(s) for s in vol_zyx)
        self.n_voxels = int(np.prod(self.vol_zyx))
        for lvl in self.levels:
            lvl.set_resolution(self.vol_zyx)

    def forward(self, w):
        """w: (B, 3, n_levels) -> DVF (B, 3, Z, Y, X)."""
        d = self.levels[0].field(w[:, :, 0:1])
        for i, lvl in enumerate(self.levels[1:], start=1):
            d = d + lvl.field(w[:, :, i:i + 1])
        return d

    def gram(self):
        """(3, n, n) inner products of the basis fields per axis, in mm^2 averaged
        over voxels: G[k, i, i] = 1 means one unit of w_i moves the anatomy 1 mm
        RMS along axis k."""
        n = len(self.levels)
        G = torch.zeros(3, n, n, device=self.levels[0].control.device)
        for i, la in enumerate(self.levels):
            for j, lb in enumerate(self.levels):
                t = torch.einsum('pP,kipqr->kiPqr', la.Bz.T @ lb.Bz, la.control)
                t = torch.einsum('qQ,kiPqr->kiPQr', la.By.T @ lb.By, t)
                t = torch.einsum('rR,kiPQr->kiPQR', la.Bx.T @ lb.Bx, t)
                G[:, i, j] = torch.einsum('kipqr,kjpqr->kij', t, lb.control)[:, 0, 0]
        s = torch.tensor(self.half_mm_zyx, device=G.device).view(3, 1, 1) ** 2
        return G * s / self.n_voxels


def polynomial_init(model):
    """Cold-start basis: constant / linear / quadratic in SI, one per level,
    each normalised to 1 mm RMS. Respiratory motion is dominated by bulk SI
    translation with smooth higher-order structure."""
    with torch.no_grad():
        for lev, l in enumerate(model.levels):
            g = torch.linspace(-1, 1, l.control.shape[-1], device=l.control.device)
            _, gy, _ = torch.meshgrid(g, g, g, indexing='ij')
            field = {0: torch.ones_like(gy), 1: gy}.get(lev, gy ** lev - (gy ** lev).mean())
            l.control.copy_(field[None, None].expand_as(l.control).clone())
        G = model.gram()
        for k in range(3):
            for i, l in enumerate(model.levels):
                l.control[k] /= G[k, i, i].clamp_min(1e-12).sqrt()


class MotionEnc(nn.Module):
    """Six stride-2 3x3 conv / BN / ReLU blocks, then a linear layer to 3 x n_comp."""

    def __init__(self, in_size, in_ch=1, n_layers=6, max_ch=32, n_comp=3):
        super().__init__()
        self.n_comp = n_comp
        chans = [in_ch] + [min(2 ** (i + 1), max_ch) for i in range(n_layers)]
        layers = []
        for i in range(n_layers):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        h, w = (-(-s // 2 ** n_layers) for s in in_size)
        self.fc = nn.Linear(chans[-1] * h * w, 3 * n_comp)

    def forward(self, y):
        return self.fc(self.features(y).flatten(1)).view(-1, 3, self.n_comp)


@contextlib.contextmanager
def inference_mode(net):
    """BatchNorm on its running statistics, then restore. Deployment is batch
    size one, where train-mode BN normalises by the sample's own statistics."""
    was = net.training
    net.eval()
    try:
        yield net
    finally:
        net.train(was)


def base_grid(vol_zyx, device):
    """(1, Z, Y, X, 3) identity sampling grid in grid_sample's (x, y, z) order."""
    g = [torch.linspace(-1, 1, s, device=device) for s in vol_zyx]
    mz, my, mx = torch.meshgrid(*g, indexing='ij')
    return torch.stack([mx, my, mz], dim=-1)[None]


def warp(vol, dvf, grid0):
    """vol (1, 1, Z, Y, X) sampled at x + dvf; dvf (B, 3, Z, Y, X) as (dz, dy, dx)."""
    d = torch.stack([dvf[:, 2], dvf[:, 1], dvf[:, 0]], dim=-1)
    return F.grid_sample(vol.expand(d.shape[0], -1, -1, -1, -1), grid0 + d,
                         mode='bilinear', padding_mode='border', align_corners=True)


def invert_dvf(d, grid0, iters=3):
    """Inverse field by the fixed point of Chen et al (2008)."""
    inv = torch.zeros_like(d)
    for _ in range(iters):
        g = grid0 + torch.stack([inv[:, 2], inv[:, 1], inv[:, 0]], dim=-1)
        inv = -F.grid_sample(d, g, mode='bilinear', padding_mode='border', align_corners=True)
    return inv


def predict(enc, model, y):
    """Deployment: one projection (1, 1, V, U), normalised -> DVF (1, 3, Z, Y, X)."""
    with torch.no_grad(), inference_mode(enc):
        return model(enc(y))
