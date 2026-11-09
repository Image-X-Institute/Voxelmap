"""The objective both stage-1 steps minimise, bound to the forward model.

    L(w, C) = sum_p rho(P_p[I_ref o phi_p] - y_p) + lambda_SC ||I_ref o phi_p o phi_p^-1 - I_ref||^2

With hyper_lambda > 0 the objective gains
lambda_H mean_x psi(det J), the volume term of hyperelastic regularisation.

rho is pseudo-Huber with delta tracking the residual rms. The residual rows
returned by Ops.resid have ||r||^2 = L exactly, so the Gauss-Newton w-step and
the c-step descend the same function.
"""
import numpy as np
import torch
import torch.nn.functional as F

from utilities.data import MU_WATER
from utilities.model import displacement, invert_dvf, warp
from utilities.regularisers import penalty as reg_penalty
from utilities.projector import project


def prj_rows(drr, y, prj_scale, huber_delta=None):
    """Rows whose squared sum is the data term. Pseudo-Huber as a reweighted
    square, sum(row^2) = sum r^2 / sqrt(1 + (r/delta)^2), with the weight
    detached (IRLS)."""
    r = (drr - y).reshape(drr.shape[0], -1)
    r = r * (prj_scale / r.shape[1]) ** 0.5
    if huber_delta is not None:
        r = r * (1.0 + (r.detach() / huber_delta).pow(2)).pow(-0.25)
    return r


def hyper_psi(v, v0=0.1):
    """Volume term of hyperelastic regularisation (Burger, Modersitzki & Ruthotto,
    SIAM J Sci Comput 2013): psi(v) = (v - 1)^4 / v^2 on det J = v. Blind to
    shear (det J = 1), flat near 1, and -> inf as v -> 0. Below v0 it continues
    as its second-order Taylor expansion, so folded fields (v <= 0) get a large
    finite, convex penalty instead of inf/NaN."""
    vc = v.clamp_min(v0)
    psi = (vc - 1) ** 4 / vc ** 2
    a = v0 - 1
    d1 = 4 * a ** 3 / v0 ** 2 - 2 * a ** 4 / v0 ** 3
    d2 = 12 * a ** 2 / v0 ** 2 - 16 * a ** 3 / v0 ** 3 + 6 * a ** 4 / v0 ** 4
    lo = (v0 - 1) ** 4 / v0 ** 2 + d1 * (v - v0) + 0.5 * d2 * (v - v0) ** 2
    return torch.where(v < v0, lo, psi)


def det_jacobian(d, stride=2):
    """det(I + grad u) of a DVF d (B, 3, Z, Y, X) in grid_sample's normalised units
    (dz, dy, dx), align_corners=True, by central differences on every `stride`-th
    voxel. The field is a cubic B-spline whose finest knots are ~6 voxels apart,
    so the subsampled Jacobian is what the full grid would give, at 1/stride^3 the
    cost. Explicit cofactor expansion: batched linalg.det on millions of 3x3s is
    ~10x slower."""
    n = d.shape[-3:]
    sc = torch.tensor([(m - 1) / 2.0 for m in n], device=d.device, dtype=d.dtype)
    u = d[..., ::stride, ::stride, ::stride] * sc.view(1, 3, 1, 1, 1)
    g = [torch.gradient(u, dim=k, spacing=float(stride))[0] for k in (2, 3, 4)]
    a = [[g[j][:, i] + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def hyper_penalty(d, vol, cfg):
    """Per-voxel volume penalty psi(det J) on det_jacobian's grid."""
    return hyper_psi(det_jacobian(d))


class Ops:
    """Everything the method needs from the forward model."""

    def __init__(self, model, iref, grid0, y, projector, vu, stats, cfg):
        self.model, self.iref, self.grid0 = model, iref, grid0
        self.y, self.pr, self.vu, self.stats = y, projector, tuple(vu), stats
        self.cfg = cfg
        self.prj_scale = float(stats['prj_max'] - stats['prj_min']) ** 2
        self.huber_delta = None
        # L_SC in the w-step on a FIXED seeded voxel subset, so the residual is
        # a deterministic function of w across Gauss-Newton iterations
        nvox = int(np.prod(iref.shape[-3:]))
        g = torch.Generator(device='cpu').manual_seed(0)
        self.sc_index = torch.randperm(nvox, generator=g)[:min(cfg.sc_vox, nvox)].to(iref.device)

    def deform(self, w):
        return displacement(self.model, w, self.cfg, self.grid0)

    def warp(self, vol, d):
        return warp(vol, d, self.grid0)

    def project(self, vol, idx):
        return project(vol, idx, self.pr, self.vu, self.stats)

    def prj_rows(self, drr, idx):
        return prj_rows(drr, self.y[idx], self.prj_scale, self.huber_delta)

    def sc_term(self, d, vol):
        """Spatial consistency: the warped image carried back by the inverse field."""
        inv = invert_dvf(d, self.grid0, self.cfg.sc_iters)
        if self.cfg.sc_mode == 'composed':
            # x -> p = x + inv(x) -> p + d(p): one sample of I_ref, no double resampling
            zyx = lambda f: torch.stack([f[:, 2], f[:, 1], f[:, 0]], dim=-1)
            p = self.grid0 + zyx(inv)
            dp = F.grid_sample(d, p, mode='bilinear', padding_mode='border', align_corners=True)
            q = p + zyx(dp)
            return F.grid_sample(self.iref.expand(d.shape[0], -1, -1, -1, -1), q, mode='bilinear',
                                 padding_mode='border', align_corners=True) - self.iref
        return self.warp(vol, inv) - self.iref

    def resid(self, w, idx):
        """(B, rows): data rows then L_SC rows, ||r||^2 = the objective."""
        with torch.no_grad():
            d = self.deform(w)
            vol = self.warp(self.iref, d)
            rows = [self.prj_rows(self.project(vol, idx), idx)]
            if self.cfg.sc_lambda:  # no L_SC rows: nothing to invert
                sc = self.sc_term(d, vol).reshape(len(idx), -1)[:, self.sc_index]
                rows.append(sc * (self.cfg.sc_lambda / self.sc_index.numel()) ** 0.5)
            if self.cfg.hyper_lambda:
                # every voxel of det_jacobian's grid, so the w-step and the c-step
                # descend the identical term; sqrt(psi) = (v - 1)^2 / v above v0
                h = hyper_penalty(d, vol, self.cfg).reshape(len(idx), -1)
                rows.append(h.sqrt() * (self.cfg.hyper_lambda / h.shape[1]) ** 0.5)
            if self.cfg.reg:
                g = reg_penalty(d, self.cfg.reg).reshape(len(idx), -1)
                rows.append(g.sqrt() * (self.cfg.reg_lambda / g.shape[1]) ** 0.5)
            return torch.cat(rows, dim=1)

    def refresh_huber(self, w, views):
        """delta = rms of the unweighted data rows at the current fit."""
        b = self.cfg.batch
        with torch.no_grad():
            self.huber_delta = None
            r = torch.cat([
                self.prj_rows(self.project(self.warp(self.iref, self.deform(w[views[s:s + b]])),
                                           views[s:s + b]), views[s:s + b]).flatten()
                for s in range(0, len(views), b)])
            self.huber_delta = max(float(r.pow(2).mean().sqrt()), 1e-12)
