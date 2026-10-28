"""The objective both stage-1 steps minimise, bound to the forward model.

    L(w, C) = sum_p rho(P_p[I_ref o phi_p] - y_p) + lambda_SC ||I_ref o phi_p o phi_p^-1 - I_ref||^2

rho is pseudo-Huber with delta tracking the residual rms. The residual rows
returned by Ops.resid have ||r||^2 = L exactly, so the Gauss-Newton w-step and
the c-step descend the same function.
"""
import numpy as np
import torch
from utilities.model import invert_dvf, warp
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
        return self.model(w)

    def warp(self, vol, d):
        return warp(vol, d, self.grid0)

    def project(self, vol, idx):
        return project(vol, idx, self.pr, self.vu, self.stats)

    def prj_rows(self, drr, idx):
        return prj_rows(drr, self.y[idx], self.prj_scale, self.huber_delta)

    def sc_term(self, d, vol):
        """Spatial consistency: the warped image carried back by the inverse field."""
        inv = invert_dvf(d, self.grid0, self.cfg.sc_iters)
        return self.warp(vol, inv) - self.iref

    def resid(self, w, idx):
        """(B, rows): data rows then L_SC rows, ||r||^2 = the objective."""
        with torch.no_grad():
            d = self.deform(w)
            vol = self.warp(self.iref, d)
            r = self.prj_rows(self.project(vol, idx), idx)
            sc = self.sc_term(d, vol).reshape(len(idx), -1)[:, self.sc_index]
            sc = sc * (self.cfg.sc_lambda / self.sc_index.numel()) ** 0.5
            return torch.cat([r, sc], dim=1)

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
