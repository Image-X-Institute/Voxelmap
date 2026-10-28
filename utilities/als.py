"""Stage 1: alternating least squares on the scores w and the basis C.

w-step  with C fixed, each view is 9 unknowns against ~87k detector rows:
        damped Gauss-Newton, solved independently per view, batched.
c-step  with w fixed, Adam on the B-spline control points over the solved views.

The factorisation is invariant under w -> a w, C -> C / a, so the basis is
renormalised to unit (1 mm RMS) norm around every c-step.
"""
import time

import numpy as np
import torch


def fd_jacobian(resid, w, idx, r0, eps):
    """Forward differences: one residual evaluation per score."""
    nb, nw = w.shape[0], w.shape[1] * w.shape[2]
    J = torch.empty(nb, r0.shape[1], nw, device=w.device)
    for j in range(nw):
        wp = w.clone().reshape(nb, nw)
        wp[:, j] += eps
        J[:, :, j] = (resid(wp.view_as(w), idx) - r0) / eps
    return J


def w_step(resid, w0, idx, iters, lm_init, eps):
    """Levenberg-Marquardt per view; a step is accepted per view, so one
    diverging view cannot discard the rest."""
    w = w0.clone()
    nb, nw = w.shape[0], w.shape[1] * w.shape[2]
    lam = torch.full((nb,), lm_init, device=w.device)
    for _ in range(iters):
        r = resid(w, idx)
        J = fd_jacobian(resid, w, idx, r, eps)
        JtJ = torch.einsum('bmi,bmj->bij', J, J)
        Jtr = torch.einsum('bmi,bm->bi', J, r)
        dmp = lam[:, None, None] * torch.diag_embed(torch.diagonal(JtJ, dim1=-2, dim2=-1) + 1e-12)
        wn = (w.reshape(nb, nw) + torch.linalg.solve(JtJ + dmp, -Jtr)).view_as(w)
        better = resid(wn, idx).norm(dim=1) < r.norm(dim=1)
        w = torch.where(better[:, None, None], wn, w)
        lam = torch.where(better, (lam * 0.5).clamp_min(1e-6), (lam * 4).clamp_max(1e3))
    return w


def c_step(model, w, sub, ops, cfg):
    """Adam on the control points; the loss is ||Ops.resid||^2 with L_SC on the
    full volume rather than the w-step's voxel subset."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.c_lr)
    for _ in range(cfg.c_steps):
        opt.zero_grad()
        for s in range(0, len(sub), cfg.batch):
            idx = sub[s:s + cfg.batch]
            d = ops.deform(w[idx])
            vol = ops.warp(ops.iref, d)
            loss = ops.prj_rows(ops.project(vol, idx), idx).pow(2).sum() / len(idx)
            loss = loss + cfg.sc_lambda * ops.sc_term(d, vol).pow(2).mean()
            (loss * len(idx) / len(sub)).backward()
        opt.step()


def renormalise(model, w):
    """Unit-norm basis fields, the scale moved into w: an exact reparametrisation."""
    with torch.no_grad():
        G = model.gram()
        for k in range(3):
            for i, l in enumerate(model.levels):
                s = G[k, i, i].clamp_min(1e-12).sqrt()
                l.control[k] /= s
                w[:, k, i] *= s


def run(model, w, views, ops, cfg, seed=0, log=print):
    """`cfg.outer` alternations from w = 0 and the polynomial basis.
    Returns w and the per-outer loss history."""
    b = cfg.batch
    t0 = time.time()
    history = []
    n_det = ops.vu[0] * ops.vu[1]
    for outer in range(1, cfg.outer + 1):
        ops.refresh_huber(w, views)

        tw = time.time()
        for s in range(0, len(views), b):
            idx = views[s:s + b]
            w[idx] = w_step(ops.resid, w[idx].clone(), idx, cfg.w_iters, cfg.lm_init, cfg.eps)
        tw = time.time() - tw

        renormalise(model, w)
        tc = time.time()
        # a fresh view order per outer, from a local generator
        sub = views[np.random.default_rng(seed * 1000 + outer).permutation(len(views))[:b * 4]]
        c_step(model, w, sub, ops, cfg)
        tc = time.time() - tc
        renormalise(model, w)

        r = torch.cat([ops.resid(w[views[s:s + b]], views[s:s + b]) for s in range(0, len(views), b)])
        rec = dict(outer=outer, resid=float(r.norm()), prj=float(r[:, :n_det].pow(2).sum()),
                   sc=float(r[:, n_det:].pow(2).sum()), rms_w=float(w.pow(2).mean().sqrt()),
                   t_w=tw, t_c=tc, mins=(time.time() - t0) / 60.0)
        history.append(rec)
        log(f"{outer:>5} {rec['resid']:>10.4f} {rec['rms_w']:>8.3f} "
            f"{tw:>7.1f}s {tc:>7.1f}s {rec['mins']:>7.1f}m")
    return w, history
