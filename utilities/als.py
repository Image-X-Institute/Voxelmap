"""Stage 1: alternating least squares on the scores w and the basis C.

w-step  with C fixed, each view is 9 unknowns against ~87k detector rows:
        damped Gauss-Newton, solved independently per view, batched.
c-step  with w fixed, Adam on the B-spline control points over the solved views.

The factorisation is invariant under w -> a w, C -> C / a, so the basis is
renormalised to unit (1 mm RMS) norm around every c-step.
"""
import copy
import time

import numpy as np
import torch

from utilities.objective import hyper_penalty
from utilities.regularisers import penalty as reg_penalty


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


def c_loss(w, idx, ops, cfg):
    """The c-step objective on one micro-batch, per view."""
    d = ops.deform(w[idx])
    vol = ops.warp(ops.iref, d)
    loss = ops.prj_rows(ops.project(vol, idx), idx).pow(2).sum() / len(idx)
    if cfg.sc_lambda:
        loss = loss + cfg.sc_lambda * ops.sc_term(d, vol).pow(2).mean()
    if cfg.hyper_lambda:
        loss = loss + cfg.hyper_lambda * hyper_penalty(d, vol, cfg).mean()
    if cfg.reg:
        loss = loss + cfg.reg_lambda * reg_penalty(d, cfg.reg).mean()
    return loss


def c_step(model, w, sub, ops, cfg):
    """Adam on the control points; the loss is ||Ops.resid||^2 with L_SC on the
    full volume rather than the w-step's voxel subset.

    c_accept re-evaluates the objective after each step and
    undoes a step that raised it (Adam state restored, rate halved; doubled back
    towards c_lr on acceptance), which makes the c-step a descent method.
    Returns the number of undone steps."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.c_lr)
    n_rej = 0
    for step in range(cfg.c_steps):
        opt.zero_grad()
        acc = 0.0
        mb = cfg.c_micro or cfg.batch
        for s in range(0, len(sub), mb):
            idx = sub[s:s + mb]
            loss = c_loss(w, idx, ops, cfg) * len(idx) / len(sub)
            loss.backward()
            acc += float(loss.detach()) if cfg.c_accept else 0.0
        if cfg.c_accept:
            keep = [p.detach().clone() for p in model.parameters()]
            state = copy.deepcopy(opt.state_dict())
        opt.step()
        if cfg.c_accept:
            with torch.no_grad():
                new = sum(float(c_loss(w, sub[s:s + mb], ops, cfg)) * len(sub[s:s + mb]) / len(sub)
                          for s in range(0, len(sub), mb))
            if new > acc:
                with torch.no_grad():
                    for p, k in zip(model.parameters(), keep):
                        p.copy_(k)
                opt.load_state_dict(state)
                n_rej += 1
                lr = opt.param_groups[0]['lr'] * 0.5
            else:
                lr = min(opt.param_groups[0]['lr'] * 2.0, cfg.c_lr)
            for g in opt.param_groups:
                g['lr'] = lr
    return n_rej


def ctrl_bound(model, w, views, mult):
    """Choi & Lee: a uniform cubic B-spline FFD is injective if every control displacement
    is below 0.40 x knot spacing. Per (level, axis), scale the lattice so the largest solved
    score times the level's peak displacement meets mult x that bound; the scores are NOT
    rescaled, so this is a hard cap on per-level deformation ."""
    with torch.no_grad():
        half = model.half_mm_zyx
        n_lv = len(model.levels)
        for k in range(3):
            for i, lv in enumerate(model.levels):
                bd = mult * 0.40 * 2.0 * half[k] / max(lv.control.shape[-1] - 3, 1)
                wv = torch.zeros(1, 3, n_lv, device=w.device)
                wv[0, k, i] = 1.0
                dmax = float(model(wv)[0, k].abs().max()) * half[k]
                cur = float(w[views, k, i].abs().max()) * dmax
                if cur > bd > 0:
                    for l in model.lattices(i):
                        l.control[k] *= bd / cur


def renormalise(model, w):
    """Unit-norm basis fields, the scale moved into w: an exact reparametrisation."""
    with torch.no_grad():
        G = model.gram()
        for k in range(3):
            for i in range(len(model.levels)):
                s = G[k, i, i].clamp_min(1e-12).sqrt()
                for l in model.lattices(i):
                    l.control[k] /= s
                w[:, k, i] *= s


def run(model, w, views, ops, cfg, seed=0, log=print):
    """`cfg.outer` alternations from w = 0 and the polynomial basis.
    Returns w and the per-outer loss history."""
    b = cfg.batch
    t0 = time.time()
    history = []
    n_det = ops.vu[0] * ops.vu[1]
    last = 0.0
    for outer in range(1, cfg.outer + 1):
        # a fixed wall-clock budget, not a stopping rule
        if cfg.stage1_minutes and (time.time() - t0 + last) / 60.0 > cfg.stage1_minutes:
            break
        to = time.time()
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
        n_rej = c_step(model, w, sub, ops, cfg)
        tc = time.time() - tc
        renormalise(model, w)
        if cfg.ctrl_bound:
            ctrl_bound(model, w, views, cfg.ctrl_bound)

        r = torch.cat([ops.resid(w[views[s:s + b]], views[s:s + b]) for s in range(0, len(views), b)])
        n_sc = n_det + (ops.sc_index.numel() if cfg.sc_lambda else 0)
        rec = dict(outer=outer, resid=float(r.norm()), prj=float(r[:, :n_det].pow(2).sum()),
                   sc=float(r[:, n_det:n_sc].pow(2).sum()), reg=float(r[:, n_sc:].pow(2).sum()),
                   rms_w=float(w.pow(2).mean().sqrt()),
                   t_w=tw, t_c=tc, n_rej=n_rej, mins=(time.time() - t0) / 60.0)
        history.append(rec)
        log(f"{outer:>5} {rec['resid']:>10.4f} {rec['rms_w']:>8.3f} "
            f"{tw:>7.1f}s {tc:>7.1f}s {rec['mins']:>7.1f}m" + (f"  rej {n_rej}" if cfg.c_accept else ''))
        last = time.time() - to
    return w, history
