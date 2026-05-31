"""
Loss terms for VoxelMap motion-guided refinement.

Supervised  : MSE(z_true, z_pred) over a thorax mask (matches published embodiment).
Unsupervised: MSE(target_vol, updated_vol) + alpha * grad(dvf).
Cycle       : route-A and route-B must produce the same DVF (proj-cycle's earned
              convergence pressure on the latent).
Consistency : the residual must not encode displacement the DVF could have explained
              (decouples cleanly while tying the residual to the motion field).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_mse(a, b, mask=None):
    if mask is None:
        return F.mse_loss(a, b)
    m = mask.expand_as(a)
    denom = m.sum().clamp_min(1.0)
    return ((a - b) ** 2 * m).sum() / denom


def gradient_penalty(dvf):
    """Mean squared spatial gradient of the DVF (smoothness regulariser)."""
    dz = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
    dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
    dx = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3.0


def cycle_consistency(dvf_a, dvf_b):
    """Penalise disagreement between the two routes to the same DVF label."""
    return F.mse_loss(dvf_a, dvf_b)


def residual_dvf_consistency(residual, dvf, transform):
    """
    Discourage the residual from explaining motion the DVF should carry.

    Intuition: if the residual is reconstructing a displaced structure, then warping
    the *updated* volume by a small fraction of the DVF should look much like warping
    the warped volume by the same fraction. We penalise the residual's sensitivity to
    that incremental motion, pushing structure into the DVF and leaving the residual
    to intensity-only corrections.
    """
    eps = 0.1
    warped_res = transform(residual, dvf * eps)
    return F.mse_loss(warped_res, residual)


def compute_loss(out, batch, cfg):
    """
    out  : dict from VoxelMapRefine.forward
    batch: dict with 'target_vol', optional 'dvf_true', optional 'thorax_mask'
    cfg  : object/namespace with fields:
           supervised (bool), alpha (float), lambda_cycle (float),
           lambda_consistency (float), coupling (str), proj_mode (str)
    """
    logs = {}
    total = 0.0

    if cfg.supervised:
        mask = batch.get("thorax_mask")
        dvf_loss = masked_mse(out["dvf"], batch["dvf_true"], mask)
        total = total + dvf_loss
        logs["dvf_mse"] = float(dvf_loss.detach())
    else:
        img_loss = masked_mse(out["updated"], batch["target_vol"],
                            batch.get("thorax_mask"))
        smooth = gradient_penalty(out["dvf"])
        total = total + img_loss + cfg.alpha * smooth
        logs["img_mse"] = float(img_loss.detach())
        logs["smooth"] = float(smooth.detach())

    # In the supervised case we can still supervise the *image* if a target is given,
    # so the residual arm has signal even when the primary loss is on the DVF.
    if cfg.supervised and "target_vol" in batch and out.get("residual") is not None:
        img_loss = masked_mse(out["updated"], batch["target_vol"],
                            batch.get("thorax_mask"))
        total = total + img_loss
        logs["img_mse"] = float(img_loss.detach())

    if cfg.proj_mode == "cycle" and "dvf_b" in out:
        cyc = cycle_consistency(out["dvf"], out["dvf_b"])
        total = total + cfg.lambda_cycle * cyc
        logs["cycle"] = float(cyc.detach())

    if cfg.coupling == "consistency" and out.get("residual") is not None:
        con = residual_dvf_consistency(out["residual"], out["dvf"], cfg._transform)
        total = total + cfg.lambda_consistency * con
        logs["consistency"] = float(con.detach())

    logs["total"] = float(total.detach())
    return total, logs
