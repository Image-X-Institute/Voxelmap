"""Differentiable cone-beam forward projection (ASTRA FP, adjoint BP)."""
import astra
import numpy as np
import torch
import torch.nn.functional as F
from utilities.data import norm_prj


def cone_vec(geo, det_spacing, voxel_mm):
    """ASTRA cone_vec rows (N, 12) in IEC mm, scaled to unit voxels.

    source(t) = SID (sin t, 0, cos t);  u = (cos t, 0, -sin t);  v = (0, 1, 0);
    detector centre = -(SDD - SID) (sin t, 0, cos t).
    """
    sid, sdd = geo['sid'], geo['sdd']
    t = np.deg2rad(geo['angles'])
    st, ct = np.sin(t), np.cos(t)
    zeros = np.zeros_like(t)
    src = np.stack([sid * st, zeros, sid * ct], 1)
    dctr = np.stack([-(sdd - sid) * st, zeros, -(sdd - sid) * ct], 1)
    u_hat = np.stack([ct, zeros, -st], 1)
    v_hat = np.stack([zeros, np.ones_like(t), zeros], 1)
    dctr = dctr + geo['offx'] * u_hat + geo['offy'] * v_hat
    vecs = np.concatenate([src, dctr, u_hat * det_spacing, v_hat * det_spacing], 1)
    return (vecs / voxel_mm).astype(np.float64)


class Projector:
    """Persistent ASTRA buffers linked to torch memory: no per-call allocation."""

    def __init__(self, geo, vol_shape_zyx, voxel_mm, det_shape_vu, det_spacing):
        self.vol_shape = vol_shape_zyx
        self.voxel_mm = float(voxel_mm)
        Z, Y, X = vol_shape_zyx
        self.vol_geom = astra.create_vol_geom(Y, X, Z)
        self.vecs_all = cone_vec(geo, det_spacing, voxel_mm)
        self.det_shape = det_shape_vu
        self._vol_buf = None

    def proj_geom(self, idx):
        V, U = self.det_shape
        return astra.create_proj_geom('cone_vec', V, U, self.vecs_all[idx])

    def _ensure_buffers(self, device):
        if self._vol_buf is not None:
            return
        Z, Y, X = self.vol_shape
        V, U = self.det_shape
        self._vol_buf = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
        self._sino_buf = torch.zeros((V, 1, U), dtype=torch.float32, device=device)
        vlink = astra.data3d.GPULink(self._vol_buf.data_ptr(), X, Y, Z,
                                     self._vol_buf.stride(1) * 4)
        self._slink = astra.data3d.GPULink(self._sino_buf.data_ptr(), U, 1, V,
                                           self._sino_buf.stride(0) * 4)
        self._vid = astra.data3d.link('-vol', self.vol_geom, vlink)

    def _run(self, algo, pg, data_key):
        sid = astra.data3d.link('-sino', pg, self._slink)
        cfg = astra.astra_dict(algo)
        cfg[data_key], cfg['ProjectionDataId'] = self._vid, sid
        aid = astra.algorithm.create(cfg)
        astra.algorithm.run(aid)
        astra.algorithm.delete(aid)
        astra.data3d.delete(sid)

    def fp(self, vol_zyx, pg):
        self._ensure_buffers(vol_zyx.device)
        self._vol_buf.copy_(vol_zyx)
        self._run('FP3D_CUDA', pg, 'VolumeDataId')
        return (self._sino_buf[:, 0, :] * self.voxel_mm).clone()

    def bp(self, sino_vu, pg):
        self._ensure_buffers(sino_vu.device)
        self._sino_buf[:, 0, :].copy_(sino_vu)
        self._vol_buf.zero_()
        self._run('BP3D_CUDA', pg, 'ReconstructionDataId')
        return (self._vol_buf * self.voxel_mm).clone()


class _ProjectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vol_batch, projector, pgs):
        ctx.projector, ctx.pgs = projector, pgs
        return torch.stack([projector.fp(vol_batch[b, 0], pgs[b])
                            for b in range(vol_batch.shape[0])])[:, None]

    @staticmethod
    def backward(ctx, grad_out):
        return torch.stack([ctx.projector.bp(grad_out[b, 0], ctx.pgs[b])
                            for b in range(grad_out.shape[0])])[:, None], None, None


def project_fast(vol_batch, projector, pgs):
    """(B, 1, Z, Y, X) -> raw line integrals (B, 1, V, U), differentiable."""
    return _ProjectFn.apply(vol_batch, projector, pgs)


def project(vol_batch, angle_idx, projector, vu, stats):
    """Render a batch at arc indices `angle_idx`, normalised to projection units."""
    drr = project_fast(vol_batch, projector, [projector.proj_geom([int(a)]) for a in angle_idx])
    if tuple(drr.shape[-2:]) != tuple(vu):
        drr = F.adaptive_avg_pool2d(drr, tuple(vu))
    return norm_prj(drr, stats)
