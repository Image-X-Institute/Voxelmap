"""DIR-Lab landmarks and target registration error, for evaluation only.

The 75-point Sampled4D sets exist at T00-T50, so they give one procedure across
every moving phase; all reported TREs use them. Coordinates follow data.py: the
cached box is (AP, SI, LR) with SI flipped, and the model works in [-1, 1]
(x, y, z) = (LR, SI, AP).
"""
import json

import numpy as np
import torch

from utilities.data import (CASE_INFO, DIRLAB_ROOT, HALF_MM_XYZ, PHASES, VOL_ZYX, VOX_MM,
                            _cache_dir, _spacing_zyx)


def _path_4d(case, phase):
    hits = sorted((DIRLAB_ROOT / f'Case{case}Pack').rglob(f'*4D-75*{phase}*.txt'))
    if not hits:
        raise FileNotFoundError(f'no 75-point {phase} set for case {case}')
    return hits[0]


def _read(path):
    """(n, 3) original-matrix (x, y, z) indices, zero-based (the files are one-based)."""
    rows = [r.split() for r in path.read_text().strip().splitlines() if r.strip()]
    return np.array([[float(x) for x in r[:3]] for r in rows], dtype=np.float64) - 1.0


def _normalise(case, lm, crop):
    lm = lm.copy()
    (_, _, nz), _ = CASE_INFO[case]
    lm[:, 2] = (nz - 1) - lm[:, 2]                         # SI flip, as read_img
    native = lm[:, [1, 2, 0]]                              # -> AP, SI, LR
    vox = (native - np.array(crop['lo_zyx'])) * (np.array(_spacing_zyx(case)) / VOX_MM)
    norm = (vox + 0.5) / np.array(VOL_ZYX) * 2.0 - 1.0     # z, y, x
    return np.ascontiguousarray(norm[:, ::-1]).astype(np.float32)


def landmarks_by_bin(case):
    """{bin: (75, 3) normalised (x, y, z)} for every phase with a 75-point set."""
    crop = json.loads((_cache_dir(case) / 'crop.json').read_text())
    out = {}
    for b, ph in enumerate(PHASES, start=1):
        try:
            out[b] = _normalise(case, _read(_path_4d(case, ph)), crop)
        except FileNotFoundError:
            continue
    return out


def tre_mm(pred_xyz, true_xyz):
    """Normalised units span [-1, 1], so the scale is half the box extent."""
    return np.linalg.norm((np.asarray(pred_xyz) - np.asarray(true_xyz)) * HALF_MM_XYZ, axis=-1)


def sample_dvf(dvf, pts_xyz):
    """dvf (B, 3, Z, Y, X) as (dz, dy, dx) at normalised points (n, 3) (x, y, z)
    -> (B, n, 3) displacement in (x, y, z)."""
    p = torch.as_tensor(pts_xyz, device=dvf.device, dtype=dvf.dtype)
    g = p.view(1, -1, 1, 1, 3).expand(dvf.shape[0], -1, -1, -1, -1)
    s = torch.nn.functional.grid_sample(dvf, g, mode='bilinear', padding_mode='border',
                                        align_corners=True)[:, :, :, 0, 0]
    return torch.stack([s[:, 2], s[:, 1], s[:, 0]], dim=-1)
