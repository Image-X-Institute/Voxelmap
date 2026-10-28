"""DIR-Lab data: reading, cropping to the reconstruction box, normalisation, and
the simulated ventilated acquisition.

    python -m voxelmap.data --case 8      # build the volume cache and stats
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# DIR-Lab is registration-gated and is not redistributed; see README.
DIRLAB_ROOT = Path(os.environ.get('VOXELMAP_DIRLAB', 'data/DIR-LAB'))
CACHE = Path(os.environ.get('DIRLAB_CACHE', str(DIRLAB_ROOT / '_cache')))

# (nx, ny, nz) and (dx, dy, dz) mm as distributed; nz is the SI (slice) axis.
CASE_INFO = {
    1:  ((256, 256, 94),  (0.97, 0.97, 2.5)),
    2:  ((256, 256, 112), (1.16, 1.16, 2.5)),
    3:  ((256, 256, 104), (1.15, 1.15, 2.5)),
    4:  ((256, 256, 99),  (1.13, 1.13, 2.5)),
    5:  ((256, 256, 106), (1.10, 1.10, 2.5)),
    6:  ((512, 512, 128), (0.97, 0.97, 2.5)),
    7:  ((512, 512, 136), (0.97, 0.97, 2.5)),
    8:  ((512, 512, 128), (0.97, 0.97, 2.5)),
    9:  ((512, 512, 128), (0.97, 0.97, 2.5)),
    10: ((512, 512, 120), (0.97, 0.97, 2.5)),
}
N_BINS = 10
PHASES = [f'T{10 * i:02d}' for i in range(N_BINS)]
REF_BIN = 1                     # T00, peak inhale: the proxy for the DIBH CBCT

# One geometry for all ten cases (IEC: x = LR, y = SI, z = AP).
# The box holds every case's landmarks with 8% margin; large patients are
# laterally truncated, as interventional CBCT is.
VOX_MM = 1.0
BOX_MM = dict(lr=288.0, si=320.0, ap=248.0)
VOL_ZYX = (int(BOX_MM['ap']), int(BOX_MM['si']), int(BOX_MM['lr']))
HALF_MM_XYZ = np.array([BOX_MM['lr'], BOX_MM['si'], BOX_MM['ap']]) / 2.0

# 1.8 mm pitch at magnification 1.5 is 1.2 mm at isocentre; u spans the
# transaxial diagonal, so the box stays in the field of view at every angle.
DET_VU = (272, 320)             # v along SI, u along LR
DET_SPACING = 1.8
SAD, SDD = 1000.0, 1500.0
N_ANGLES = 680
ARC_DEG = 360.0

MU_WATER = 0.02
MASK_THRESH = 0.01


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def geometry(n=N_ANGLES, arc=ARC_DEG):
    return dict(sid=SAD, sdd=SDD, offx=0.0, offy=0.0,
                angles=np.arange(n, dtype=np.float64) * (arc / n))


# ---------------------------------------------------------------- volumes

def _cache_dir(case):
    p = CACHE / f'case{case:02d}'
    p.mkdir(parents=True, exist_ok=True)
    return p


def _img_path(case, phase):
    # most packs are Case<N>Pack/Images/, some nest a level deeper
    pack = DIRLAB_ROOT / f'Case{case}Pack'
    for d in (pack / 'Images', pack):
        for stem in (f'case{case}_{phase}_s.img', f'case{case}_{phase}.img'):
            if (d / stem).exists():
                return d / stem
    hits = sorted(p for p in pack.rglob(f'*{phase}*.img')
                  if f'case{case}_' in p.name.lower())
    if hits:
        return hits[0]
    raise FileNotFoundError(f'no volume for case {case} phase {phase} under {pack}')


def read_img(case, phase):
    """Raw volume -> (z=AP, y=SI, x=LR) float32 HU, SI flipped."""
    (nx, ny, nz), _ = CASE_INFO[case]
    v = np.fromfile(_img_path(case, phase), dtype=np.int16)
    if v.size != nx * ny * nz:
        raise ValueError(f'case {case} {phase}: {v.size} voxels, expected {nx * ny * nz}')
    vol = (v.reshape(nz, ny, nx).astype(np.float32) - 1024.0).transpose(1, 0, 2)
    return vol[:, ::-1, :].copy()


def hu_to_mu(vol):
    """HU -> attenuation in mm^-1, clamped at air."""
    return np.clip(MU_WATER * (1.0 + vol / 1000.0), 0.0, None)


def _spacing_zyx(case):
    _, (dx, dy, dz) = CASE_INFO[case]
    return (dy, dz, dx)


def _crop(case, ref_mu):
    """Low corner of the common box, centred on the reference body. Where the
    matrix is shorter than the box the rest is air."""
    m = ref_mu > MASK_THRESH
    c = np.array([float(np.average(np.where(m)[i])) for i in range(3)])
    half = np.array(VOL_ZYX) * VOX_MM / np.array(_spacing_zyx(case)) / 2.0
    n = np.array(ref_mu.shape, dtype=np.float64)
    lo = np.clip(c - half, 0.0, np.maximum(n - 2 * half, 0.0))
    return dict(lo_zyx=lo.tolist(), half_zyx=half.tolist(), centroid_zyx=c.tolist(),
                vol_zyx=list(VOL_ZYX), vox_mm=VOX_MM)


def _resample(vol_mu, case, off):
    """Native spacing -> VOL_ZYX at VOX_MM, cropped."""
    t = torch.from_numpy(np.ascontiguousarray(vol_mu))[None, None]
    n, sp = vol_mu.shape, _spacing_zyx(case)

    def axis(n_out, lo, step, n_native):
        pos = lo + (np.arange(n_out) + 0.5) * (VOX_MM / step)
        return torch.from_numpy(((pos + 0.5) / n_native * 2.0 - 1.0).astype(np.float32))

    g = [axis(VOL_ZYX[i], off['lo_zyx'][i], sp[i], n[i]) for i in range(3)]
    mz, my, mx = torch.meshgrid(*g, indexing='ij')
    grid = torch.stack([mx, my, mz], dim=-1)[None]
    o = F.grid_sample(t, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return o[0, 0].numpy().astype(np.float32)


def _save_atomic(path, arr):
    tmp = f'{path}.{os.getpid()}.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def build_volumes(case):
    """Crop and resample all ten phases once; cached as vol_XX.npy."""
    d = _cache_dir(case)
    fp = d / 'crop.json'
    paths = [d / f'vol_{b:02d}.npy' for b in range(1, N_BINS + 1)]
    if fp.exists() and all(p.exists() for p in paths):
        off = json.loads(fp.read_text())
        if tuple(off.get('vol_zyx', ())) == VOL_ZYX and off.get('vox_mm') == VOX_MM:
            return off
    ref = hu_to_mu(read_img(case, PHASES[REF_BIN - 1]))
    off = _crop(case, ref)
    for b, (ph, p) in enumerate(zip(PHASES, paths), start=1):
        mu = ref if b == REF_BIN else hu_to_mu(read_img(case, ph))
        _save_atomic(p, _resample(mu, case, off))
        print(f'  case {case} {ph} -> {p.name}')
    fp.write_text(json.dumps(off, indent=1))
    return off


def load_volume(case, b):
    p = _cache_dir(case) / f'vol_{b:02d}.npy'
    if not p.exists():
        build_volumes(case)
    return np.load(p).astype(np.float32)


def projection_stats(case, projector=None, device='cuda'):
    """Range of the line integrals over every phase at every angle, which fixes
    the projection normalisation. Cached as stats.json."""
    fp = _cache_dir(case) / 'stats.json'
    if fp.exists():
        return json.loads(fp.read_text())
    from voxelmap.projector import Projector, project_fast
    pr = projector or Projector(geometry(), VOL_ZYX, VOX_MM, DET_VU, DET_SPACING)
    pmin, pmax = np.inf, -np.inf
    with torch.no_grad():
        for b in range(1, N_BINS + 1):
            vol = torch.from_numpy(load_volume(case, b))[None, None].to(device)
            for a in range(N_ANGLES):
                d = project_fast(vol, pr, [pr.proj_geom([a])])
                pmin, pmax = min(pmin, float(d.min())), max(pmax, float(d.max()))
            print(f'  case {case} bin {b}: projection range [{pmin:.4f}, {pmax:.4f}]')
    s = dict(prj_min=pmin, prj_max=pmax, panel=list(DET_VU))
    fp.write_text(json.dumps(s))
    return s


def norm_prj(x, s):
    return (x - s['prj_min']) / (s['prj_max'] - s['prj_min'] + 1e-7)


# ---------------------------------------------------------------- acquisition

def phase_of(index, n_angles=N_ANGLES, n_breaths=1, n_bins=N_BINS):
    """1-based phase bin seen at arc index `index`: n_breaths machine-paced
    triangular cycles (T00 -> T90 -> T00) across the arc.

    Simulation ground truth only: it chooses which phase volume a projection is
    rendered from. No part of the method reads it.
    """
    period = 2 * (n_bins - 1)
    k = int(np.floor(index * n_breaths * period / float(n_angles))) % period
    return int(k + 1 if k < n_bins else period - k + 1)


def poisson_noise(raw, i0, gen):
    """Quantum noise on line integrals: counts ~ Poisson(I0 exp(-p)),
    returned as -ln(N / I0). i0 <= 0 is noiseless."""
    if i0 <= 0:
        return raw
    n = torch.poisson(i0 * torch.exp(-raw.double()), generator=gen).clamp_min(1.0)
    return (-torch.log(n / i0)).to(raw.dtype)


def simulate_arc(case, projector, stats, device, n_breaths=1, noise_i0=0.0, seed=0):
    """The ventilated scan: projection i rendered from the phase volume the
    patient is in at arc index i. Returns (N_ANGLES, 1, V, U), normalised."""
    from voxelmap.projector import project_fast
    vols = {b: torch.from_numpy(load_volume(case, b))[None, None]
            for b in range(1, N_BINS + 1)}
    gen = torch.Generator().manual_seed(1000003 * seed + int(case))
    out = []
    with torch.no_grad():
        for a in range(N_ANGLES):
            vol = vols[phase_of(a, N_ANGLES, n_breaths)].to(device)
            raw = project_fast(vol, projector, [projector.proj_geom([a])])[0].cpu()
            out.append(poisson_noise(raw, noise_i0, gen))
    return norm_prj(torch.stack(out), stats)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--case', type=int, required=True, choices=sorted(CASE_INFO))
    P.add_argument('--gpu', type=int, default=0)
    a = P.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    build_volumes(a.case)
    print(projection_stats(a.case))
