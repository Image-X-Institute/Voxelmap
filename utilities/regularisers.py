"""Comparator regularizers.

Each returns a non-negative per-voxel penalty map (B, Z', Y', X') on the same
every-second-voxel grid as objective.det_jacobian, so that sqrt(map) are exact
Gauss-Newton rows and the c-step descends the identical mean. All are computed
from central differences of the displacement in voxel units; no mask.

    bending    sum_k [sum_a (d_aa u_k)^2 + 2 sum_{a<b} (d_ab u_k)^2]   Rueckert et al 1999
    diffusion  sum_{k,a} (d_a u_k)^2                                  VoxelMorph default
    tv         sum_k sqrt(|grad u_k|^2 + eta)                         Vishnevskiy et al 2017 (pTVreg)
    selfcon    |(grad u) u|^2, the first-order inverse error         
    jneg       relu(-det J)                                           Mok & Chung 2020
    logj       sqrt(log(det J)^2 + eta), log continued linearly below v0   Rohlfing et al 2003
"""
import torch

STRIDE = 2
ETA = 1e-4      # smoothing of the non-differentiable points of tv and logj
V0 = 0.1        # logj: log J continues as its tangent below J = V0, so folds stay finite


def field_gradient(d, stride=STRIDE):
    """(B, 3, 3, Z', Y', X') with [:, k, a] = d u_k / d x_a in voxel units, for a DVF d
    (B, 3, Z, Y, X) in grid_sample's normalised units (dz, dy, dx), align_corners=True."""
    n = d.shape[-3:]
    sc = torch.tensor([(m - 1) / 2.0 for m in n], device=d.device, dtype=d.dtype)
    u = d[..., ::stride, ::stride, ::stride] * sc.view(1, 3, 1, 1, 1)
    return torch.stack([torch.gradient(u, dim=k, spacing=float(stride))[0] for k in (2, 3, 4)], 2)


def _det(G):
    a = [[G[:, i, j] + (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def penalty(d, name):
    G = field_gradient(d)
    if name == 'diffusion':
        return G.pow(2).sum((1, 2))
    if name == 'tv':
        return (G.pow(2).sum(2) + ETA).sqrt().sum(1)
    if name == 'selfcon':
        n = d.shape[-3:]
        sc = torch.tensor([(m - 1) / 2.0 for m in n], device=d.device, dtype=d.dtype)
        u = d[..., ::STRIDE, ::STRIDE, ::STRIDE] * sc.view(1, 3, 1, 1, 1)     # (B, 3, ...), zyx
        return torch.einsum('bka...,ba...->bk...', G, u).pow(2).sum(1)
    if name == 'bending':
        # second derivatives: d/dx_b of G[:, k, a]
        H = torch.stack([torch.gradient(G, dim=k, spacing=float(STRIDE))[0] for k in (3, 4, 5)], 3)
        return H.pow(2).sum((1, 2, 3))          # off-diagonals appear twice: the factor 2
    J = _det(G)
    if name == 'jneg':
        return torch.relu(-J)
    if name == 'logj':
        Jc = J.clamp_min(V0)
        lg = torch.where(J < V0, torch.log(torch.tensor(V0, device=J.device)) + (J - V0) / V0, torch.log(Jc))
        return (lg.pow(2) + ETA).sqrt()
    raise ValueError(f'reg {name!r}')
