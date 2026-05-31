"""
VoxelMap motion-guided volume refinement.

Inputs : real-time 2D projection(s) + source 3D image (e.g. planning CT)
Outputs: 3D DVF (clinically actionable) + updated 3D image = warp(source, DVF) + residual

Residual coupling is a config switch mapping onto the three embodiments:
  - 'decoupled'    : residual arm has its own encoder; reads warped volume only.
  - 'shared_latent': residual decoder reads the SAME latent as the DVF decoder.
  - 'consistency'  : decoupled, but a consistency loss ties the residual to the DVF
                     (see losses.py). Architecturally identical to 'decoupled'.

Conventions follow the published embodiment: encoding arm of n residual blocks
(2D conv 4x4 stride2 -> 3x3 stride1 -> BN), reshape latent to 3D, decoding arm of n
transpose-conv residual blocks, two final 3x3x3 convs, then scaling-and-squaring
integration and a spatial transform module.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Residual building blocks
# ----------------------------------------------------------------------------- #
class ResBlock2d(nn.Module):
    """Downsampling 2D residual block: 4x4 s2 then 3x3 s1, BN, ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # project the (downsampled) input onto the residual path
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)

    def forward(self, x):
        s = self.skip(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + s)


class ResBlock3dUp(nn.Module):
    """Upsampling 3D residual block: transpose 4x4x4 s2 then 3x3x3 s1, BN, ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        s = self.skip(x)
        h = F.relu(self.bn1(self.up(x)))
        h = self.bn2(self.conv(h))
        return F.relu(h + s)


# ----------------------------------------------------------------------------- #
# Encoder: 2D projection pair -> latent vector
# ----------------------------------------------------------------------------- #
class ProjectionEncoder(nn.Module):
    """
    Encoding arm. Input tensor (B, C_in, 2^n, 2^n) -> latent (B, latent_ch, 1, 1).
    Channels double each block, starting at base_ch. n blocks reduce 2^n -> 1.
    """

    def __init__(self, in_ch: int, n_blocks: int, base_ch: int = 4):
        super().__init__()
        blocks = []
        c_in = in_ch
        c_out = base_ch
        for _ in range(n_blocks):
            blocks.append(ResBlock2d(c_in, c_out))
            c_in = c_out
            c_out = c_out * 2
        self.blocks = nn.Sequential(*blocks)
        self.latent_ch = c_in  # channels after the final block

    def forward(self, x):
        return self.blocks(x)  # (B, latent_ch, 1, 1)


# ----------------------------------------------------------------------------- #
# DVF decoder: latent -> 3D deformation field
# ----------------------------------------------------------------------------- #
class DVFDecoder(nn.Module):
    """
    Reshape latent to a 3D tensor, upsample through n blocks to (2^n)^3,
    two final 3x3x3 convs (Tanh then linear) -> 3-channel DVF.
    """

    def __init__(self, latent_ch: int, n_blocks: int, base_ch: int = 4):
        super().__init__()
        self.latent_ch = latent_ch
        blocks = []
        c_in = latent_ch
        c_out = max(base_ch, latent_ch // 2)
        for _ in range(n_blocks):
            blocks.append(ResBlock3dUp(c_in, c_out))
            c_in = c_out
            c_out = max(base_ch, c_out // 2)
        self.blocks = nn.Sequential(*blocks)
        self.pre_out = nn.Conv3d(c_in, c_in, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv3d(c_in, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, latent):
        # latent (B, latent_ch, 1, 1) -> (B, latent_ch, 1, 1, 1)
        b, c, _, _ = latent.shape
        h = latent.view(b, c, 1, 1, 1)
        h = self.blocks(h)
        h = torch.tanh(self.pre_out(h))   # penultimate: Tanh
        v = self.out(h)                   # final: linear -> velocity/DVF field
        return v


# ----------------------------------------------------------------------------- #
# Scaling-and-squaring integration (stationary velocity field -> diffeomorphic DVF)
# ----------------------------------------------------------------------------- #
class ScalingSquaring(nn.Module):
    """Integrate a stationary velocity field via scaling and squaring."""

    def __init__(self, steps: int = 7):
        super().__init__()
        self.steps = steps

    def forward(self, vel, transformer: "SpatialTransform"):
        disp = vel / (2 ** self.steps)
        for _ in range(self.steps):
            disp = disp + transformer.warp_field(disp, disp)
        return disp


# ----------------------------------------------------------------------------- #
# Spatial transform module (the published "spatial transformation" slot)
# ----------------------------------------------------------------------------- #
class SpatialTransform(nn.Module):
    """
    Warps a volume by a displacement field using trilinear sampling.
    Builds an identity grid lazily and caches per spatial-size.
    """

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        self._grids: dict = {}

    def _identity(self, shape, device, dtype):
        key = (shape, device, dtype)
        if key not in self._grids:
            d, h, w = shape
            zz, yy, xx = torch.meshgrid(
                torch.arange(d, device=device, dtype=dtype),
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W) order x,y,z
            self._grids[key] = grid
        return self._grids[key]

    def _normalise(self, coords, shape):
        # coords (B, 3, D, H, W) in voxel units (x,y,z) -> [-1, 1] for grid_sample
        d, h, w = shape
        size = torch.tensor([w - 1, h - 1, d - 1], device=coords.device,
                            dtype=coords.dtype).view(1, 3, 1, 1, 1)
        return 2.0 * coords / size - 1.0

    def warp_field(self, field, disp):
        """Sample `field` (B,3,D,H,W) at identity+disp; used by scaling-squaring."""
        return self(field, disp)

    def forward(self, vol, disp):
        """
        vol : (B, C, D, H, W)  volume (or vector field) to sample
        disp: (B, 3, D, H, W)  displacement in voxels, channel order (x, y, z)
        """
        shape = vol.shape[2:]
        ident = self._identity(shape, vol.device, vol.dtype)  # (3,D,H,W)
        coords = ident.unsqueeze(0) + disp                    # (B,3,D,H,W)
        norm = self._normalise(coords, shape)                 # x,y,z in [-1,1]
        # grid_sample expects (B, D, H, W, 3) with last dim (x, y, z)
        grid = norm.permute(0, 2, 3, 4, 1)
        return F.grid_sample(vol, grid, mode=self.mode,
                            padding_mode="border", align_corners=True)


# ----------------------------------------------------------------------------- #
# Residual refinement arm
# ----------------------------------------------------------------------------- #
class ResidualArm(nn.Module):
    """
    Predicts a (bounded) residual correction to the warped source volume.

    decoupled / consistency: small 3D U-net style net over the warped volume.
    shared_latent          : conditions on the DVF decoder's latent in addition
                             to the warped volume (the 'two masters' embodiment).
    """

    def __init__(self, coupling: str, latent_ch: int = 0,
                residual_scale: float = 0.1):
        super().__init__()
        assert coupling in ("decoupled", "shared_latent", "consistency")
        self.coupling = coupling
        self.residual_scale = residual_scale  # bounds residual via tanh * scale

        in_ch = 1
        if coupling == "shared_latent":
            # project latent to a constant channel that we broadcast over the volume
            self.latent_proj = nn.Conv3d(latent_ch, 8, kernel_size=1)
            in_ch += 8

        self.enc1 = nn.Conv3d(in_ch, 8, 3, padding=1)
        self.enc2 = nn.Conv3d(8, 16, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1)
        self.dec2 = nn.Conv3d(16, 8, 3, padding=1)   # 16 = 8 (skip) + 8
        self.out = nn.Conv3d(8, 1, 3, padding=1)

    def forward(self, warped_vol, latent=None):
        x = warped_vol
        if self.coupling == "shared_latent":
            assert latent is not None
            b, c, _, _ = latent.shape
            lat = self.latent_proj(latent.view(b, c, 1, 1, 1))
            lat = lat.expand(-1, -1, *warped_vol.shape[2:])
            x = torch.cat([warped_vol, lat], dim=1)

        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        d1 = F.relu(self.dec1(e2))
        d1 = torch.cat([d1, e1], dim=1)
        d2 = F.relu(self.dec2(d1))
        residual = torch.tanh(self.out(d2)) * self.residual_scale  # bounded -> 0-ish
        return residual


# ----------------------------------------------------------------------------- #
# Full model
# ----------------------------------------------------------------------------- #
class VoxelMapRefine(nn.Module):
    """
    proj_mode  : 'single' (one acquired projection) or 'cycle' (two routes to the
                 same DVF; see proj-cycle). In 'cycle' two encoders/decoders predict
                 the same DVF and a cycle consistency loss is applied in training.
    coupling   : 'decoupled' | 'shared_latent' | 'consistency'
    use_residual: if False, the model is pure warping (published base embodiment).
    """

    def __init__(self, vol_size: int = 128, in_ch: int = 2,
                proj_mode: str = "single", coupling: str = "decoupled",
                use_residual: bool = True, base_ch: int = 4,
                integrate_steps: int = 7, residual_scale: float = 0.1):
        super().__init__()
        assert proj_mode in ("single", "cycle")
        n_blocks = int(round(torch.log2(torch.tensor(float(vol_size))).item()))
        self.proj_mode = proj_mode
        self.coupling = coupling
        self.use_residual = use_residual

        self.encoder = ProjectionEncoder(in_ch, n_blocks, base_ch)
        latent_ch = self.encoder.latent_ch
        self.dvf_decoder = DVFDecoder(latent_ch, n_blocks, base_ch)

        if proj_mode == "cycle":
            # second route to the same DVF label (different input projection pair)
            self.encoder_b = ProjectionEncoder(in_ch, n_blocks, base_ch)
            self.dvf_decoder_b = DVFDecoder(latent_ch, n_blocks, base_ch)

        self.integrate = ScalingSquaring(integrate_steps)
        self.transform = SpatialTransform()

        if use_residual:
            self.residual_arm = ResidualArm(coupling, latent_ch, residual_scale)

    def _predict_dvf(self, proj, encoder, decoder):
        latent = encoder(proj)
        vel = decoder(latent)
        dvf = self.integrate(vel, self.transform)
        return dvf, latent

    def forward(self, proj_a, source_vol, proj_b=None):
        """
        proj_a : (B, in_ch, H, W) acquired/source 2D projection pair (route A)
        source_vol: (B, 1, D, H, W) source 3D image to warp
        proj_b : (B, in_ch, H, W) second route input, required if proj_mode=='cycle'
        """
        out = {}
        dvf_a, latent_a = self._predict_dvf(proj_a, self.encoder, self.dvf_decoder)
        out["dvf"] = dvf_a
        out["latent"] = latent_a

        if self.proj_mode == "cycle":
            assert proj_b is not None, "proj_mode='cycle' needs proj_b"
            dvf_b, latent_b = self._predict_dvf(proj_b, self.encoder_b, self.dvf_decoder_b)
            out["dvf_b"] = dvf_b
            out["latent_b"] = latent_b

        warped = self.transform(source_vol, dvf_a)
        out["warped"] = warped

        if self.use_residual:
            latent_for_res = latent_a if self.coupling == "shared_latent" else None
            residual = self.residual_arm(warped, latent_for_res)
            out["residual"] = residual
            out["updated"] = warped + residual   # updated = warp(source) + residual
        else:
            out["updated"] = warped

        return out
