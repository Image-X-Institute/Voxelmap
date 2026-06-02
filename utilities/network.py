"""
network.py

Unified model definitions for the ablation study.

Consequences for the implementation:
  * dvf_decoder : out_ch=3 -> integrate -> spatial transform -> warped vol
  * img_decoder : out_ch=1 -> intensity volume

Variants
--------
  proj-single  : 2-D projection pair -> DVF -> warped volume  (no image arm)
  proj-dual    : proj-single + image-decoder arm (manifold constraint)
  vol-dual     : source volume + target projection -> DVF + image arm
  vol-dual-z   : vol-dual + learned z-coordinate channel

Factory
-------
  build_model(variant, im_size=128, int_steps=7)
"""

import numpy as np
import torch
import torch.nn as nn

from utilities import layers
from utilities.modelio import LoadableModel, store_config_args


# ============================================================================
# PRIMITIVES
# ============================================================================

def _gn(ch):
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn    = _gn(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                      if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.gn(self.conv2(h))
        return self.act(h + self.skip(x))


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn    = _gn(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.skip  = (nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
                      if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.gn(self.conv2(h))
        return self.act(h + self.skip(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.conv = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn   = _gn(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(self.act(self.up(x)))))


class ExtraBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.act   = nn.Tanh()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


# ============================================================================
# ENCODER / DECODER
# ============================================================================

def _enc_nf(im_size):
    return [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]


class Encoder3D(nn.Module):
    def __init__(self, in_ch, im_size):
        super().__init__()
        nfs = _enc_nf(im_size)
        self.nfs = nfs
        layers_ = []
        prev = in_ch
        for nf in nfs:
            layers_.append(ResBlock3D(prev, nf, stride=2))
            prev = nf
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder3D(nn.Module):
    def __init__(self, enc_nf, out_ch):
        super().__init__()
        dec_nf = enc_nf[::-1] + [out_ch]
        self.uparm = nn.ModuleList()
        prev = enc_nf[-1]
        for nf in dec_nf[:len(enc_nf)]:
            self.uparm.append(UpBlock3D(prev, nf))
            prev = nf
        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock3D(prev, nf))
            prev = nf

    def forward(self, x):
        for layer in self.uparm:  x = layer(x)
        for layer in self.extras: x = layer(x)
        return x


# ============================================================================
# PROJ VARIANTS  (proj-single, proj-dual)
# ============================================================================

class ProjModel(LoadableModel):
    """
    Inputs : source_proj [B,1,H,W], target_proj [B,1,H,W], source_vol [B,1,D,H,W]
    Returns: (y_source, dvf, y_cycle|None, None, log_var_dvf|None, log_var_img|None, None)

    y_source : source_vol warped by integrated DVF (flow arm)
    y_cycle  : synthesised intensity volume from the image arm (dual only).
               This is an IMAGE in intensity space -- it is NOT integrated
               and NOT passed through the spatial transformer.
    """

    @store_config_args
    def __init__(self, im_size=128, int_steps=7, dual=False, embed_ch=16):
        super().__init__()
        self.dual     = dual
        self.embed_ch = embed_ch

        # 2-D projection embedders
        self.src_proj_embed = ResBlock2D(1, embed_ch)
        self.tgt_proj_embed = ResBlock2D(1, embed_ch)

        # 3-D encoder: 2x embed_ch (proj features broadcast) + 1 (source vol)
        enc_in = embed_ch * 2 + 1
        self.encoder     = Encoder3D(enc_in, im_size)
        enc_nf           = self.encoder.nfs

        # Flow arm: 3-channel DVF
        self.dvf_decoder = Decoder3D(enc_nf, out_ch=3)

        if dual:
            # Image arm: 1-channel intensity volume (different output space).
            self.img_decoder = Decoder3D(enc_nf, out_ch=1)
            self.log_var_dvf = nn.Parameter(torch.zeros(1))
            self.log_var_img = nn.Parameter(torch.zeros(1))
        else:
            self.img_decoder = None
            self.log_var_dvf = None
            self.log_var_img = None

        vol_shape        = [im_size] * 3
        self.integrate   = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, source_proj, target_proj, source_vol):
        B, _, D, H, W = source_vol.shape

        sp = self.src_proj_embed(source_proj).unsqueeze(2).expand(-1, -1, D, -1, -1)
        tp = self.tgt_proj_embed(target_proj).unsqueeze(2).expand(-1, -1, D, -1, -1)
        x  = torch.cat([source_vol, sp, tp], dim=1)

        bottleneck = self.encoder(x)

        # ── Flow arm: decode DVF, integrate, warp source ────────────────
        dvf = self.dvf_decoder(bottleneck)
        if self.integrate:
            dvf = self.integrate(dvf)
        y_source = self.transformer(source_vol, dvf)

        if self.dual:
            y_cycle = torch.sigmoid(self.img_decoder(bottleneck))
            return y_source, dvf, y_cycle, None, self.log_var_dvf, self.log_var_img, None

        return y_source, dvf, None, None, None, None, None


# ============================================================================
# VOL VARIANTS  (vol-dual, vol-dual-z)
# ============================================================================

class VolModel(LoadableModel):
    """
    Inputs : source_vol [B,1,D,H,W], target_proj [B,1,H,W]
    Returns: (y_source, dvf, y_cycle, None, log_var_dvf, log_var_img, None)

    Always dual: a flow arm (DVF) and an image arm (intensity volume) share
    the encoder bottleneck. y_cycle is a synthesised intensity volume --
    NOT integrated, NOT spatially transformed.
    """

    @store_config_args
    def __init__(self, im_size=128, int_steps=7, use_z_coord=False, embed_ch=16):
        super().__init__()
        self.use_z_coord = use_z_coord

        self.src_embedder  = ResBlock3D(1, embed_ch)
        self.proj_embedder = ResBlock2D(1, embed_ch)

        enc_in = embed_ch * 2 + (1 if use_z_coord else 0)
        self.encoder     = Encoder3D(enc_in, im_size)
        enc_nf           = self.encoder.nfs

        # Flow arm (DVF) and image arm (intensity) share the bottleneck.
        self.dvf_decoder = Decoder3D(enc_nf, out_ch=3)
        self.img_decoder = Decoder3D(enc_nf, out_ch=1)

        self.log_var_dvf = nn.Parameter(torch.zeros(1))
        self.log_var_img = nn.Parameter(torch.zeros(1))

        vol_shape        = [im_size] * 3
        self.integrate   = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(vol_shape)

    def _z_coord(self, source_vol):
        B, _, D, H, W = source_vol.shape
        z = torch.linspace(0, 1, D, device=source_vol.device)
        return z.view(1, 1, D, 1, 1).expand(B, 1, D, H, W)

    def forward(self, source_vol, target_proj):
        B, _, D, H, W = source_vol.shape

        src_feat  = self.src_embedder(source_vol)
        proj_feat = self.proj_embedder(target_proj).unsqueeze(2).expand(-1, -1, D, -1, -1)

        feats = [src_feat, proj_feat]
        if self.use_z_coord:
            feats.append(self._z_coord(source_vol))
        x = torch.cat(feats, dim=1)

        bottleneck = self.encoder(x)

        # ── Flow arm: decode DVF, integrate, warp source ────────────────
        dvf = self.dvf_decoder(bottleneck)
        if self.integrate:
            dvf = self.integrate(dvf)
        y_source = self.transformer(source_vol, dvf)

        # ── Image arm: decode intensity volume DIRECTLY ─────────────────
        y_cycle = torch.sigmoid(self.img_decoder(bottleneck))

        return y_source, dvf, y_cycle, None, self.log_var_dvf, self.log_var_img, None


# ============================================================================
# FACTORY
# ============================================================================

_VARIANTS = {
    'proj-single': dict(cls=ProjModel, kwargs=dict(dual=False)),
    'proj-dual':   dict(cls=ProjModel, kwargs=dict(dual=True)),
    'vol-dual':    dict(cls=VolModel,  kwargs=dict(use_z_coord=False)),
    'vol-dual-z':  dict(cls=VolModel,  kwargs=dict(use_z_coord=True)),
}

def build_model(variant, im_size=128, int_steps=7):
    if variant not in _VARIANTS:
        raise ValueError(f'Unknown variant "{variant}". Choose from: {list(_VARIANTS)}')
    cfg = _VARIANTS[variant]
    return cfg['cls'](im_size=im_size, int_steps=int_steps, **cfg['kwargs'])
