"""
network_irb.py

Self-contained network definitions for the iterative-refinement-block (IRB)
ablation. No external project imports: all primitives, the spatial transformer,
and the scaling-and-squaring integrator are defined here.

Question
--------
Can a single SHARED-WEIGHT update operator, applied repeatedly, refine a
projection-conditioned 3D deformation (and optionally a bounded photometric
residual) to better reconstruct the target volume?

Architecture
------------
  src_embedder (3D)   : source_vol  [B,1,D,H,W] -> [B,C,D,H,W]
  proj_embedder (2D)  : target_proj [B,1,H,W]   -> [B,C,H,W] -> broadcast over D
  encoder (3D)        : concat -> bottleneck
  init DVF decoder    : bottleneck -> initial velocity -> integrate -> flow0
                        y0 = warp(source_vol, flow0)
  IRB (shared weights), applied num_irb times:
        warped_t  = warp(source_vol, flow_t)
        feat_t    = embed(warped_t)                       # current residual error
        h_t       = ConvGRU(h_{t-1}, [bottleneck_proj, feat_t, flow_t])
        dvel_t    = dvf_head(h_t)                          # velocity update
        flow_{t+1}= compose(integrate(dvel_t), flow_t)     # diffeomorphic compose
        y_{t+1}   = warp(source_vol, flow_{t+1})
        (dual)  img_acc += raw_dimg_t                      # accumulate raw increments
                dimg_t = img_eps * tanh(img_acc)           # bound ACCUMULATOR to eps
                y_{t+1}= y_{t+1} + dimg_t
        The accumulator (not each step) is bounded, so the total photometric
        budget is <= img_eps at every depth -- depth and image-correction
        budget are not confounded across the num_irb = 1/2/3 ablation.

Modes
-----
  baseline : initial decoder only (num_irb forced to 0)
  dvf      : initial decoder + num_irb DVF-only IRBs
  dual     : initial decoder + num_irb dual (DVF + bounded image) IRBs
  bigfly   : parameter/FLOP control -- a single deeper one-shot DVF decoder,
             NO recurrence. Used to test whether IRB gains come from the
             refinement operator vs. merely more capacity.

Forward returns a dict (stable keys across modes):
  {
    'y_final'   : [B,1,D,H,W]  final reconstructed volume
    'flow_final': [B,3,D,H,W]  final integrated displacement field
    'y_steps'   : list of [B,1,D,H,W], one per step incl. step 0 (for deep sup)
    'flow_steps': list of [B,3,D,H,W], one per step incl. step 0
    'dvel_steps': list of [B,3,...] velocity update per IRB step (len num_irb)
    'dimg_steps': list of [B,1,...] bounded image residual per IRB step
                  (dual only; else empty list)
  }

Factory
-------
  build_model(mode, num_irb=0, im_size=128, int_steps=7, img_eps=0.05)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ============================================================================
# SPATIAL TRANSFORMER + SCALING-AND-SQUARING INTEGRATION  (self-contained)
# Adapted from the VoxelMorph formulation (Balakrishnan et al.).
# ============================================================================

class SpatialTransformer(nn.Module):
    """N-D spatial transformer. flow is a displacement field [B,N,*spatial]."""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids   = torch.meshgrid(vectors, indexing='ij')
        grid    = torch.stack(grids).unsqueeze(0).type(torch.float32)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Integrate a stationary velocity field via scaling and squaring."""

    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, f'nsteps must be >= 0, got {nsteps}'
        self.nsteps = nsteps
        self.scale  = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


def compose_flows(transformer, flow_a, flow_b):
    """
    Compose two displacement fields so the result applies flow_b first then
    flow_a:  phi = flow_a ∘ flow_b, i.e. for a sampling location x,
    total displacement = flow_b(x) + flow_a(x + flow_b(x)).
    Implemented as: flow_b + warp(flow_a, flow_b).
    """
    return flow_b + transformer(flow_a, flow_b)


# ============================================================================
# PRIMITIVE BLOCKS
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
        layers_, prev = [], in_ch
        for nf in nfs:
            layers_.append(ResBlock3D(prev, nf, stride=2)); prev = nf
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x   # [B, nfs[-1], 1, 1, 1] for im_size a power of two


class Decoder3D(nn.Module):
    """Upsamples a [B,enc_nf[-1],1,1,1] bottleneck to a full-res out_ch field."""

    def __init__(self, enc_nf, out_ch, extra_depth=0):
        super().__init__()
        dec_nf = enc_nf[::-1] + [out_ch]
        self.uparm = nn.ModuleList()
        prev = enc_nf[-1]
        for nf in dec_nf[:len(enc_nf)]:
            self.uparm.append(UpBlock3D(prev, nf)); prev = nf
        # Optional extra full-resolution refinement blocks (used by 'bigfly').
        self.deep = nn.ModuleList(ResBlock3D(prev, prev) for _ in range(extra_depth))
        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock3D(prev, nf)); prev = nf

    def forward(self, x):
        for layer in self.uparm: x = layer(x)
        for layer in self.deep:  x = layer(x)
        for layer in self.extras: x = layer(x)
        return x


# ============================================================================
# CONV-GRU CELL  (operates on the spatial bottleneck of the IRB)
# ============================================================================

class ConvGRU3D(nn.Module):
    """Minimal 3D ConvGRU cell. Hidden and input are [B,C,*spatial]."""

    def __init__(self, hidden_ch, input_ch, ksize=3):
        super().__init__()
        p = ksize // 2
        self.conv_z = nn.Conv3d(hidden_ch + input_ch, hidden_ch, ksize, padding=p)
        self.conv_r = nn.Conv3d(hidden_ch + input_ch, hidden_ch, ksize, padding=p)
        self.conv_q = nn.Conv3d(hidden_ch + input_ch, hidden_ch, ksize, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z  = torch.sigmoid(self.conv_z(hx))
        r  = torch.sigmoid(self.conv_r(hx))
        q  = torch.tanh(self.conv_q(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


# ============================================================================
# ITERATIVE REFINEMENT BLOCK  (shared weights; applied num_irb times)
# ============================================================================

class IRB(nn.Module):
    """
    Shared-weight update operator (lightweight).

    Inputs per step:
      bottleneck feature (static context, tiled to GRU resolution),
      embedded current warped source (residual-error signal),
      current flow.
    Maintains a ConvGRU hidden state across steps.

    Produces a velocity update (always) and, in dual mode, a RAW image
    increment (accumulation + ε-bounding is done by the caller so the total
    photometric budget is fixed regardless of depth). Block parameters are
    identical at every step -- depth comes from repeated application.

    Memory design (24 GB-friendly):
      * The warped source is DOWNSAMPLED to gru_res BEFORE embedding, so no
        full-resolution (128^3) feature map is ever created or stored.
      * The velocity update is decoded at gru_res and the 3-CHANNEL field is
        trilinearly upsampled to full resolution (cheap; velocity is smooth).
        We never upsample feature maps.
      * The GRU runs at a configurable, small resolution (gru_res) and width
        (gru_hidden), with optional 1x1x1 convs (spatial mixing already done
        by embed/decode, so the GRU can be near channel-wise).
      * Optional gradient checkpointing of the per-step body.
    """

    def __init__(self, im_size, enc_nf, embed_ch, dual, img_eps,
                 gru_res=16, gru_hidden=96, gru_ksize=1, use_checkpoint=False):
        super().__init__()
        self.dual           = dual
        self.img_eps        = img_eps
        self.im_size        = im_size
        self.gru_res        = gru_res
        self.gru_hidden     = gru_hidden
        self.use_checkpoint = use_checkpoint

        # Embed the (already downsampled) warped source at gru_res.
        self.warp_embed = ResBlock3D(1, embed_ch)

        # Project the bottleneck context (enc_nf[-1]) down to a compact width
        # so the GRU input isn't dominated by ~256 context channels.
        ctx_proj_ch = min(enc_nf[-1], gru_hidden)
        self.ctx_proj = nn.Conv3d(enc_nf[-1], ctx_proj_ch, 1)

        input_ch  = ctx_proj_ch + embed_ch + 3   # context + warp feat + flow(3)
        self.gru  = ConvGRU3D(gru_hidden, input_ch, ksize=gru_ksize)

        # Velocity head: a couple of light convs at gru_res -> 3-channel field.
        # The FIELD (not features) is upsampled to full resolution outside.
        self.dvf_body = nn.Sequential(
            nn.Conv3d(gru_hidden, gru_hidden, 3, padding=1), _gn(gru_hidden),
            nn.ReLU(inplace=True),
        )
        self.dvf_head = nn.Conv3d(gru_hidden, 3, 3, padding=1)
        nn.init.zeros_(self.dvf_head.weight); nn.init.zeros_(self.dvf_head.bias)

        if dual:
            self.img_body = nn.Sequential(
                nn.Conv3d(gru_hidden, gru_hidden, 3, padding=1), _gn(gru_hidden),
                nn.ReLU(inplace=True),
            )
            self.img_head = nn.Conv3d(gru_hidden, 1, 3, padding=1)
            nn.init.zeros_(self.img_head.weight); nn.init.zeros_(self.img_head.bias)

    def init_hidden(self, bottleneck):
        B = bottleneck.shape[0]
        return torch.zeros(B, self.gru_hidden, self.gru_res, self.gru_res,
                           self.gru_res, device=bottleneck.device)

    def _down(self, x):
        return F.interpolate(x, size=(self.gru_res,) * 3, mode='trilinear',
                             align_corners=True)

    def _up_field(self, field):
        return F.interpolate(field, size=(self.im_size,) * 3, mode='trilinear',
                             align_corners=True)

    def _step_body(self, h, ctx_low, warped_source, flow):
        # Downsample-then-embed: never build a full-res feature map.
        feat = self.warp_embed(self._down(warped_source))   # [B,embed_ch,gru_res^3]
        fl   = self._down(flow)                              # [B,3,gru_res^3]
        x    = torch.cat([ctx_low, feat, fl], dim=1)

        h = self.gru(h, x)

        dvel_low = self.dvf_head(self.dvf_body(h))           # [B,3,gru_res^3]
        dvel     = self._up_field(dvel_low)                  # 3-channel field upsample

        dimg = None
        if self.dual:
            dimg_low = self.img_head(self.img_body(h))       # [B,1,gru_res^3]
            dimg     = self._up_field(dimg_low)
        return h, dvel, dimg

    def forward(self, h, bottleneck, warped_source, flow):
        # Project + tile bottleneck context to gru_res once per step.
        ctx_low = self.ctx_proj(bottleneck).expand(
            -1, -1, self.gru_res, self.gru_res, self.gru_res)

        if self.use_checkpoint and self.training:
            # Checkpoint the per-step body: recompute activations in backward.
            # warped_source/flow must require grad for checkpoint to track them;
            # they do (derived from differentiable warps), so this is safe.
            h, dvel, dimg = torch.utils.checkpoint.checkpoint(
                self._step_body, h, ctx_low, warped_source, flow,
                use_reentrant=False)
        else:
            h, dvel, dimg = self._step_body(h, ctx_low, warped_source, flow)
        return h, dvel, dimg


# ============================================================================
# MAIN MODEL
# ============================================================================

MODES = ('baseline', 'dvf', 'dual', 'bigfly')


class IRBNet(nn.Module):
    def __init__(self, mode='dvf', num_irb=1, im_size=128, int_steps=7,
                 embed_ch=16, img_eps=0.05,
                 gru_res=16, gru_hidden=96, gru_ksize=1, use_checkpoint=False):
        super().__init__()
        assert mode in MODES, f'mode must be one of {MODES}, got {mode}'
        self.mode    = mode
        self.im_size = im_size
        self.img_eps = img_eps

        if mode == 'baseline':
            num_irb = 0
        if mode == 'bigfly':
            num_irb = 0
        self.num_irb = num_irb
        self.dual    = (mode == 'dual')

        self.src_embedder  = ResBlock3D(1, embed_ch)
        self.proj_embedder = ResBlock2D(1, embed_ch)

        enc_in       = embed_ch * 2
        self.encoder = Encoder3D(enc_in, im_size)
        enc_nf       = self.encoder.nfs
        self.enc_nf  = enc_nf

        # Initial DVF decoder. 'bigfly' uses a deeper one-shot decoder (the
        # parameter/FLOP control) instead of any recurrence.
        extra = 3 if mode == 'bigfly' else 0
        self.init_dvf_decoder = Decoder3D(enc_nf, out_ch=3, extra_depth=extra)

        # Shared IRB (one instance, applied num_irb times).
        if num_irb > 0:
            self.irb = IRB(im_size, enc_nf, embed_ch, self.dual, img_eps,
                           gru_res=gru_res, gru_hidden=gru_hidden,
                           gru_ksize=gru_ksize, use_checkpoint=use_checkpoint)
        else:
            self.irb = None

        vol_shape        = [im_size] * 3
        self.integrate   = VecInt(vol_shape, int_steps) if int_steps > 0 else None
        self.transformer = SpatialTransformer(vol_shape)

        # Uncertainty-weighting (UW) log-variance parameters for the two final
        # reconstruction terms. They are genuinely competing observations of the
        # target (clean motion vs motion + bounded photometric correction), so
        # UW is the right tool here (cf. the per-STEP weighting, which uses a
        # fixed gamma prior because steps are an ordered sequence, not competing
        # observations). For non-dual modes only log_var_warp is used.
        self.log_var_warp = nn.Parameter(torch.zeros(1))
        self.log_var_img  = nn.Parameter(torch.zeros(1))

    def _integrate(self, vel):
        return self.integrate(vel) if self.integrate is not None else vel

    def forward(self, source_vol, target_proj):
        B, _, D, H, W = source_vol.shape

        src_feat  = self.src_embedder(source_vol)
        proj_feat = self.proj_embedder(target_proj).unsqueeze(2).expand(-1, -1, D, -1, -1)
        x         = torch.cat([src_feat, proj_feat], dim=1)
        bottleneck = self.encoder(x)

        # ── Initial DVF ────────────────────────────────────────────────
        vel0 = self.init_dvf_decoder(bottleneck)
        flow = self._integrate(vel0)
        y    = self.transformer(source_vol, flow)

        y_steps    = [y]
        flow_steps = [flow]
        dvel_steps = []
        dimg_steps = []

        # ── Iterative refinement (shared weights) ──────────────────────
        if self.irb is not None:
            h = self.irb.init_hidden(bottleneck)
            img_acc = None   # running RAW image accumulator (dual only)
            for _ in range(self.num_irb):
                warped = self.transformer(source_vol, flow)
                # detach the flow fed back as input (RAFT-style): gradients flow
                # through each step's prediction, not the full accumulation chain.
                h, dvel, dimg_raw = self.irb(h, bottleneck, warped, flow.detach())

                dflow = self._integrate(dvel)
                flow  = compose_flows(self.transformer, dflow, flow)
                y     = self.transformer(source_vol, flow)

                if dimg_raw is not None:
                    # Accumulate raw increments (parallels flow accumulation),
                    # then bound the ACCUMULATOR -> total budget <= img_eps at
                    # any depth. dimg stored is the bounded residual added to y.
                    img_acc = dimg_raw if img_acc is None else img_acc + dimg_raw
                    dimg = self.img_eps * torch.tanh(img_acc)
                    y = y + dimg
                    dimg_steps.append(dimg)

                dvel_steps.append(dvel)
                y_steps.append(y)
                flow_steps.append(flow)

        # Clean flow-only warp of the final composed field (NO image residual).
        # For non-dual modes this equals y_final; for dual it is the motion-only
        # reconstruction used by the UW warp-recon term and the test-time
        # motion-vs-image decomposition.
        y_flow_final = self.transformer(source_vol, flow)

        return {
            'y_final':      y,
            'y_flow_final': y_flow_final,
            'flow_final':   flow,
            'y_steps':      y_steps,
            'flow_steps':   flow_steps,
            'dvel_steps':   dvel_steps,
            'dimg_steps':   dimg_steps,
            'log_var_warp': self.log_var_warp,
            'log_var_img':  self.log_var_img,
        }


# ============================================================================
# FACTORY
# ============================================================================

def build_model(mode, num_irb=0, im_size=128, int_steps=7, img_eps=0.05,
                embed_ch=16, gru_res=16, gru_hidden=96, gru_ksize=1,
                use_checkpoint=False):
    return IRBNet(mode=mode, num_irb=num_irb, im_size=im_size,
                  int_steps=int_steps, embed_ch=embed_ch, img_eps=img_eps,
                  gru_res=gru_res, gru_hidden=gru_hidden, gru_ksize=gru_ksize,
                  use_checkpoint=use_checkpoint)
