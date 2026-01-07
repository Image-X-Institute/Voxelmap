import torch
import torch.nn as nn
import numpy as np
from utilities import layers
from utilities.modelio import LoadableModel, store_config_args


class ProjectionEmbedder(nn.Module):
    """Embeds 2D projection into feature space"""

    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        out = self.activation(conv1 + conv2)
        return out


class Encoder2D(nn.Module):
    """2D Encoder for projection data"""

    def __init__(self, in_channels, enc_nf):
        super().__init__()
        self.downarm = nn.ModuleList()

        prev_nf = in_channels
        for nf in enc_nf:
            self.downarm.append(DownBlock2D(prev_nf, nf))
            prev_nf = nf

    def forward(self, x):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        return x_enc


class Encoder3D(nn.Module):
    """3D Encoder for volume data"""

    def __init__(self, in_channels, enc_nf):
        super().__init__()
        self.downarm = nn.ModuleList()

        prev_nf = in_channels
        for nf in enc_nf:
            self.downarm.append(DownBlock3D(prev_nf, nf))
            prev_nf = nf

    def forward(self, x):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        return x_enc


class Decoder3D(nn.Module):
    """3D Decoder with optional skip connections"""

    def __init__(self, enc_nf, dec_start_channels, out_channels=3, use_skip_connections=False):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        dec_nf = enc_nf[::-1]

        self.uparm = nn.ModuleList()
        prev_nf = dec_start_channels
        for i, nf in enumerate(dec_nf[:len(enc_nf)]):
            if use_skip_connections and i > 0:
                skip_nf = enc_nf[-(i + 1)]
                self.uparm.append(UpBlock3D(prev_nf + skip_nf, nf))
            else:
                self.uparm.append(UpBlock3D(prev_nf, nf))
            prev_nf = nf

        self.extras = nn.ModuleList()
        self.extras.append(ExtraBlock(prev_nf, out_channels))

    def forward(self, x_enc):
        x = x_enc[-1]

        for i, layer in enumerate(self.uparm):
            if self.use_skip_connections and i > 0:
                skip_idx = -(i + 1)
                x = torch.cat([x, x_enc[skip_idx]], dim=1)
            x = layer(x)

        for layer in self.extras:
            x = layer(x)

        return x


# ============================================================================
# Architecture Variants for Ablation Study
# ============================================================================

class Proj2VolRegistration(LoadableModel):
    """
    Unified projection-to-volume registration network with multiple architecture options.

    Architecture variants:
    - 'original_mri': Separate 2D encoders for each projection, 2D->3D transform at bottleneck
    - 'simple_3d': Single 3D encoder with concatenated projection features
    - 'dual_stream_2d': Separate 2D encoders, concatenate before 2D->3D transform
    - 'hybrid': 2D encoders for projections, 3D encoder for volume, merge at bottleneck
    """

    @store_config_args
    def __init__(self, im_size, architecture='original_mri', int_steps=7, skip_connections=False):
        super().__init__()

        self.im_size = im_size
        self.architecture = architecture
        self.skip_connections = skip_connections

        # Build feature dimensions
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf

        # Architecture-specific setup
        if architecture == 'original_mri':
            self._setup_original_mri(enc_nf)
        elif architecture == 'simple_3d':
            self._setup_simple_3d(enc_nf)
        elif architecture == 'dual_stream_2d':
            self._setup_dual_stream_2d(enc_nf)
        elif architecture == 'hybrid':
            self._setup_hybrid(enc_nf)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Flow integrator
        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None

        # Final transformer
        self.transformer = layers.SpatialTransformer(vol_shape)

    def _setup_original_mri(self, enc_nf):
        """Original MRI-style: separate 2D encoders, 2D->3D transform"""
        self.encoder_c = Encoder2D(2, enc_nf)  # source + target concatenated
        self.encoder_s = Encoder2D(2, enc_nf)

        # Transform block
        self.transform = TransBlock2Dto3D()

        # 3D decoder - starts with 2*enc_nf[-1] channels due to concatenation
        self.decoder = Decoder3D(enc_nf, dec_start_channels=2 * enc_nf[-1], out_channels=3, use_skip_connections=False)

    def _setup_simple_3d(self, enc_nf):
        """Simple 3D: embed projections, concatenate with volume, single 3D encoder-decoder"""
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        self.encoder = Encoder3D(2 * enc_nf[0] + 1, enc_nf)  # 2 projections + volume
        self.decoder = Decoder3D(enc_nf, dec_start_channels=enc_nf[-1], out_channels=3,
                                 use_skip_connections=self.skip_connections)

    def _setup_dual_stream_2d(self, enc_nf):
        """Dual stream: separate 2D encoders with early fusion"""
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        self.encoder_c = Encoder2D(1, enc_nf)
        self.encoder_s = Encoder2D(1, enc_nf)

        self.transform = TransBlock2Dto3D()
        self.decoder = Decoder3D(enc_nf, dec_start_channels=2 * enc_nf[-1], out_channels=3, use_skip_connections=False)

    def _setup_hybrid(self, enc_nf):
        """Hybrid: 2D encoders for projections, 3D encoder for volume, late fusion"""
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        self.encoder_c = Encoder2D(1, enc_nf)
        self.encoder_s = Encoder2D(1, enc_nf)
        self.encoder_vol = Encoder3D(1, enc_nf)

        self.transform = TransBlock2Dto3D()
        # Decoder expects concatenated features: 2D features + volume features
        self.decoder = Decoder3D(enc_nf, dec_start_channels=3 * enc_nf[-1], out_channels=3,
                                 use_skip_connections=self.skip_connections)

    def forward(self, source_c, source_s, target_c, target_s, source_vol):
        """
        Args:
            source_c: Source coronal projection [B, 1, H, W]
            source_s: Source sagittal projection [B, 1, H, W]
            target_c: Target coronal projection [B, 1, H, W]
            target_s: Target sagittal projection [B, 1, H, W]
            source_vol: Source volume [B, 1, D, H, W]
        """
        if self.architecture == 'original_mri':
            return self._forward_original_mri(source_c, source_s, target_c, target_s, source_vol)
        elif self.architecture == 'simple_3d':
            return self._forward_simple_3d(source_c, source_s, target_c, target_s, source_vol)
        elif self.architecture == 'dual_stream_2d':
            return self._forward_dual_stream_2d(source_c, source_s, target_c, target_s, source_vol)
        elif self.architecture == 'hybrid':
            return self._forward_hybrid(source_c, source_s, target_c, target_s, source_vol)

    def _forward_original_mri(self, source_c, source_s, target_c, target_s, source_vol):
        """Original MRI architecture forward pass"""
        # Concatenate source and target for each view
        cat_c = torch.cat([source_c, target_c], dim=1)
        cat_s = torch.cat([source_s, target_s], dim=1)

        # Encode separately
        enc_c = self.encoder_c(cat_c)
        enc_s = self.encoder_s(cat_s)

        # Get bottleneck features and concatenate
        feat_c = enc_c[-1]
        feat_s = enc_s[-1]
        x = torch.cat([feat_c, feat_s], dim=1)

        # Transform to 3D
        x = self.transform(x)

        # Decode (no skip connections in original)
        x_enc = [x]  # Wrap in list for decoder interface
        flow = self.decoder(x_enc)

        # Integrate and warp
        if self.integrator is not None:
            flow = self.integrator(flow)
        y_source = self.transformer(source_vol, flow)

        return y_source, flow

    def _forward_simple_3d(self, source_c, source_s, target_c, target_s, source_vol):
        """Simple 3D architecture forward pass"""
        # Embed target projections
        target_c_feat = self.proj_embedder(target_c).unsqueeze(2)
        target_s_feat = self.proj_embedder(target_s).unsqueeze(2)

        # Expand to volume depth
        depth = source_vol.shape[2]
        target_c_feat = target_c_feat.expand(-1, -1, depth, -1, -1)
        target_s_feat = target_s_feat.expand(-1, -1, depth, -1, -1)

        # Concatenate all inputs
        x = torch.cat([target_c_feat, target_s_feat, source_vol], dim=1)

        # Encode and decode
        x_enc = self.encoder(x)
        flow = self.decoder(x_enc)

        # Integrate and warp
        if self.integrator is not None:
            flow = self.integrator(flow)
        y_source = self.transformer(source_vol, flow)

        return y_source, flow

    def _forward_dual_stream_2d(self, source_c, source_s, target_c, target_s, source_vol):
        """Dual stream 2D architecture forward pass"""
        # Embed target projections
        target_c_feat = self.proj_embedder(target_c)
        target_s_feat = self.proj_embedder(target_s)

        # Encode separately
        enc_c = self.encoder_c(target_c_feat)
        enc_s = self.encoder_s(target_s_feat)

        # Concatenate at bottleneck
        feat_c = enc_c[-1]
        feat_s = enc_s[-1]
        x = torch.cat([feat_c, feat_s], dim=1)

        # Transform to 3D
        x = self.transform(x)

        # Decode
        x_enc = [x]
        flow = self.decoder(x_enc)

        # Integrate and warp
        if self.integrator is not None:
            flow = self.integrator(flow)
        y_source = self.transformer(source_vol, flow)

        return y_source, flow

    def _forward_hybrid(self, source_c, source_s, target_c, target_s, source_vol):
        """Hybrid architecture forward pass"""
        # Embed target projections
        target_c_feat = self.proj_embedder(target_c)
        target_s_feat = self.proj_embedder(target_s)

        # Encode projections in 2D
        enc_c = self.encoder_c(target_c_feat)
        enc_s = self.encoder_s(target_s_feat)

        # Encode volume in 3D
        enc_vol = self.encoder_vol(source_vol)

        # Concatenate 2D features
        feat_2d = torch.cat([enc_c[-1], enc_s[-1]], dim=1)

        # Transform to 3D
        feat_2d_3d = self.transform(feat_2d)

        # Concatenate with volume features
        x = torch.cat([feat_2d_3d, enc_vol[-1]], dim=1)

        # Decode
        x_enc = [x]
        flow = self.decoder(x_enc)

        # Integrate and warp
        if self.integrator is not None:
            flow = self.integrator(flow)
        y_source = self.transformer(source_vol, flow)

        return y_source, flow


# ============================================================================
# Building Blocks
# ============================================================================

class DownBlock2D(nn.Module):
    """2D downsampling block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        out = self.activation(conv1 + conv2)
        return out


class DownBlock3D(nn.Module):
    """3D downsampling block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        out = self.activation(conv1 + conv2)
        return out


class TransBlock2Dto3D(nn.Module):
    """Transform 2D features to 3D by reshaping"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(-1, x.shape[1], 1, 1, 1)
        return x


class UpBlock3D(nn.Module):
    """3D upsampling block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        out = self.activation(conv1 + conv2)
        return out


class ExtraBlock(nn.Module):
    """Final output block with tanh activation"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out
