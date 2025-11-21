import torch
import torch.nn as nn
import numpy as np

from utilities import layers
from utilities.modelio import LoadableModel, store_config_args


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning on gantry angle"""
    def __init__(self, num_features, angle_dim=1):
        super().__init__()
        self.gamma_fc = nn.Linear(angle_dim, num_features)
        self.beta_fc = nn.Linear(angle_dim, num_features)
        
    def forward(self, x, angle):
        # angle shape: (batch, 1)
        gamma = self.gamma_fc(angle).unsqueeze(-1).unsqueeze(-1)  # For 2D
        beta = self.beta_fc(angle).unsqueeze(-1).unsqueeze(-1)
        
        if x.dim() == 5:  # 3D case
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            
        return gamma * x + beta


class DownBlock2D(nn.Module):
    """2D Residual block for encoding"""
    def __init__(self, in_channels, out_channels, use_film=False):
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)
        
        if use_film:
            self.film = FiLMLayer(out_channels)

    def forward(self, x, angle=None):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        
        if self.use_film and angle is not None:
            conv2 = self.film(conv2, angle)
            
        out = self.activation(conv1 + conv2)
        return out


class DownBlock3D(nn.Module):
    """3D Residual block for encoding"""
    def __init__(self, in_channels, out_channels, use_film=False):
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)
        
        if use_film:
            self.film = FiLMLayer(out_channels)

    def forward(self, x, angle=None):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        
        if self.use_film and angle is not None:
            conv2 = self.film(conv2, angle)
            
        out = self.activation(conv1 + conv2)
        return out


class TransBlock2Dto3D(nn.Module):
    """Transformation from 2D to 3D"""
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, x):
        x = x.view(-1, round(x.shape[1]), 1, 1, 1)
        return x


class UpBlock(nn.Module):
    """3D upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn(self.conv2(conv1))
        out = self.activation(conv1 + conv2)
        return out


class ExtraBlock(nn.Module):
    """Final convolution block"""
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


class BaseNetwork(nn.Module):
    """Base class for all network variants"""
    def __init__(self, im_size, use_film=False):
        super().__init__()
        self.im_size = im_size
        self.use_film = use_film
        
    def build_feature_list(self, im_size, start_level=2):
        """Build encoder/decoder feature list"""
        enc_nf = []
        for nb in range(start_level, int(np.log2(im_size)) + 2):
            enc_nf.append(2 ** nb)
        dec_nf = enc_nf[::-1]
        dec_nf.append(3)
        return enc_nf, dec_nf


class ConcatenatedEncoder(BaseNetwork):
    """Original: concatenate projections then encode"""
    def __init__(self, im_size, use_film=False):
        super().__init__(im_size, use_film)
        
        enc_nf, dec_nf = self.build_feature_list(im_size)
        self.enc_nf, self.dec_nf = enc_nf, dec_nf
        
        # Encoder for concatenated projections
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(DownBlock2D(prev_nf, nf, use_film=use_film))
            prev_nf = nf
        
        # Transform to 3D
        self.transform = TransBlock2Dto3D(prev_nf)
        
        # Decoder
        self.uparm = nn.ModuleList()
        for nf in self.dec_nf[:len(self.enc_nf)]:
            self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf
        
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, source_proj, target_proj, angle=None):
        x = torch.cat([source_proj, target_proj], dim=1)
        
        for layer in self.downarm:
            if self.use_film:
                x = layer(x, angle)
            else:
                x = layer(x)
        
        x = self.transform(x)
        
        for layer in self.uparm:
            x = layer(x)
        
        for layer in self.extras:
            x = layer(x)
        
        return x


class DualEncoder(BaseNetwork):
    """Separate identical encoders for each projection"""
    def __init__(self, im_size, use_film=False):
        super().__init__(im_size, use_film)
        
        enc_nf, dec_nf = self.build_feature_list(im_size, start_level=3)  # One fewer block
        self.enc_nf, self.dec_nf = enc_nf, dec_nf
        
        # Two identical encoders
        prev_nf = 1
        self.source_encoder = nn.ModuleList()
        self.target_encoder = nn.ModuleList()
        
        for nf in self.enc_nf:
            self.source_encoder.append(DownBlock2D(prev_nf, nf, use_film=use_film))
            self.target_encoder.append(DownBlock2D(prev_nf, nf, use_film=use_film))
            prev_nf = nf
        
        # Transform to 3D (concatenate features)
        self.transform = TransBlock2Dto3D(prev_nf * 2)
        
        # Decoder
        prev_nf = prev_nf * 2
        self.uparm = nn.ModuleList()
        for nf in self.dec_nf[:len(self.enc_nf)]:
            self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf
        
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, source_proj, target_proj, angle=None):
        x_source = source_proj
        x_target = target_proj
        
        for src_layer, tgt_layer in zip(self.source_encoder, self.target_encoder):
            if self.use_film:
                x_source = src_layer(x_source, angle)
                x_target = tgt_layer(x_target, angle)
            else:
                x_source = src_layer(x_source)
                x_target = tgt_layer(x_target)
        
        # Concatenate features
        x = torch.cat([x_source, x_target], dim=1)
        x = self.transform(x)
        
        for layer in self.uparm:
            x = layer(x)
        
        for layer in self.extras:
            x = layer(x)
        
        return x


class SeparateProjectionVolumeEncoder(BaseNetwork):
    """Separate encoders for 2D projection and 3D volume (target_proj only)"""
    def __init__(self, im_size, use_film=False):
        super().__init__(im_size, use_film)
        
        enc_nf, dec_nf = self.build_feature_list(im_size, start_level=3)  # One fewer block
        self.enc_nf, self.dec_nf = enc_nf, dec_nf
        
        # 2D encoder for target projection only
        prev_nf = 1
        self.proj_encoder = nn.ModuleList()
        for nf in self.enc_nf:
            self.proj_encoder.append(DownBlock2D(prev_nf, nf, use_film=use_film))
            prev_nf = nf
        
        proj_features = prev_nf
        
        # 3D encoder for volume
        prev_nf = 1
        self.vol_encoder = nn.ModuleList()
        for nf in self.enc_nf:
            self.vol_encoder.append(DownBlock3D(prev_nf, nf, use_film=use_film))
            prev_nf = nf
        
        vol_features = prev_nf
        
        # Transform projections to 3D
        self.transform = TransBlock2Dto3D(proj_features)
        
        # Combine features
        combined_features = proj_features + vol_features
        
        # Decoder
        prev_nf = combined_features
        self.uparm = nn.ModuleList()
        for nf in self.dec_nf[:len(self.enc_nf)]:
            self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf
        
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, target_proj, source_vol, angle=None):
        # Encode target projection only
        x_proj = target_proj
        for layer in self.proj_encoder:
            if self.use_film:
                x_proj = layer(x_proj, angle)
            else:
                x_proj = layer(x_proj)
        
        # Encode volume
        x_vol = source_vol
        for layer in self.vol_encoder:
            if self.use_film:
                x_vol = layer(x_vol, angle)
            else:
                x_vol = layer(x_vol)
        
        # Transform and combine
        x_proj = self.transform(x_proj)
        x = torch.cat([x_proj, x_vol], dim=1)
        
        for layer in self.uparm:
            x = layer(x)
        
        for layer in self.extras:
            x = layer(x)
        
        return x


class BroadcastEncoder(BaseNetwork):
    """Broadcast 2D projection to 3D, process with residual block, then encode with volume (target_proj only)"""
    def __init__(self, im_size, use_film=False):
        super().__init__(im_size, use_film)
        
        enc_nf, dec_nf = self.build_feature_list(im_size, start_level=3)  # One fewer block
        self.enc_nf, self.dec_nf = enc_nf, dec_nf
        
        # Broadcast and process target projection only
        self.proj_conv = DownBlock3D(1, enc_nf[0], use_film=use_film)
        
        # Combined 3D encoder
        prev_nf = enc_nf[0] + 1  # proj features + volume
        self.encoder = nn.ModuleList()
        for nf in self.enc_nf:
            self.encoder.append(DownBlock3D(prev_nf, nf, use_film=use_film))
            prev_nf = nf
        
        # Decoder
        self.uparm = nn.ModuleList()
        for nf in self.dec_nf[:len(self.enc_nf)]:
            self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf
        
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, target_proj, source_vol, angle=None):
        # Use target projection only
        x_proj = target_proj
        
        # Broadcast to 3D
        batch, channels, h, w = x_proj.shape
        x_proj_3d = x_proj.unsqueeze(2).repeat(1, 1, self.im_size, 1, 1)
        
        # Process with residual block
        if self.use_film:
            x_proj_3d = self.proj_conv(x_proj_3d, angle)
        else:
            x_proj_3d = self.proj_conv(x_proj_3d)
        
        # Combine with volume
        x = torch.cat([x_proj_3d, source_vol], dim=1)
        
        # Encode
        for layer in self.encoder:
            if self.use_film:
                x = layer(x, angle)
            else:
                x = layer(x)
        
        # Decode
        for layer in self.uparm:
            x = layer(x)
        
        for layer in self.extras:
            x = layer(x)
        
        return x


class Model(LoadableModel):
    """Wrapper model with integration and transformer"""
    @store_config_args
    def __init__(self, im_size, architecture='concatenated', use_film=False, int_steps=10):
        """
        Parameters:
            im_size: Input shape (e.g., 128)
            architecture: 'concatenated', 'dual', 'separate', or 'broadcast'
            use_film: Whether to use FiLM conditioning on gantry angle
            int_steps: Number of flow integration steps
        """
        super().__init__()
        
        self.architecture = architecture
        self.use_film = use_film
        
        # Select network architecture
        if architecture == 'concatenated':
            self.backbone = ConcatenatedEncoder(im_size, use_film)
        elif architecture == 'dual':
            self.backbone = DualEncoder(im_size, use_film)
        elif architecture == 'separate':
            self.backbone = SeparateProjectionVolumeEncoder(im_size, use_film)
        elif architecture == 'broadcast':
            self.backbone = BroadcastEncoder(im_size, use_film)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Integration layer
        vol_shape = [im_size, im_size, im_size]
        self.integrate = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        
        # Transformer
        self.transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, source_proj, target_proj, source_vol, angle=None):
        # Get flow field from backbone
        if self.architecture in ['concatenated', 'dual']:
            pos_flow = self.backbone(source_proj, target_proj, angle)
        else:  # separate or broadcast - only use target_proj
            pos_flow = self.backbone(target_proj, source_vol, angle)
        
        # Integrate
        if self.integrate is not None:
            pos_flow = self.integrate(pos_flow)
        
        # Transform
        y_source = self.transformer(source_vol, pos_flow)
        
        return y_source, pos_flow
