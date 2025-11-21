import torch
import torch.nn as nn
import numpy as np

from utilities import layers
from utilities.modelio import LoadableModel, store_config_args


class ProjectionEmbedder(nn.Module):
    """Embeds 2D projection into feature space compatible with 3D volumes"""

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


class Encoder3D(nn.Module):
    """3D Encoder module with down-sampling"""

    def __init__(self, in_channels, enc_nf):
        super().__init__()
        self.enc_nf = enc_nf
        self.downarm = nn.ModuleList()
        
        prev_nf = in_channels
        for nf in enc_nf:
            self.downarm.append(DownBlock3d(prev_nf, nf))
            prev_nf = nf

    def forward(self, x):
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        return x_enc


class Decoder3D(nn.Module):
    """3D Decoder module with up-sampling and optional skip connections"""

    def __init__(self, enc_nf, out_channels=3, use_skip_connections=False):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        dec_nf = enc_nf[::-1]
        dec_nf.append(out_channels)
        
        self.uparm = nn.ModuleList()
        prev_nf = enc_nf[-1]
        for i, nf in enumerate(dec_nf[:len(enc_nf)]):
            if use_skip_connections and i > 0:
                # Add skip connection from corresponding encoder level
                skip_nf = enc_nf[-(i+1)]
                self.uparm.append(UpBlock(prev_nf + skip_nf, nf))
            else:
                self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf

        self.extras = nn.ModuleList()
        for nf in dec_nf[len(enc_nf):]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, x_enc):
        x = x_enc[-1]
        decoder_features = []
        
        for i, layer in enumerate(self.uparm):
            if self.use_skip_connections and i > 0:
                # Concatenate skip connection from encoder
                skip_idx = -(i+1)
                x = torch.cat([x, x_enc[skip_idx]], dim=1)
            x = layer(x)
            decoder_features.append(x)

        for layer in self.extras:
            x = layer(x)

        return x, decoder_features


class FeatureWarper3D(nn.Module):
    """Warps 3D features at multiple resolution levels"""

    def __init__(self, im_size, num_levels):
        super().__init__()
        self.transformers = nn.ModuleList()
        
        for level in range(num_levels):
            level_size = im_size // (2 ** (num_levels - level - 1))
            level_shape = [level_size, level_size, level_size]
            self.transformers.append(layers.SpatialTransformer(level_shape))

    def forward(self, features, flows):
        """
        Args:
            features: List of feature tensors at different resolutions
            flows: List of flow fields at corresponding resolutions
        Returns:
            warped_features: List of warped feature tensors
        """
        warped_features = []
        for feat, flow, transformer in zip(features, flows, self.transformers):
            warped = transformer(feat, flow)
            warped_features.append(warped)
        return warped_features


# ============================================================================
# VARIANT 1: Single Encoder, Dual Decoders (Motion + Image)
# ============================================================================

class SingleEncoderDualDecoder(LoadableModel):
    """
    Single encoder with separate motion and image decoders.
    Motion decoder trained first, then image decoder.
    Features from encoder warped at each level before image decoder.
    """

    @store_config_args
    def __init__(self, im_size, int_steps=7, num_levels=4, skip_connections=False):
        super().__init__()
        
        self.im_size = im_size
        self.num_levels = num_levels
        self.skip_connections = skip_connections
        
        # Build feature dimensions
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf
        
        # Projection embedder
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        
        # Single shared encoder
        self.encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        
        # Motion decoder
        self.motion_decoder = Decoder3D(enc_nf, out_channels=3)
        
        # Image decoder (takes warped features, optionally with skip connections)
        if skip_connections:
            # Skip connections: concatenate warped features with motion decoder features
            # Need to adjust input channels for each level
            self.image_decoder = Decoder3D(enc_nf, out_channels=3)
            self.skip_fusion = nn.ModuleList()
            for nf in enc_nf[::-1][:num_levels]:
                # Fusion layer to combine warped features with motion decoder features
                self.skip_fusion.append(nn.Conv3d(nf * 2, nf, kernel_size=1))
        else:
            self.image_decoder = Decoder3D(enc_nf, out_channels=3)
        
        # Feature warper
        self.feature_warper = FeatureWarper3D(im_size, num_levels)
        
        # Flow integrators
        self.integrators = nn.ModuleList()
        for level in range(num_levels):
            level_size = im_size // (2 ** (num_levels - level - 1))
            level_shape = [level_size, level_size, level_size]
            self.integrators.append(
                layers.VecInt(level_shape, int_steps) if int_steps > 0 else None
            )
        
        # Final transformer
        vol_shape = [im_size, im_size, im_size]
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol, mode='motion'):
        """
        Args:
            mode: 'motion' for training motion decoder, 'image' for training image decoder
        """
        # Embed projection
        target_feat = self.proj_embedder(target_proj)
        target_feat = target_feat.unsqueeze(2)
        depth = source_vol.shape[2]
        target_feat = target_feat.expand(-1, -1, depth, -1, -1)
        
        # Concatenate and encode
        x = torch.cat([target_feat, source_vol], dim=1)
        x_enc = self.encoder(x)
        
        if mode == 'motion':
            # Motion decoder path
            final_flow, motion_decoder_features = self.motion_decoder(x_enc)
            
            # Extract flows at multiple levels
            flows_at_levels = []
            num_available = len(motion_decoder_features)
            start_idx = max(0, num_available - self.num_levels)
            selected_features = motion_decoder_features[start_idx:]
            
            for level, feat in enumerate(selected_features):
                level_flow = feat[:, :3, :, :, :]
                if self.integrators[level] is not None:
                    level_flow = self.integrators[level](level_flow)
                flows_at_levels.append(level_flow)
            
            final_integrated_flow = flows_at_levels[-1]
            y_source = self.final_transformer(source_vol, final_integrated_flow)
            
            return y_source, final_integrated_flow, flows_at_levels
        
        elif mode == 'image':
            # First get motion predictions (frozen)
            with torch.no_grad():
                _, motion_decoder_features = self.motion_decoder(x_enc)
                
                flows_at_levels = []
                num_available = len(motion_decoder_features)
                start_idx = max(0, num_available - self.num_levels)
                selected_features = motion_decoder_features[start_idx:]
                
                for level, feat in enumerate(selected_features):
                    level_flow = feat[:, :3, :, :, :]
                    if self.integrators[level] is not None:
                        level_flow = self.integrators[level](level_flow)
                    flows_at_levels.append(level_flow)
            
            # Warp encoder features at corresponding levels
            # Match encoder features to flow resolutions
            encoder_features_to_warp = x_enc[-(self.num_levels+1):-1][::-1]
            warped_encoder_features = self.feature_warper(encoder_features_to_warp, flows_at_levels)
            
            # Construct modified encoder features for image decoder
            modified_x_enc = list(x_enc)
            
            if self.skip_connections:
                # Fuse warped encoder features with motion decoder features
                for i, (warped_feat, motion_feat) in enumerate(zip(warped_encoder_features, selected_features)):
                    fused = torch.cat([warped_feat, motion_feat], dim=1)
                    fused = self.skip_fusion[i](fused)
                    # Update corresponding encoder features
                    level_idx = -(self.num_levels - i)
                    modified_x_enc[level_idx] = fused
            else:
                # Just use warped encoder features
                for i, warped_feat in enumerate(warped_encoder_features):
                    level_idx = -(self.num_levels - i)
                    modified_x_enc[level_idx] = warped_feat
            
            # Image decoder path
            final_output, image_decoder_features = self.image_decoder(modified_x_enc)
            
            return final_output, flows_at_levels[-1], flows_at_levels


# ============================================================================
# VARIANT 2: Dual Encoders, Dual Decoders (Motion + Image)
# ============================================================================

class DualEncoderDualDecoder(LoadableModel):
    """
    Separate encoders and decoders for motion and image.
    Motion encoder-decoder trained first, then image encoder-decoder.
    Features from image encoder warped at each level.
    
    Note: Skip connections from motion decoder to image decoder may not make
    semantic sense here since they operate on different feature spaces. The
    skip_connections parameter controls whether motion decoder features are
    concatenated with warped image encoder features.
    """

    @store_config_args
    def __init__(self, im_size, int_steps=7, num_levels=4, skip_connections=False):
        super().__init__()
        
        self.im_size = im_size
        self.num_levels = num_levels
        self.skip_connections = skip_connections
        
        # Build feature dimensions
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf
        
        # Projection embedder
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        
        # Motion encoder-decoder
        self.motion_encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.motion_decoder = Decoder3D(enc_nf, out_channels=3)
        
        # Image encoder-decoder
        self.image_encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.image_decoder = Decoder3D(enc_nf, out_channels=3)
        
        # Skip connection fusion (if enabled)
        # Note: This concatenates motion decoder features with warped image encoder features
        # Semantics: motion features guide/condition the image synthesis
        if skip_connections:
            self.skip_fusion = nn.ModuleList()
            for nf in enc_nf[::-1][:num_levels]:
                self.skip_fusion.append(nn.Conv3d(nf * 2, nf, kernel_size=1))
        
        # Feature warper
        self.feature_warper = FeatureWarper3D(im_size, num_levels)
        
        # Flow integrators
        self.integrators = nn.ModuleList()
        for level in range(num_levels):
            level_size = im_size // (2 ** (num_levels - level - 1))
            level_shape = [level_size, level_size, level_size]
            self.integrators.append(
                layers.VecInt(level_shape, int_steps) if int_steps > 0 else None
            )
        
        # Final transformer
        vol_shape = [im_size, im_size, im_size]
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol, mode='motion'):
        """
        Args:
            mode: 'motion' for training motion network, 'image' for training image network
        """
        # Embed projection
        target_feat = self.proj_embedder(target_proj)
        target_feat = target_feat.unsqueeze(2)
        depth = source_vol.shape[2]
        target_feat = target_feat.expand(-1, -1, depth, -1, -1)
        
        # Concatenate
        x = torch.cat([target_feat, source_vol], dim=1)
        
        if mode == 'motion':
            # Motion encoder-decoder path
            x_enc = self.motion_encoder(x)
            final_flow, motion_decoder_features = self.motion_decoder(x_enc)
            
            # Extract flows at multiple levels
            flows_at_levels = []
            num_available = len(motion_decoder_features)
            start_idx = max(0, num_available - self.num_levels)
            selected_features = motion_decoder_features[start_idx:]
            
            for level, feat in enumerate(selected_features):
                level_flow = feat[:, :3, :, :, :]
                if self.integrators[level] is not None:
                    level_flow = self.integrators[level](level_flow)
                flows_at_levels.append(level_flow)
            
            final_integrated_flow = flows_at_levels[-1]
            y_source = self.final_transformer(source_vol, final_integrated_flow)
            
            return y_source, final_integrated_flow, flows_at_levels
        
        elif mode == 'image':
            # Get motion predictions (frozen)
            with torch.no_grad():
                motion_x_enc = self.motion_encoder(x)
                _, motion_decoder_features = self.motion_decoder(motion_x_enc)
                
                flows_at_levels = []
                num_available = len(motion_decoder_features)
                start_idx = max(0, num_available - self.num_levels)
                selected_motion_features = motion_decoder_features[start_idx:]
                
                for level, feat in enumerate(selected_motion_features):
                    level_flow = feat[:, :3, :, :, :]
                    if self.integrators[level] is not None:
                        level_flow = self.integrators[level](level_flow)
                    flows_at_levels.append(level_flow)
            
            # Image encoder path
            image_x_enc = self.image_encoder(x)
            
            # Warp image encoder features at corresponding levels
            image_features_to_warp = image_x_enc[-(self.num_levels+1):-1][::-1]
            warped_image_features = self.feature_warper(image_features_to_warp, flows_at_levels)
            
            # Construct modified encoder features for image decoder
            modified_x_enc = list(image_x_enc)
            
            if self.skip_connections:
                # Fuse warped image features with motion decoder features
                # Semantics: Motion features provide guidance/conditioning for image synthesis
                for i, (warped_feat, motion_feat) in enumerate(zip(warped_image_features, selected_motion_features)):
                    fused = torch.cat([warped_feat, motion_feat], dim=1)
                    fused = self.skip_fusion[i](fused)
                    level_idx = -(self.num_levels - i)
                    modified_x_enc[level_idx] = fused
            else:
                # Just use warped image encoder features
                for i, warped_feat in enumerate(warped_image_features):
                    level_idx = -(self.num_levels - i)
                    modified_x_enc[level_idx] = warped_feat
            
            # Image decoder path
            final_output, image_decoder_features = self.image_decoder(modified_x_enc)
            
            return final_output, flows_at_levels[-1], flows_at_levels


# ============================================================================
# Original Model (for reference)
# ============================================================================

class OriginalModel(LoadableModel):
    """Original single encoder-decoder architecture"""

    @store_config_args
    def __init__(self, im_size, int_steps=7, num_levels=4):
        super().__init__()
        
        self.im_size = im_size
        self.num_levels = num_levels
        
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf
        
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        self.encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.decoder = Decoder3D(enc_nf, out_channels=3)
        
        self.integrators = nn.ModuleList()
        self.transformers = nn.ModuleList()
        
        for level in range(num_levels):
            level_size = im_size // (2 ** (num_levels - level - 1))
            level_shape = [level_size, level_size, level_size]
            self.integrators.append(
                layers.VecInt(level_shape, int_steps) if int_steps > 0 else None
            )
            self.transformers.append(layers.SpatialTransformer(level_shape))
        
        vol_shape = [im_size, im_size, im_size]
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol):
        target_feat = self.proj_embedder(target_proj)
        target_feat = target_feat.unsqueeze(2)
        depth = source_vol.shape[2]
        target_feat = target_feat.expand(-1, -1, depth, -1, -1)
        
        x = torch.cat([target_feat, source_vol], dim=1)
        x_enc = self.encoder(x)
        final_flow, decoder_features = self.decoder(x_enc)
        
        flows_at_levels = []
        num_available = len(decoder_features)
        start_idx = max(0, num_available - self.num_levels)
        selected_features = decoder_features[start_idx:]
        
        for level, feat in enumerate(selected_features):
            level_flow = feat[:, :3, :, :, :]
            if self.integrators[level] is not None:
                level_flow = self.integrators[level](level_flow)
            flows_at_levels.append(level_flow)
        
        final_integrated_flow = flows_at_levels[-1]
        y_source = self.final_transformer(source_vol, final_integrated_flow)
        
        return y_source, final_integrated_flow, flows_at_levels


# ============================================================================
# Building Blocks
# ============================================================================

class DownBlock3d(nn.Module):
    """Residual layers for the encoding direction"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn((self.conv2(conv1)))
        out = self.activation(conv1 + conv2)
        return out


class UpBlock(nn.Module):
    """Residual layers for the decoding direction"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn((self.conv2(conv1)))
        out = self.activation(conv1 + conv2)
        return out


class ExtraBlock(nn.Module):
    """Specific convolutional block with tanh activation"""

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
