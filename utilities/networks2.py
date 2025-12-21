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


# ============================================================================
# VARIANT 1: Single Encoder, Dual Decoders with Feature Concatenation
# ============================================================================

class SingleEncoderDualDecoder(LoadableModel):
    """
    Single encoder with separate motion and image decoders.
    Motion decoder trained first, then image decoder.
    Features from motion decoder concatenated with encoder features for image decoder.
    Outputs single motion field at full resolution.
    """

    @store_config_args
    def __init__(self, im_size, int_steps=7, skip_connections=False):
        super().__init__()
        
        self.im_size = im_size
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
        
        # Image decoder with adjusted input channels for concatenation
        if skip_connections:
            # Image decoder receives concatenated features from encoder and motion decoder
            self.image_decoder_uparm = nn.ModuleList()
            dec_nf = enc_nf[::-1]
            
            prev_nf = enc_nf[-1]
            for i in range(len(enc_nf)):
                current_dec_nf = dec_nf[i]
                
                if i > 0:
                    # Concatenate: upsampled + encoder skip + motion decoder feature
                    skip_nf = enc_nf[-(i+1)]
                    motion_nf = dec_nf[i]
                    self.image_decoder_uparm.append(UpBlock(prev_nf + skip_nf + motion_nf, current_dec_nf))
                else:
                    self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))
                
                prev_nf = current_dec_nf
            
            self.image_decoder_extras = nn.ModuleList()
            self.image_decoder_extras.append(ExtraBlock(prev_nf, 3))
        else:
            # Without skip connections: just concatenate motion decoder features
            self.image_decoder_uparm = nn.ModuleList()
            dec_nf = enc_nf[::-1]
            
            prev_nf = enc_nf[-1]
            for i in range(len(enc_nf)):
                current_dec_nf = dec_nf[i]
                
                if i > 0:
                    motion_nf = dec_nf[i]
                    self.image_decoder_uparm.append(UpBlock(prev_nf + motion_nf, current_dec_nf))
                else:
                    self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))
                
                prev_nf = current_dec_nf
            
            self.image_decoder_extras = nn.ModuleList()
            self.image_decoder_extras.append(ExtraBlock(prev_nf, 3))
        
        # Flow integrator (single level at full resolution)
        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        
        # Final transformer
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol, mode='motion'):
        """
        Args:
            mode: 'motion' for training motion decoder, 'image' for training image decoder
        Returns:
            y_source: warped source volume
            flow: motion field (integrated if int_steps > 0)
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
            flow, motion_decoder_features = self.motion_decoder(x_enc)
            
            # Integrate flow if needed
            if self.integrator is not None:
                flow = self.integrator(flow)
            
            # Warp source volume
            y_source = self.final_transformer(source_vol, flow)
            
            return y_source, flow
        
        elif mode == 'image':
            # Get motion decoder features (frozen)
            with torch.no_grad():
                _, motion_decoder_features = self.motion_decoder(x_enc)
            
            # Image decoder with concatenated features
            x = x_enc[-1]  # Start from bottleneck
            
            for i, layer in enumerate(self.image_decoder_uparm):
                if i > 0:
                    # Concatenate features at this resolution level
                    features_to_concat = [x]
                    
                    if self.skip_connections:
                        # Add encoder skip connection
                        skip_idx = -(i+1)
                        features_to_concat.append(x_enc[skip_idx])
                    
                    # Add motion decoder feature
                    features_to_concat.append(motion_decoder_features[i])
                    
                    x = torch.cat(features_to_concat, dim=1)
                
                x = layer(x)
            
            # Final extra blocks
            for layer in self.image_decoder_extras:
                x = layer(x)
            
            flow = x
            
            # Integrate flow if needed
            if self.integrator is not None:
                flow = self.integrator(flow)
            
            # Warp source volume
            y_source = self.final_transformer(source_vol, flow)
            
            return y_source, flow


# ============================================================================
# VARIANT 2: Dual Encoders, Dual Decoders with Feature Concatenation
# ============================================================================

class DualEncoderDualDecoder(LoadableModel):
    """
    Separate encoders and decoders for motion and image.
    Motion encoder-decoder trained first, then image encoder-decoder.
    Features from motion decoder concatenated with image encoder features.
    Outputs single motion field at full resolution.
    """

    @store_config_args
    def __init__(self, im_size, int_steps=7, skip_connections=False):
        super().__init__()
        
        self.im_size = im_size
        self.skip_connections = skip_connections
        
        # Build feature dimensions
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf
        
        # Projection embedder
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        
        # Motion encoder-decoder
        self.motion_encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.motion_decoder = Decoder3D(enc_nf, out_channels=3)
        
        # Image encoder
        self.image_encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        
        # Image decoder with adjusted input channels for concatenation
        if skip_connections:
            self.image_decoder_uparm = nn.ModuleList()
            dec_nf = enc_nf[::-1]
            
            prev_nf = enc_nf[-1]
            for i in range(len(enc_nf)):
                current_dec_nf = dec_nf[i]
                
                if i > 0:
                    skip_nf = enc_nf[-(i+1)]
                    motion_nf = dec_nf[i]
                    self.image_decoder_uparm.append(UpBlock(prev_nf + skip_nf + motion_nf, current_dec_nf))
                else:
                    self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))
                
                prev_nf = current_dec_nf
            
            self.image_decoder_extras = nn.ModuleList()
            self.image_decoder_extras.append(ExtraBlock(prev_nf, 3))
        else:
            self.image_decoder_uparm = nn.ModuleList()
            dec_nf = enc_nf[::-1]
            
            prev_nf = enc_nf[-1]
            for i in range(len(enc_nf)):
                current_dec_nf = dec_nf[i]
                
                if i > 0:
                    motion_nf = dec_nf[i]
                    self.image_decoder_uparm.append(UpBlock(prev_nf + motion_nf, current_dec_nf))
                else:
                    self.image_decoder_uparm.append(UpBlock(prev_nf, current_dec_nf))
                
                prev_nf = current_dec_nf
            
            self.image_decoder_extras = nn.ModuleList()
            self.image_decoder_extras.append(ExtraBlock(prev_nf, 3))
        
        # Flow integrator (single level at full resolution)
        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        
        # Final transformer
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol, mode='motion'):
        """
        Args:
            mode: 'motion' for training motion network, 'image' for training image network
        Returns:
            y_source: warped source volume
            flow: motion field (integrated if int_steps > 0)
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
            flow, motion_decoder_features = self.motion_decoder(x_enc)
            
            # Integrate flow if needed
            if self.integrator is not None:
                flow = self.integrator(flow)
            
            # Warp source volume
            y_source = self.final_transformer(source_vol, flow)
            
            return y_source, flow
        
        elif mode == 'image':
            # Get motion predictions (frozen)
            with torch.no_grad():
                motion_x_enc = self.motion_encoder(x)
                _, motion_decoder_features = self.motion_decoder(motion_x_enc)
            
            # Image encoder path
            image_x_enc = self.image_encoder(x)
            
            # Image decoder with concatenated features
            x = image_x_enc[-1]  # Start from bottleneck
            
            for i, layer in enumerate(self.image_decoder_uparm):
                if i > 0:
                    # Concatenate features at this resolution level
                    features_to_concat = [x]
                    
                    if self.skip_connections:
                        # Add image encoder skip connection
                        skip_idx = -(i+1)
                        features_to_concat.append(image_x_enc[skip_idx])
                    
                    # Add motion decoder feature
                    features_to_concat.append(motion_decoder_features[i])
                    
                    x = torch.cat(features_to_concat, dim=1)
                
                x = layer(x)
            
            # Final extra blocks
            for layer in self.image_decoder_extras:
                x = layer(x)
            
            flow = x
            
            # Integrate flow if needed
            if self.integrator is not None:
                flow = self.integrator(flow)
            
            # Warp source volume
            y_source = self.final_transformer(source_vol, flow)
            
            return y_source, flow


# ============================================================================
# Original Model (for reference)
# ============================================================================

class OriginalModel(LoadableModel):
    """Original single encoder-decoder architecture"""

    @store_config_args
    def __init__(self, im_size, int_steps=7):
        super().__init__()
        
        self.im_size = im_size
        
        enc_nf = [2 ** nb for nb in range(2, int(np.log2(im_size)) + 2)]
        self.enc_nf = enc_nf
        
        self.proj_embedder = ProjectionEmbedder(enc_nf[0])
        self.encoder = Encoder3D(enc_nf[0] + 1, enc_nf)
        self.decoder = Decoder3D(enc_nf, out_channels=3)
        
        # Flow integrator (single level)
        vol_shape = [im_size, im_size, im_size]
        self.integrator = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None
        
        # Final transformer
        self.final_transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol, mode='motion'):
        target_feat = self.proj_embedder(target_proj)
        target_feat = target_feat.unsqueeze(2)
        depth = source_vol.shape[2]
        target_feat = target_feat.expand(-1, -1, depth, -1, -1)
        
        x = torch.cat([target_feat, source_vol], dim=1)
        x_enc = self.encoder(x)
        flow, decoder_features = self.decoder(x_enc)
        
        # Integrate flow if needed
        if self.integrator is not None:
            flow = self.integrator(flow)
        
        # Warp source volume
        y_source = self.final_transformer(source_vol, flow)
        
        return y_source, flow


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