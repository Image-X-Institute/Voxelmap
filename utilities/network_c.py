import torch
import torch.nn as nn
import numpy as np
from utilities import layers
from utilities.modelio import LoadableModel, store_config_args

class net2Dto3D(nn.Module):

    def __init__(self, im_size):
        super().__init__()
        """
        Parameters:
            im_size: Input shape. e.g. 128
        """

        # build feature list automatically
        enc_nf = []
        for nb in range(1, int(np.log2(im_size)) + 1):
            enc_nf.append(2 ** nb)

        dec_nf = enc_nf[::-1]
        dec_nf.append(3)

        self.enc_nf, self.dec_nf = enc_nf, dec_nf

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm_2d = nn.ModuleList()
        self.downarm_3d = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm_2d.append(DownBlock2d(prev_nf, nf))
            self.downarm_3d.append(DownBlock3d(prev_nf, nf))
            prev_nf = nf

        # configure decoder (up-sampling path)
        prev_nf = 2 * self.dec_nf[0]
        self.uparm = nn.ModuleList()
        for nf in self.dec_nf[:len(self.dec_nf)]:
            self.uparm.append(UpBlock(prev_nf, nf))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.dec_nf)-1:]:
            self.extras.append(ExtraBlock(prev_nf, nf))
            prev_nf = nf

    def forward(self, target_proj, source_vol):

        # get encoder activations
        target_enc = [target_proj]
        for layer in self.downarm_2d:
            target_enc.append(layer(target_enc[-1]))

        source_enc = [source_vol]
        for layer in self.downarm_3d:
            source_enc.append(layer(source_enc[-1]))

        # concatenate encoder activations
        target_enc = target_enc.pop()
        target_enc = target_enc.view(-1, round(target_enc.shape[1]), 1, 1, 1)
        source_enc = source_enc.pop()
        x = torch.cat((target_enc, source_enc), 1)

        # get decoder activations
        for layer in self.uparm:
            x = layer(x)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
        return x

class model(LoadableModel):

    @store_config_args
    def __init__(self,
                 im_size,
                 int_steps=10):
        """
        Parameters:
            im_size: Input shape. e.g. 128
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # configure core net2Dto3D model
        self.net2Dto3D_model = net2Dto3D(im_size)

        # configure optional integration layer for diffeomorphic warp
        vol_shape = [im_size, im_size, im_size]
        self.integrate = layers.VecInt(vol_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(vol_shape)

    def forward(self, target_proj, source_vol):
        '''
        Parameters:
            target_proj: Target projection tensor.
            source_vol: Source volume tensor.
        '''

        # concatenate inputs and propagate unet
        pos_flow = self.net2Dto3D_model(target_proj, source_vol)

        # integrate to produce diffeomorphic warp
        pos_flow = self.integrate(pos_flow)

        # warp image with flow field
        y_source = self.transformer(source_vol, pos_flow)

        # return warped image
        return y_source, pos_flow

class DownBlock2d(nn.Module):
    """
    Residual layers for the encoding direction
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.activation(self.conv1(x))
        conv2 = self.bn((self.conv2(conv1)))
        out = self.activation(conv1 + conv2)
        return out

class DownBlock3d(nn.Module):
    """
    Residual layers for the encoding direction
    """

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
    """
    Residual layers for the decoding direction
    """

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
    """
    Specific convolutional block with tanh activation
    """

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
