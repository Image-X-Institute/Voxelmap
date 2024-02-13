import torch
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd
import numpy as np
import math

class image:
    """
    Computes the MSE between a predicted and ground-truth image
    """

    def loss(self, target_vol, predict_vol):
        error = target_vol - predict_vol
        return torch.mean(error ** 2)

class image_mask:
    """
    Computes the MSE between a predicted and ground-truth image inside a binary mask
    """

    def loss(self, target_vol, predict_vol, mask):
        error = target_vol - predict_vol
        error[mask == 0] = 0
        return torch.sum(error ** 2) / torch.count_nonzero(error)

class flow:
    """
    Computes the MSE between a predicted and ground-truth DVF
    """

    def loss(self, target_flow, predict_flow):
        error = target_flow - predict_flow
        return torch.mean(error ** 2)

class flow_mask:
    """
    Computes the MSE between a predicted and ground-truth DVF inside a binary mask
    """

    def loss(self, target_flow, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        error = target_flow - predict_flow
        error[mask == 0] = 0
        return torch.sum(error ** 2) / torch.count_nonzero(error)

class flow_ptv:
    """
    Computes the mean 3D flows inside a PTV mask
    """

    def loss(self, flow, mask):
        mask = mask[:, 0, :, :, :]
        lr = flow[:, 0, :, :, :]
        si = flow[:, 1, :, :, :]
        ap = flow[:, 2, :, :, :]

        lr[mask == 0] = 0
        si[mask == 0] = 0
        ap[mask == 0] = 0

        lr = torch.sum(lr) / torch.count_nonzero(lr)
        si = torch.sum(si) / torch.count_nonzero(si)
        ap = torch.sum(ap) / torch.count_nonzero(ap)
        return lr, si, ap

class centroid_error:
    """
    Computes the lr,si,ap error between the centroids of two masks
    """

    def loss(self, target, pred):
        ind = np.nonzero(target)
        lr_tar = int(np.mean(ind[0]))
        si_tar = int(np.mean(ind[1]))
        ap_tar = int(np.mean(ind[2]))

        ind = np.nonzero(pred)
        lr_pre = int(np.mean(ind[0]))
        si_pre = int(np.mean(ind[1]))
        ap_pre = int(np.mean(ind[2]))

        lr = lr_tar - lr_pre
        si = si_tar - si_pre
        ap = ap_tar - ap_pre
        return lr, si, ap

class centroid_ptv:
    """
    Computes the lr,si,ap position of a binary mask
    """

    def loss(self, mask):
        metric_input = mask.cpu().detach().numpy()
        mask = np.asarray(metric_input[0][:][:][:][0], dtype=np.float32)
        ind = np.nonzero(mask)

        lr = int(np.mean(ind[0]))
        si = int(np.mean(ind[1]))
        ap = int(np.mean(ind[2]))

        return lr, si, ap

class l2:
    """
    Computes the squared L2-norm of a predicted DVF
    """

    def loss(self, predict_flow):
        return torch.mean(predict_flow ** 2)

class l2_mask:
    """
    Computes the squared L2-norm of a predicted DVF inside a binary mask
    """

    def loss(self, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        predict_flow[mask == 0] = 0
        return torch.sum(predict_flow ** 2) / torch.count_nonzero(mask)

class grad:
    """
    Simplified gradient loss
    """

    def loss(self, predict_flow):
        dy = torch.abs(predict_flow[:, :, 1:, :, :] - predict_flow[:, :, :-1, :, :])
        dx = torch.abs(predict_flow[:, :, :, 1:, :] - predict_flow[:, :, :, :-1, :])
        dz = torch.abs(predict_flow[:, :, :, :, 1:] - predict_flow[:, :, :, :, :-1])
        d = torch.mean(dx ** 2) + torch.mean(dy ** 2) + torch.mean(dz ** 2)
        return d / 3

class grad_mask:
    """
    Simplified gradient loss inside a binary mask
    """

    def loss(self, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        predict_flow[mask == 0] = 0

        dy = predict_flow[:, :, 1:, :, :] - predict_flow[:, :, :-1, :, :]
        dx = predict_flow[:, :, :, 1:, :] - predict_flow[:, :, :, :-1, :]
        dz = predict_flow[:, :, :, :, 1:] - predict_flow[:, :, :, :, :-1]
        d = torch.sum(dx ** 2) + torch.sum(dy ** 2) + torch.sum(dz ** 2)
        return d / (3 * torch.count_nonzero(mask))

class dist3d:
    """
    Mean 3D error between a predicted and ground-truth DVF
    """

    def loss(self, target_flow, predict_flow):
        dx = target_flow[:, 0, :, :, :] - predict_flow[:, 0, :, :, :]
        dy = target_flow[:, 1, :, :, :] - predict_flow[:, 1, :, :, :]
        dz = target_flow[:, 2, :, :, :] - predict_flow[:, 2, :, :, :]
        return torch.mean(torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

class dist3d_mask:
    """
    Mean 3D error between a predicted and ground-truth DVF inside a binary mask
    """

    def loss(self, target_flow, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        target_flow[mask == 0] = 0
        predict_flow[mask == 0] = 0

        dx = target_flow[:, 0, :, :, :] - predict_flow[:, 0, :, :, :]
        dy = target_flow[:, 1, :, :, :] - predict_flow[:, 1, :, :, :]
        dz = target_flow[:, 2, :, :, :] - predict_flow[:, 2, :, :, :]
        return torch.sum(torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)) / torch.count_nonzero(mask)

class BindingEnergy:
    """
    3D binding energy loss
    """

    def loss(self, flow):
        # compute derivatives
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dx2 = torch.abs(dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :])
        dxdy = torch.abs(dx[:, :, 1:, :, :] - dx[:, :, :-1, :, :])

        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dy2 = torch.abs(dy[:, :, 1:, :, :] - dy[:, :, :-1, :, :])
        dydz = torch.abs(dy[:, :, :, :, 1:] - dy[:, :, :, :, :-1])

        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        dz2 = torch.abs(dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1])
        dxdz = torch.abs(dx[:, :, :, :, 1:] - dx[:, :, :, :, :-1])

        # reshape tensors
        dx2 = dx2[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]
        dxdy = dxdy[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]

        dy2 = dy2[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]
        dydz = dydz[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]

        dz2 = dz2[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]
        dxdz = dxdz[:, :, :flow.shape[2] - 2, :flow.shape[3] - 2, :flow.shape[4] - 2]

        # sum values
        loss = torch.mean(dx2 * dx2)
        loss += torch.mean(dy2 * dy2)
        loss += torch.mean(dz2 * dz2)
        loss += 2 * torch.mean(dxdy * dxdy)
        loss += 2 * torch.mean(dydz * dydz)
        loss += 2 * torch.mean(dxdz * dxdz)
        return loss

class dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return dice

class jacobian_determinant:
    """
    jacobian determinant of a displacement field.
    """

    def loss(self, disp):

        # check inputs
        volshape = disp.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

        # compute grid
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))

        # compute gradients
        J = np.gradient(disp + grid)

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
            Jdet = Jdet0 - Jdet1 + Jdet2

            # return the proportion of element for which Jdet <= 0 #sum(i <= 0 for i in Jdet.flatten()) / Jdet.size
            return Jdet

        else:  # must be 2

            dfdx = J[0]
            dfdy = J[1]
            Jdet = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

            return Jdet

