import os
import numpy as np
from utilities import network_d, losses, spatialTransform
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

expt_file = 'train_d'
test_file = 'data/combat/test'

class validateCOMBATDataset(Dataset):

    def __init__(self, im_dir=None, ref_dir=None, im_size=None):
        self.im_dir = im_dir
        self.ref_dir = ref_dir
        self.im_size = im_size

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('s.npy')])

    def __getitem__(self, idx):
        # Find target angle
        angle = os.path.join(self.im_dir, format('Angles.csv'))
        angle = np.genfromtxt(angle)
        angle = angle[idx]

        # Find target slices
        slice_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('s.npy')])
        target_file = slice_list[idx]
        slice_name = os.path.join(self.im_dir, format(target_file))
        target_slice = np.load(slice_name)

        target_slice_c = target_slice[0, :, :]
        target_slice_c = (target_slice_c - np.min(target_slice_c)) / (np.max(target_slice_c) - np.min(target_slice_c))

        target_slice_s = target_slice[1, :, :]
        target_slice_s = (target_slice_s - np.min(target_slice_s)) / (np.max(target_slice_s) - np.min(target_slice_s))

        # Find source slice a
        source_file = '06_' + target_file[3:]
        slice_name = os.path.join(self.im_dir, format(source_file))
        source_slice = np.load(slice_name)

        source_slice_c = source_slice[0, :, :]
        source_slice_c = (source_slice_c - np.min(source_slice_c)) / (np.max(source_slice_c) - np.min(source_slice_c))

        source_slice_s = source_slice[1, :, :]
        source_slice_s = (source_slice_s - np.min(source_slice_s)) / (np.max(source_slice_s) - np.min(source_slice_s))

        # Find source volume
        vol_name = os.path.join(self.ref_dir, format('sub_CT_06_mha.npy'))
        source_vol = np.load(vol_name)

        # Find source contours
        vol_name = os.path.join(self.ref_dir, format('Source_mha.npy'))
        source_ptv = np.load(vol_name)

        vol_name = os.path.join(self.ref_dir, format('Eso_source_mha.npy'))
        source_eso = np.load(vol_name)

        vol_name = os.path.join(self.ref_dir, format('Sto_source_mha.npy'))
        source_sto = np.load(vol_name)

        vol_name = os.path.join(self.ref_dir, format('LungL_source_mha.npy'))
        source_lungL = np.load(vol_name)

        vol_name = os.path.join(self.ref_dir, format('LungR_source_mha.npy'))
        source_lungR = np.load(vol_name)

        vol_name = os.path.join(self.ref_dir, format('SpCord_source_mha.npy'))
        source_spcord = np.load(vol_name)

        # Find target volume
        vol_list = sorted([n for n in os.listdir(self.im_dir) if n.startswith('Vol') & n.endswith('_mha.npy')])
        target_file = vol_list[idx]
        vol_name = os.path.join(self.im_dir, format(target_file))
        target_vol = np.load(vol_name)

        # Find target contours
        vol_name = os.path.join(self.im_dir, format('Target_' + target_file[5:10] + '_mha.npy'))
        target_ptv = np.load(vol_name)

        vol_name = os.path.join(self.im_dir, format('Eso_' + target_file[5:10] + '_mha.npy'))
        target_eso = np.load(vol_name)

        vol_name = os.path.join(self.im_dir, format('Sto_' + target_file[5:10] + '_mha.npy'))
        target_sto = np.load(vol_name)

        vol_name = os.path.join(self.im_dir, format('LungL_' + target_file[5:10] + '_mha.npy'))
        target_lungL = np.load(vol_name)

        vol_name = os.path.join(self.im_dir, format('LungR_' + target_file[5:10] + '_mha.npy'))
        target_lungR = np.load(vol_name)

        vol_name = os.path.join(self.im_dir, format('SpCord_' + target_file[5:10] + '_mha.npy'))
        target_spcord = np.load(vol_name)

        # Reshape data to correct format
        target_c = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        target_c[0, :, :] = np.asarray(target_slice_c)
        target_s = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        target_s[0, :, :] = np.asarray(target_slice_s)

        source_c = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        source_c[0, :, :] = np.asarray(source_slice_c)
        source_s = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        source_s[0, :, :] = np.asarray(source_slice_s)

        source_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_volumes[0, :, :, :] = np.asarray(source_vol)

        source_ptvs = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_ptvs[0, :, :, :] = np.asarray(source_ptv)

        source_esophagus = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_esophagus[0, :, :, :] = np.asarray(source_eso)

        source_stomach = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_stomach[0, :, :, :] = np.asarray(source_sto)

        source_lungsR = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_lungsR[0, :, :, :] = np.asarray(source_lungR)

        source_lungsL = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_lungsL[0, :, :, :] = np.asarray(source_lungL)

        source_spinalcord = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_spinalcord[0, :, :, :] = np.asarray(source_spcord)

        target_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_volumes[0, :, :, :] = np.asarray(target_vol)

        target_ptvs = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_ptvs[0, :, :, :] = np.asarray(target_ptv)

        target_esophagus = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_esophagus[0, :, :, :] = np.asarray(target_eso)

        target_stomach = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_stomach[0, :, :, :] = np.asarray(target_sto)

        target_lungsL = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_lungsL[0, :, :, :] = np.asarray(target_lungL)

        target_lungsR = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_lungsR[0, :, :, :] = np.asarray(target_lungR)

        target_spinalcord = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_spinalcord[0, :, :, :] = np.asarray(target_spcord)

        data = {'target_slice_c': torch.from_numpy(target_c),
                'target_slice_s': torch.from_numpy(target_s),
                'source_slice_c': torch.from_numpy(source_c),
                'source_slice_s': torch.from_numpy(source_s),
                'source_volumes': torch.from_numpy(source_volumes),
                'source_ptvs': torch.from_numpy(source_ptvs),
                'source_esophagus': torch.from_numpy(source_esophagus),
                'source_stomach': torch.from_numpy(source_stomach),
                'source_lungsL': torch.from_numpy(source_lungsL),
                'source_lungsR': torch.from_numpy(source_lungsR),
                'source_spinalcord': torch.from_numpy(source_spinalcord),
                'target_volumes': torch.from_numpy(target_volumes),
                'target_ptvs': torch.from_numpy(target_ptvs),
                'target_esophagus': torch.from_numpy(target_esophagus),
                'target_stomach': torch.from_numpy(target_stomach),
                'target_lungsL': torch.from_numpy(target_lungsL),
                'target_lungsR': torch.from_numpy(target_lungsR),
                'target_spinalcord': torch.from_numpy(target_spinalcord),
                'angle': angle}

        return data

# Generate test loader
batch_size = 1
im_size = 128
dataset = validateCOMBATDataset(im_dir=test_file, ref_dir=test_file + '/source', im_size=im_size)
testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Import network and set up cuda implementation
vxm_model = network_d.VxmDense(im_size, int_steps=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vxm_model.to(device)

# Load transformer
transformer = spatialTransform.Network([im_size, im_size, im_size])
transformer.to(device)

print('Number of paramters: %d' % (sum(p.numel() for p in vxm_model.parameters() if p.requires_grad)))

# Load weights
PATH = 'weights/' + expt_file + '.pth'
vxm_model.load_state_dict(torch.load(PATH, map_location=device))
vxm_model.eval()

print('Testing...')
centroid_ptv = losses.centroid_ptv()
dice = losses.dice()
jacobian_determinant = losses.jacobian_determinant()

# initialise target tracking metrics
tar_lr, tar_si, tar_ap, pred_lr, pred_si, pred_ap, test_dice, test_detJ, test_angles = [], [], [], [], [], [], [], [], []

# initialise OAR tracking metrics
stomach_dice, esophagus_dice, lungL_dice, lungR_dice, spinalcord_dice, = [], [], [], [], []

# initialise image metrics
test_mse, test_ssim, test_psnr = [], [], []

for i, data in enumerate(testloader, 0):
    (target_slice_c, target_slice_s, source_slice_c, source_slice_s, source_vol, source_ptv, source_eso, source_sto,
     source_lungL, source_lungR, source_spcord, target_vol, target_ptv, target_eso, target_sto, target_lungL, target_lungR, target_spcord, angle) = \
    data['target_slice_c'].to(device), \
        data['target_slice_s'].to(device), \
        data['source_slice_c'].to(device), \
        data['source_slice_s'].to(device), \
        data['source_volumes'].to(device), \
        data['source_ptvs'].to(device), \
        data['source_esophagus'].to(device), \
        data['source_stomach'].to(device), \
        data['source_lungsL'].to(device), \
        data['source_lungsR'].to(device), \
        data['source_spinalcord'].to(device), \
        data['target_volumes'].to(device), \
        data['target_ptvs'].to(device), \
        data['target_esophagus'].to(device), \
        data['target_stomach'].to(device), \
        data['target_lungsL'].to(device), \
        data['target_lungsR'].to(device), \
        data['target_spinalcord'].to(device), \
        data['angle'].to(device)

    predict_ptv, predict_flow = vxm_model.forward(source_slice_c, source_slice_s, target_slice_c, target_slice_s, source_ptv)

    lr, si, ap = centroid_ptv.loss(target_ptv)

    # convert to mm (to account for subsampling)
    lr = 4 * lr
    si = 2 * si
    ap = 4 * ap

    tar_lr.append(lr)
    tar_si.append(si)
    tar_ap.append(ap)

    lr, si, ap = centroid_ptv.loss(predict_ptv)

    # convert to mm (to account for subsampling)
    lr = 4 * lr
    si = 2 * si
    ap = 4 * ap

    pred_lr.append(lr)
    pred_si.append(si)
    pred_ap.append(ap)

    # compute and save dice similarity
    dice_sim = dice.loss(target_ptv, predict_ptv)
    test_dice.append(dice_sim.item())

    # compute and save Jacobian violation ratio
    disp = np.zeros((im_size, im_size, im_size, 3), dtype=np.float32)
    metric_flows = np.squeeze(predict_flow.detach().cpu().numpy())
    disp[:, :, :, 0] = metric_flows[0, :, :, :]
    disp[:, :, :, 1] = metric_flows[1, :, :, :]
    disp[:, :, :, 2] = metric_flows[2, :, :, :]
    detJ = jacobian_determinant.loss(disp)
    detJ = sum(i <= 0 for i in detJ.flatten()) / detJ.size
    test_detJ.append(detJ.item())

    # save gantry angle
    test_angles.append(angle.item())

    # warp images
    predict_vol = transformer.forward(source_vol, predict_flow)

    # detach images from GPU
    metric_input = target_vol.cpu().detach().numpy()
    metric_input = metric_input.flatten()
    metric_pred = predict_vol.cpu().detach().numpy()
    metric_pred = metric_pred.flatten()

    mse_loss = np.sqrt(np.mean(np.square((metric_input.ravel() - metric_pred.ravel()))))
    ssim_loss = ssim(metric_input, metric_pred, data_range=np.max(metric_pred) - np.min(metric_pred))
    psnr_loss = psnr(metric_input, metric_pred)

    test_mse.append(mse_loss)
    test_ssim.append(ssim_loss)
    test_psnr.append(psnr_loss)

    # stomach
    predict_sto = transformer.forward(source_sto, predict_flow)
    dice_sim = dice.loss(target_sto, predict_sto)
    stomach_dice.append(dice_sim.item())

    # esophagus
    predict_eso = transformer.forward(source_eso, predict_flow)
    dice_sim = dice.loss(target_eso, predict_eso)
    esophagus_dice.append(dice_sim.item())

    # lungL
    predict_lungL = transformer.forward(source_lungL, predict_flow)
    dice_sim = dice.loss(target_lungL, predict_lungL)
    lungL_dice.append(dice_sim.item())

    # lungR
    predict_lungR = transformer.forward(source_lungR, predict_flow)
    dice_sim = dice.loss(target_lungR, predict_lungR)
    lungR_dice.append(dice_sim.item())

    # spinal cord
    predict_spcord = transformer.forward(source_spcord, predict_flow)
    dice_sim = dice.loss(target_spcord, predict_spcord)
    spinalcord_dice.append(dice_sim.item())

    if i % 10 == 0:
        print(str(i) + ' of ' + str(len(testloader)))

if not os.path.exists('plots/' + expt_file):
    os.mkdir('plots/' + expt_file)

# save target metrics
np.save('plots/' + expt_file + '/' + 'test_angles', test_angles)
np.save('plots/' + expt_file + '/' + 'tar_lr (mm)', tar_lr)
np.save('plots/' + expt_file + '/' + 'tar_si (mm)', tar_si)
np.save('plots/' + expt_file + '/' + 'tar_ap (mm)', tar_ap)
np.save('plots/' + expt_file + '/' + 'pred_lr (mm)', pred_lr)
np.save('plots/' + expt_file + '/' + 'pred_si (mm)', pred_si)
np.save('plots/' + expt_file + '/' + 'pred_ap (mm)', pred_ap)
np.save('plots/' + expt_file + '/' + 'test_dice', test_dice)
np.save('plots/' + expt_file + '/' + 'test_detJ', test_detJ)

# save OAR metrics
np.save('plots/' + expt_file + '/' + 'stomach_dice', stomach_dice)
np.save('plots/' + expt_file + '/' + 'esophagus_dice', esophagus_dice)
np.save('plots/' + expt_file + '/' + 'lungL_dice', lungL_dice)
np.save('plots/' + expt_file + '/' + 'lungR_dice', lungR_dice)
np.save('plots/' + expt_file + '/' + 'spinalcord_dice', spinalcord_dice)

# save image metrics
np.save('plots/' + expt_file + '/' + 'test_mse', test_mse)
np.save('plots/' + expt_file + '/' + 'test_ssim', test_ssim)
np.save('plots/' + expt_file + '/' + 'test_psnr', test_psnr)

# print results
dx = tar_lr - pred_lr
dy = tar_si - pred_si
dz = tar_ap - pred_ap

print('[PTV Dice: %.2f ± %.2f' % (np.mean(test_dice), np.std(test_dice)) +
    ' | LR: %.1f ± %.1f' % (np.mean(dx), np.std(dx)) +
    ' | SI : %.1f ± %.1f' % (np.mean(dy), np.std(dy)) +
    ' | AP : %.1f ± %.1f' % (np.mean(dz), np.std(dz)) +
    ' | 3D : %.1f ± %.1f' % (np.mean(np.sqrt(dx**2 + dy**2 + dz**2)), np.std(np.sqrt(dx**2 + dy**2 + dz**2))) +
    ' | detJ: %.0E' % (np.mean(test_detJ)) + ']')

print('[Target DSC: %.2f ± %.2f' % (np.mean(test_dice), np.std(test_dice)) +
    ' | Stomach DSC: %.2f ± %.2f' % (np.mean(stomach_dice), np.std(stomach_dice)) +
    ' | Esophagus DSC: %.2f ± %.2f' % (np.mean(esophagus_dice), np.std(esophagus_dice)) +
    ' | Lung L DSC: %.2f ± %.2f' % (np.mean(lungL_dice), np.std(lungL_dice)) +
    ' | Lung R DSC: %.2f ± %.2f' % (np.mean(lungR_dice), np.std(lungR_dice)) +
    ' | Spinal cord DSC: %.2f ± %.2f' % (np.mean(spinalcord_dice), np.std(spinalcord_dice)) + ']')

print('[RMSE: %.2f ± %.2f' % (np.mean(test_mse), np.std(test_mse)) +
    ' | SSIM: %.2f ± %.2f' % (np.mean(test_ssim), np.std(test_ssim)) +
    ' | PSNR: %.1f ± %.1f' % (np.mean(test_psnr), np.std(test_psnr)) +']')

# plot centroid traces
tar_lr = tar_lr - np.mean(tar_lr)
tar_lr = [tar_lr for _, tar_lr in sorted(zip(test_angles, tar_lr))]
tar_si = tar_si - np.mean(tar_si)
tar_si = [tar_si for _, tar_si in sorted(zip(test_angles, tar_si))]
tar_ap = tar_ap - np.mean(tar_ap)
tar_ap = [tar_ap for _, tar_ap in sorted(zip(test_angles, tar_ap))]

pred_lr = pred_lr - np.mean(pred_lr)
pred_lr = [pred_lr for _, pred_lr in sorted(zip(test_angles, pred_lr))]
pred_si = pred_si - np.mean(pred_si)
pred_si = [pred_si for _, pred_si in sorted(zip(test_angles, pred_si))]
pred_ap = pred_ap - np.mean(pred_ap)
pred_ap = [pred_ap for _, pred_ap in sorted(zip(test_angles, pred_ap))]

test_angles = sorted(test_angles)

f = plt.figure()
f.add_subplot(3, 1, 1)
plt.plot(test_angles, tar_lr)
plt.plot(test_angles, pred_lr)
plt.xlabel('Angle (degrees)')
plt.ylabel('LR position\n(mm)')
plt.title('XCAT - Patient 1')
plt.legend(['Ground-truth', 'Prediction'], loc='upper right')
plt.ylim([-6, 8])
plt.xlim([0, 360])
plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])

f.add_subplot(3, 1, 2)
plt.plot(test_angles, tar_si)
plt.plot(test_angles, pred_si)
plt.xlabel('Angle (degrees)')
plt.ylabel('SI position\n(mm)')
plt.legend(['Ground-truth', 'Prediction'], loc='upper right')
plt.ylim([-6, 8])
plt.xlim([0, 360])
plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])

f.add_subplot(3, 1, 3)
plt.plot(test_angles, tar_ap)
plt.plot(test_angles, pred_ap)
plt.xlabel('Angle (degrees)')
plt.ylabel('AP position\n(mm)')
plt.legend(['Ground-truth', 'Prediction'], loc='upper right')
plt.ylim([-6, 8])
plt.xlim([0, 360])
plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])

plt.savefig('plots/' + expt_file + 'centroid_trace.png')
plt.show()
