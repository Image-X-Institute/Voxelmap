import time
import os
import numpy as np
from matplotlib import pyplot as plt
from utilities import network_a, losses
import torch
from torch.utils.data import Dataset
import torch.optim as optim

im_dir = 'data/xcat/train'
expt_description = 'Network A'
filename = os.path.basename(__file__)
filename = filename[:len(filename)-3]

class SupervisedDataset(Dataset):
    def __init__(self, im_dir=None, im_size=None):
        self.im_dir = im_dir
        self.im_size = im_size
        #self.proj_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])
        # return len(self.proj_list)

    def __getitem__(self, idx):
        # Find target projection
        proj_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])
        target_file = proj_list[idx]
        proj_name = os.path.join(self.im_dir, format(target_file))
        target_proj = np.load(proj_name)
        target_proj = (target_proj - np.min(target_proj)) / (np.max(target_proj) - np.min(target_proj))

        # Find target DVF
        vol_num = target_file[:2]
        dvf_name = os.path.join(self.im_dir, format('sub_DVF_' + vol_num + '_mha.npy'))
        target_dvf = np.load(dvf_name)

        # Find source projection
        source_file = '06_' + target_file[3:]
        proj_name = os.path.join(self.im_dir, format(source_file))
        source_proj = np.load(proj_name)
        source_proj = (source_proj - np.min(source_proj)) / (np.max(source_proj) - np.min(source_proj))

        # Find source volume
        vol_name = os.path.join(self.im_dir, format('sub_CT_06_mha.npy'))
        source_vol = np.load(vol_name)
        source_vol = (source_vol - np.min(source_vol)) / (np.max(source_vol) - np.min(source_vol))

        # Find source abdomen
        vol_name = os.path.join(self.im_dir, format('sub_Abdomen_mha.npy'))
        source_hull = np.load(vol_name)

        # Reshape data
        source_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        source_projections[0, :, :] = np.asarray(source_proj)
        target_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        target_projections[0, :, :] = np.asarray(target_proj)

        source_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_volumes[0, :, :, :] = np.asarray(source_vol)
        source_abdomen = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_abdomen[0, :, :, :] = np.asarray(source_hull)

        target_flow = np.zeros((3, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_flow[0, :, :, :] = target_dvf[:, :, :, 0]
        target_flow[1, :, :, :] = target_dvf[:, :, :, 1]
        target_flow[2, :, :, :] = target_dvf[:, :, :, 2]

        data = {'source_projections': torch.from_numpy(source_projections),
                'target_projections': torch.from_numpy(target_projections),
                'source_volumes': torch.from_numpy(source_volumes),
                'source_abdomen': torch.from_numpy(source_abdomen),
                'target_flow': torch.from_numpy(target_flow)}
        return data

# generate train/test split
im_size = 128
batch_size = 8
dataset = SupervisedDataset(im_dir=im_dir, im_size=im_size)
split = [int(len(dataset) * 0.9), int(len(dataset) * 0.1)]
trainset, valset = torch.utils.data.dataset.random_split(dataset, split)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)

# set up network
model = network_a.model(im_size, int_steps=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# set loss function and optimizer
flow_mask = losses.flow_mask()
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)

print('Training (on ' + str(device) + ')...')
tic = time.time()
min_val_loss = float('inf')
train_losses, val_losses = [], []
epoch_num = 50

for epoch in range(1, epoch_num + 1):
    train_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        source_proj, target_proj, source_vol, source_abdomen, target_flow = data['source_projections'].to(device), \
                                                                            data['target_projections'].to(device), \
                                                                            data['source_volumes'].to(device), \
                                                                            data['source_abdomen'].to(device), \
                                                                            data['target_flow'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        _, predict_flow = model.forward(source_proj, target_proj, source_vol)
        loss = flow_mask.loss(target_flow, predict_flow, source_abdomen)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    # test and print every epoch
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for j, valdata in enumerate(valloader, 0):
            source_proj, target_proj, source_vol, source_abdomen, target_flow = data['source_projections'].to(device), \
                                                                                data['target_projections'].to(device), \
                                                                                data['source_volumes'].to(device), \
                                                                                data['source_abdomen'].to(device), \
                                                                                data['target_flow'].to(device)

            _, predict_flow = model.forward(source_proj, target_proj, source_vol)
            loss = flow_mask.loss(target_flow, predict_flow, source_abdomen)
            val_loss += loss.item()

    toc = time.time()
    time_elapsed = (toc - tic) / 3600
    hours = np.floor(time_elapsed)
    minutes = (time_elapsed - hours) * 60

    print('Epoch: %d | train loss: %.4f | val loss: %.4f | total time: %d hours %d minutes' %
          (epoch, train_loss / len(trainset), val_loss / len(valset), hours, minutes))
    train_losses.append(train_loss / len(trainset))
    val_losses.append(val_loss / len(valset))

    # save model with lowest validation cost
    if val_loss < min_val_loss:
        if not os.path.exists('weights'):
            os.mkdir('weights')

        PATH = 'weights/' + filename + '.pth'
        torch.save(model.state_dict(), PATH)
        min_val_loss = val_loss

    # plot training
    plt.figure()
    plt.title(expt_description)
    plt.plot(np.array(range(1, epoch + 1)), np.array(train_losses), 'b')
    plt.plot(np.array(range(1, epoch + 1)), np.array(val_losses), 'r')
    plt.legend(['Train', 'Validation'])
    plt.ylabel('Loss')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if minutes > 30:
        hours += 1
    plt.xlabel('Epochs' + ' (' + str(int(hours)) + ' hours)')
    if not os.path.exists('plots'):
        os.mkdir('plots')
    plt.savefig('plots/' + filename + '.png')
    plt.close()

print('Finished training')
torch.cuda.empty_cache()
