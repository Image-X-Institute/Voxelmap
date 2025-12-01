import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utilities import networks
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import argparse


class SupervisedDataset(Dataset):
    def __init__(self, im_dir=None, im_size=None, use_angles=False):
        self.im_dir = im_dir
        self.im_size = im_size
        self.use_angles = use_angles
        
        # Load angles if needed
        if use_angles:
            angles_path = os.path.join(im_dir, 'Angles.csv')
            self.angles_df = pd.read_csv(angles_path)

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])

    def __getitem__(self, idx):
        # Find target projection
        proj_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])
        target_file = proj_list[idx]
        proj_name = os.path.join(self.im_dir, format(target_file))
        target_proj = np.load(proj_name)
        target_proj = (target_proj - np.min(target_proj)) / (np.max(target_proj) - np.min(target_proj))

        # Find target DVF
        vol_num = target_file[:2]
        dvf_name = os.path.join(self.im_dir, format('DVF_' + vol_num + '_mha.npy'))
        target_dvf = np.load(dvf_name)

        # Find source projection
        source_file = '06_' + target_file[3:]
        proj_name = os.path.join(self.im_dir, format(source_file))
        source_proj = np.load(proj_name)
        source_proj = (source_proj - np.min(source_proj)) / (np.max(source_proj) - np.min(source_proj))

        # Find source volume
        vol_name = os.path.join(self.im_dir, format('subCT_06_mha.npy'))
        source_vol = np.load(vol_name)
        source_vol = (source_vol - np.min(source_vol)) / (np.max(source_vol) - np.min(source_vol))

        # Find source abdomen
        vol_name = os.path.join(self.im_dir, format('sub_Abdomen_mha.npy'))
        source_hull = np.load(vol_name)

        # Get angle if needed
        angle = None
        if self.use_angles:
            # Extract phase and projection number from filename
            # Assuming format: XX_YY_bin.npy where XX is phase
            angle_row = self.angles_df[self.angles_df['filename'] == target_file]
            if not angle_row.empty:
                angle = angle_row['angle'].values[0]
            else:
                angle = 0.0  # Default angle

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

        data = {
            'source_projections': torch.from_numpy(source_projections),
            'target_projections': torch.from_numpy(target_projections),
            'source_volumes': torch.from_numpy(source_volumes),
            'source_abdomen': torch.from_numpy(source_abdomen),
            'target_flow': torch.from_numpy(target_flow)
        }
        
        if self.use_angles:
            data['angle'] = torch.tensor([angle], dtype=torch.float32)
        
        return data


def train_model(args):
    """Train a specific network variant"""
    
    # Create output directory
    output_dir = f'outputs/{args.architecture}'
    if args.use_film:
        output_dir += '_film'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/weights', exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Setup dataset
    dataset = SupervisedDataset(
        im_dir=args.im_dir,
        im_size=args.im_size,
        use_angles=args.use_film
    )
    
    split = [int(len(dataset) * 0.9), int(len(dataset) * 0.1)]
    trainset, valset = torch.utils.data.dataset.random_split(dataset, split)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)
    
    # Setup model
    model = networks.Model(
        im_size=args.im_size,
        architecture=args.architecture,
        use_film=args.use_film,
        int_steps=args.int_steps
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup loss and optimizer
    MSE = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Check if architecture needs source_proj
    needs_source_proj = args.architecture in ['concatenated', 'dual']
    
    print(f'Training {args.architecture} (FiLM: {args.use_film}) on {device}...')
    tic = time.time()
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            target_proj = data['target_projections'].to(device)
            source_vol = data['source_volumes'].to(device)
            source_abdomen = data['source_abdomen'].to(device)
            target_flow = data['target_flow'].to(device)
            
            angle = data['angle'].to(device) if args.use_film else None
            
            optimizer.zero_grad()
            
            if needs_source_proj:
                source_proj = data['source_projections'].to(device)
                _, predict_flow = model.forward(source_proj, target_proj, source_vol, angle)
            else:
                _, predict_flow = model.forward(None, target_proj, source_vol, angle)
            
            loss = MSE(target_flow, predict_flow)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for j, valdata in enumerate(valloader, 0):
                target_proj = valdata['target_projections'].to(device)
                source_vol = valdata['source_volumes'].to(device)
                source_abdomen = valdata['source_abdomen'].to(device)
                target_flow = valdata['target_flow'].to(device)
                
                angle = valdata['angle'].to(device) if args.use_film else None
                
                if needs_source_proj:
                    source_proj = valdata['source_projections'].to(device)
                    _, predict_flow = model.forward(source_proj, target_proj, source_vol, angle)
                else:
                    _, predict_flow = model.forward(None, target_proj, source_vol, angle)
                
                loss = MSE(target_flow, predict_flow)
                val_loss += loss.item()
        
        # Compute time
        toc = time.time()
        time_elapsed = (toc - tic) / 3600
        hours = np.floor(time_elapsed)
        minutes = (time_elapsed - hours) * 60
        
        print('Epoch: %d | train loss: %.4f | val loss: %.4f | time: %dh %dm' %
              (epoch, train_loss / len(trainset), val_loss / len(valset), hours, minutes))
        
        train_losses.append(train_loss / len(trainset))
        val_losses.append(val_loss / len(valset))
        
        # Save best model
        if val_loss < min_val_loss:
            save_path = f'{output_dir}/weights/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss / len(trainset),
                'val_loss': val_loss / len(valset),
                'args': vars(args)
            }, save_path)
            min_val_loss = val_loss
            print(f'  Saved best model (val_loss: {val_loss / len(valset):.4f})')
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.title(f'{args.architecture.capitalize()} {"+ FiLM" if args.use_film else ""}')
        plt.plot(np.array(range(1, epoch + 1)), np.array(train_losses), 'b', label='Train')
        plt.plot(np.array(range(1, epoch + 1)), np.array(val_losses), 'r', label='Validation')
        plt.legend()
        plt.ylabel('Loss')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if minutes > 30:
            hours += 1
        plt.xlabel('Epochs' + ' (' + str(int(hours)) + ' hours)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/plots/training_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print('Finished training')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network variants')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='concatenated',
                        choices=['concatenated', 'dual', 'separate', 'broadcast'],
                        help='Network architecture variant')
    parser.add_argument('--use_film', action='store_true',
                        help='Use FiLM conditioning on gantry angle')
    
    # Training parameters
    parser.add_argument('--im_dir', type=str, default='/srv/shared/SPARE/MC_V_P1_NS_01',
                        help='Image directory')
    parser.add_argument('--im_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--int_steps', type=int, default=10,
                        help='Integration steps for diffeomorphic warp')
    
    args = parser.parse_args()
    train_model(args)
