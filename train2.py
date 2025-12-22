import time
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utilities import networks2
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import argparse


class SupervisedDataset(Dataset):
    def __init__(self, im_dir=None, im_size=None):
        self.im_dir = im_dir
        self.im_size = im_size

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

        # Find target volume
        vol_name = os.path.join(self.im_dir, format('subCT_' + vol_num + '_mha.npy'))
        target_vol = np.load(vol_name)
        target_vol = (target_vol - np.min(target_vol)) / (np.max(target_vol) - np.min(target_vol))

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

        # Reshape data
        source_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        source_projections[0, :, :] = np.asarray(source_proj)
        target_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        target_projections[0, :, :] = np.asarray(target_proj)

        source_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_volumes[0, :, :, :] = np.asarray(source_vol)
        
        target_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_volumes[0, :, :, :] = np.asarray(target_vol)
        
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
            'target_volumes': torch.from_numpy(target_volumes),
            'source_abdomen': torch.from_numpy(source_abdomen),
            'target_flow': torch.from_numpy(target_flow)
        }
        
        return data


def train_model(args):
    """Train a specific network variant with joint training"""
    
    # Create output directory
    output_dir = f'outputs/{args.architecture}'
    if args.skip_connections:
        output_dir += '_skip'
    if args.supervised:
        output_dir += '_supervised'
    output_dir += f'_lambda{args.image_loss_weight}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/weights', exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Setup dataset
    dataset = SupervisedDataset(
        im_dir=args.im_dir,
        im_size=args.im_size
    )
    
    split = [int(len(dataset) * 0.9), int(len(dataset) * 0.1)]
    trainset, valset = torch.utils.data.dataset.random_split(dataset, split)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)
    
    # Setup model based on architecture
    if args.architecture == 'single_encoder':
        from utilities.networks import SingleEncoderDualDecoder
        model = SingleEncoderDualDecoder(
            im_size=args.im_size,
            int_steps=args.int_steps,
            skip_connections=args.skip_connections
        )
    elif args.architecture == 'dual_encoder':
        from utilities.networks import DualEncoderDualDecoder
        model = DualEncoderDualDecoder(
            im_size=args.im_size,
            int_steps=args.int_steps,
            skip_connections=args.skip_connections
        )
    elif args.architecture == 'original':
        from utilities.networks import OriginalModel
        model = OriginalModel(
            im_size=args.im_size,
            int_steps=args.int_steps
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup losses
    L2_loss = torch.nn.MSELoss()  # For supervised motion (flow)
    L1_loss = torch.nn.L1Loss()   # For unsupervised motion (volume) and image (volume)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Determine if we need dual decoder training
    is_dual_decoder = args.architecture in ['single_encoder', 'dual_encoder']
    
    print(f'\n{"="*60}')
    if is_dual_decoder:
        if args.supervised:
            print(f'Joint Training (Supervised): Motion (L2 on flow) + Image (L1, λ={args.image_loss_weight})')
        else:
            print(f'Joint Training (Unsupervised): Motion (L1 on volume) + Image (L1, λ={args.image_loss_weight})')
    else:
        if args.supervised:
            print(f'Training: Motion only (L2 on flow - supervised)')
        else:
            print(f'Training: Motion only (L1 on volume - unsupervised)')
    print(f'Architecture: {args.architecture}')
    print(f'Skip connections: {args.skip_connections}')
    print(f'{"="*60}\n')
    
    print(f'Training on {device}...')
    tic = time.time()
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_motion_losses, train_image_losses = [], []
    val_motion_losses, val_image_losses = [], []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_motion_loss = 0.0
        train_image_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            target_proj = data['target_projections'].to(device)
            source_vol = data['source_volumes'].to(device)
            target_vol = data['target_volumes'].to(device)
            target_flow = data['target_flow'].to(device)
            
            angle = data['angle'].to(device) if args.use_film else None
            
            optimizer.zero_grad()
            
            # Compute losses for both motion and image decoders
            total_loss = 0.0
            
            # Motion decoder loss
            y_source_motion, predict_flow_motion = model.forward(target_proj, source_vol, mode='motion')
            
            if args.supervised:
                # Supervised: L2 loss on flow field
                motion_loss = L2_loss(target_flow, predict_flow_motion)
            else:
                # Unsupervised: L1 loss on warped volume vs target volume
                motion_loss = L1_loss(y_source_motion, target_vol)
            
            total_loss += motion_loss
            train_motion_loss += motion_loss.item()
            
            # Image decoder loss (L1) - only for dual decoder architectures
            if is_dual_decoder:
                y_source_image, predict_flow_image = model.forward(target_proj, source_vol, mode='image')
                image_loss = L1_loss(y_source_image, target_vol)
                total_loss += args.image_loss_weight * image_loss
                train_image_loss += image_loss.item()
            
            train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
        
        # Validation
        val_loss = 0.0
        val_motion_loss = 0.0
        val_image_loss = 0.0
        model.eval()
        with torch.no_grad():
            for j, valdata in enumerate(valloader, 0):
                target_proj = valdata['target_projections'].to(device)
                source_vol = valdata['source_volumes'].to(device)
                target_vol = valdata['target_volumes'].to(device)
                target_flow = valdata['target_flow'].to(device)
                
                total_loss = 0.0
                
                # Motion decoder validation
                y_source_motion, predict_flow_motion = model.forward(target_proj, source_vol, mode='motion')
                
                if args.supervised:
                    # Supervised: L2 loss on flow field
                    motion_loss = L2_loss(target_flow, predict_flow_motion)
                else:
                    # Unsupervised: L1 loss on warped volume
                    motion_loss = L1_loss(y_source_motion, target_vol)
                
                total_loss += motion_loss
                val_motion_loss += motion_loss.item()
                
                # Image decoder validation
                if is_dual_decoder:
                    y_source_image, predict_flow_image = model.forward(target_proj, source_vol, mode='image')
                    image_loss = L1_loss(y_source_image, target_vol)
                    total_loss += args.image_loss_weight * image_loss
                    val_image_loss += image_loss.item()
                
                val_loss += total_loss.item()
        
        # Compute time
        toc = time.time()
        time_elapsed = (toc - tic) / 3600
        hours = np.floor(time_elapsed)
        minutes = (time_elapsed - hours) * 60
        
        # Normalize losses
        avg_train_loss = train_loss / len(trainset)
        avg_val_loss = val_loss / len(valset)
        avg_train_motion = train_motion_loss / len(trainset)
        avg_val_motion = val_motion_loss / len(valset)
        
        if is_dual_decoder:
            avg_train_image = train_image_loss / len(trainset)
            avg_val_image = val_image_loss / len(valset)
            print('Epoch: %d | train: %.4f (motion: %.4f, image: %.4f) | val: %.4f (motion: %.4f, image: %.4f) | time: %dh %dm' %
                  (epoch, avg_train_loss, avg_train_motion, avg_train_image, 
                   avg_val_loss, avg_val_motion, avg_val_image, hours, minutes))
        else:
            print('Epoch: %d | train loss: %.4f | val loss: %.4f | time: %dh %dm' %
                  (epoch, avg_train_loss, avg_val_loss, hours, minutes))
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_motion_losses.append(avg_train_motion)
        val_motion_losses.append(avg_val_motion)
        if is_dual_decoder:
            train_image_losses.append(avg_train_image)
            val_image_losses.append(avg_val_image)
        
        # Save best model
        if val_loss < min_val_loss:
            save_path = f'{output_dir}/weights/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_motion_loss': avg_train_motion,
                'val_motion_loss': avg_val_motion,
                'train_image_loss': avg_train_image if is_dual_decoder else None,
                'val_image_loss': avg_val_image if is_dual_decoder else None,
                'args': vars(args)
            }, save_path)
            min_val_loss = val_loss
            print(f'  Saved best model (val_loss: {avg_val_loss:.4f})')
        
        # Plot training curves
        epochs_range = np.array(range(1, epoch + 1))
        
        # Total loss plot
        plt.figure(figsize=(10, 6))
        title = f'{args.architecture.replace("_", " ").title()}'
        if args.skip_connections:
            title += ' + Skip'
        if is_dual_decoder:
            title += f' (λ={args.image_loss_weight})'
        plt.title(title)
        plt.plot(epochs_range, np.array(train_losses), 'b-', label='Train Total', linewidth=2)
        plt.plot(epochs_range, np.array(val_losses), 'r-', label='Val Total', linewidth=2)
        if is_dual_decoder:
            plt.plot(epochs_range, np.array(train_motion_losses), 'b--', label='Train Motion', alpha=0.7)
            plt.plot(epochs_range, np.array(val_motion_losses), 'r--', label='Val Motion', alpha=0.7)
            plt.plot(epochs_range, np.array(train_image_losses), 'b:', label='Train Image', alpha=0.7)
            plt.plot(epochs_range, np.array(val_image_losses), 'r:', label='Val Image', alpha=0.7)
        plt.legend()
        plt.ylabel('Loss')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if minutes > 30:
            hours += 1
        plt.xlabel('Epochs' + ' (' + str(int(hours)) + ' hours)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/plots/training_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print('\nFinished training')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network variants with joint training')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='single_encoder',
                        choices=['single_encoder', 'dual_encoder', 'original'],
                        help='Network architecture variant')
    parser.add_argument('--skip_connections', action='store_true',
                        help='Use skip connections in dual decoder architectures')
    parser.add_argument('--supervised', action='store_true',
                        help='Use supervised learning (L2 on flow). Default is unsupervised (L1 on volume)')
    
    # Loss parameters
    parser.add_argument('--image_loss_weight', type=float, default=10.0,
                        help='Weight for image reconstruction loss (L1)')
    
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
    parser.add_argument('--int_steps', type=int, default=7,
                        help='Integration steps for diffeomorphic warp')
    
    args = parser.parse_args()
    train_model(args)
