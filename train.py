import time
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utilities import network_unified, losses
import torch
from torch.utils.data import Dataset
import torch.optim as optim


class SupervisedDataset(Dataset):
    """Dataset with DVF ground truth"""
    def __init__(self, im_dir, im_size):
        self.im_dir = im_dir
        self.im_size = im_size

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('s.npy')])

    def __getitem__(self, idx):
        slice_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('s.npy')])
        target_file = slice_list[idx]
        
        # Load and normalize target projections
        target_slice = np.load(os.path.join(self.im_dir, target_file))
        target_c = self._normalize(target_slice[0])
        target_s = self._normalize(target_slice[1])
        
        # Load target DVF
        vol_num = target_file[:2]
        target_dvf = np.load(os.path.join(self.im_dir, f'sub_DVF_{vol_num}_mha.npy'))
        
        # Load and normalize source projections
        source_file = '06_' + target_file[3:]
        source_slice = np.load(os.path.join(self.im_dir, source_file))
        source_c = self._normalize(source_slice[0])
        source_s = self._normalize(source_slice[1])
        
        # Load source volume
        source_vol = np.load(os.path.join(self.im_dir, 'sub_Abdomen_mha.npy'))
        
        return {
            'target_c': torch.from_numpy(target_c[None].astype(np.float32)),
            'target_s': torch.from_numpy(target_s[None].astype(np.float32)),
            'source_c': torch.from_numpy(source_c[None].astype(np.float32)),
            'source_s': torch.from_numpy(source_s[None].astype(np.float32)),
            'source_vol': torch.from_numpy(source_vol[None].astype(np.float32)),
            'target_flow': torch.from_numpy(target_dvf.transpose(3,0,1,2).astype(np.float32))
        }
    
    def _normalize(self, img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)


class UnsupervisedDataset(Dataset):
    """Dataset without DVF ground truth"""
    def __init__(self, im_dir, im_size):
        self.im_dir = im_dir
        self.im_size = im_size

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('s.npy')])

    def __getitem__(self, idx):
        slice_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('s.npy')])
        target_file = slice_list[idx]
        
        # Load and normalize target projections
        target_slice = np.load(os.path.join(self.im_dir, target_file))
        target_c = self._normalize(target_slice[0])
        target_s = self._normalize(target_slice[1])
        
        # Load and normalize source projections
        source_file = '06_' + target_file[3:]
        source_slice = np.load(os.path.join(self.im_dir, source_file))
        source_c = self._normalize(source_slice[0])
        source_s = self._normalize(source_slice[1])
        
        # Load source volume
        source_vol = np.load(os.path.join(self.im_dir, 'sub_Abdomen_mha.npy'))
        
        return {
            'target_c': torch.from_numpy(target_c[None].astype(np.float32)),
            'target_s': torch.from_numpy(target_s[None].astype(np.float32)),
            'source_c': torch.from_numpy(source_c[None].astype(np.float32)),
            'source_s': torch.from_numpy(source_s[None].astype(np.float32)),
            'source_vol': torch.from_numpy(source_vol[None].astype(np.float32))
        }
    
    def _normalize(self, img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)


def train_epoch(model, dataloader, optimizer, criterion, device, supervised):
    model.train()
    total_loss = 0.0
    
    for data in dataloader:
        target_c = data['target_c'].to(device)
        target_s = data['target_s'].to(device)
        source_c = data['source_c'].to(device)
        source_s = data['source_s'].to(device)
        source_vol = data['source_vol'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_source, pred_flow = model(source_c, source_s, target_c, target_s, source_vol)
        
        # Compute loss
        if supervised:
            target_flow = data['target_flow'].to(device)
            loss = criterion.loss(target_flow, pred_flow, source_vol)
        else:
            # Unsupervised: image similarity + regularization
            loss = criterion.loss(target_c, target_s, y_source, pred_flow)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device, supervised):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            target_c = data['target_c'].to(device)
            target_s = data['target_s'].to(device)
            source_c = data['source_c'].to(device)
            source_s = data['source_s'].to(device)
            source_vol = data['source_vol'].to(device)
            
            # Forward pass
            y_source, pred_flow = model(source_c, source_s, target_c, target_s, source_vol)
            
            # Compute loss
            if supervised:
                target_flow = data['target_flow'].to(device)
                loss = criterion.loss(target_flow, pred_flow, source_vol)
            else:
                loss = criterion.loss(target_c, target_s, y_source, pred_flow)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Projection-to-Volume Registration Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/combat/train',
                        help='Directory containing training data')
    parser.add_argument('--im_size', type=int, default=128,
                        help='Image size (cubic volumes)')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='original_mri',
                        choices=['original_mri', 'simple_3d', 'dual_stream_2d', 'hybrid'],
                        help='Network architecture for ablation study')
    parser.add_argument('--int_steps', type=int, default=7,
                        help='Number of diffeomorphic integration steps (0=none)')
    parser.add_argument('--skip_connections', action='store_true',
                        help='Use skip connections in decoder')
    
    # Training arguments
    parser.add_argument('--supervised', action='store_true',
                        help='Use supervised learning (requires DVF ground truth)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Fraction of data for training')
    
    # Output arguments
    parser.add_argument('--output_name', type=str, default=None,
                        help='Name for output files (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output_name is None:
        mode = 'sup' if args.supervised else 'unsup'
        skip = 'skip' if args.skip_connections else 'noskip'
        args.output_name = f'{args.architecture}_{mode}_{skip}_int{args.int_steps}'
    
    print(f'\n{"="*60}')
    print(f'Training Configuration:')
    print(f'  Architecture: {args.architecture}')
    print(f'  Mode: {"Supervised" if args.supervised else "Unsupervised"}')
    print(f'  Skip connections: {args.skip_connections}')
    print(f'  Integration steps: {args.int_steps}')
    print(f'{"="*60}\n')
    
    # Setup dataset
    if args.supervised:
        dataset = SupervisedDataset(args.data_dir, args.im_size)
    else:
        dataset = UnsupervisedDataset(args.data_dir, args.im_size)
    
    # Train/val split
    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    
    # Setup model
    model = network_unified.Proj2VolRegistration(
        im_size=args.im_size,
        architecture=args.architecture,
        int_steps=args.int_steps,
        skip_connections=args.skip_connections
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup loss and optimizer
    if args.supervised:
        criterion = losses.flow_mask()
    else:
        criterion = losses.UnsupervisedLoss()  # Define this in your losses module
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f'Training on {device}...\n')
    tic = time.time()
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, trainloader, optimizer, criterion, device, args.supervised)
        
        # Validate
        val_loss = validate_epoch(model, valloader, criterion, device, args.supervised)
        
        # Timing
        toc = time.time()
        time_elapsed = (toc - tic) / 3600
        hours = int(time_elapsed)
        minutes = int((time_elapsed - hours) * 60)
        
        print(f'Epoch: {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f} | '
              f'time: {hours}h {minutes}m')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < min_val_loss:
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), f'weights/{args.output_name}.pth')
            min_val_loss = val_loss
            print(f'  -> Saved best model (val loss: {val_loss:.4f})')
        
        # Plot training progress
        if epoch % 5 == 0 or epoch == args.epochs:
            os.makedirs('plots', exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.title(f'{args.architecture} - {"Supervised" if args.supervised else "Unsupervised"}')
            plt.plot(range(1, epoch + 1), train_losses, 'b-', label='Train')
            plt.plot(range(1, epoch + 1), val_losses, 'r-', label='Validation')
            plt.legend()
            plt.ylabel('Loss')
            plt.xlabel(f'Epochs ({hours}h {minutes}m)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'plots/{args.output_name}.png', dpi=150)
            plt.close()
    
    print('\nFinished training')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
