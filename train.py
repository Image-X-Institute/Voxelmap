import time
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from utilities import networks, losses
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
# Dataset
# ============================================================================

class SupervisedDataset(Dataset):
    def __init__(self, im_dir=None, im_size=None):
        self.im_dir = im_dir
        self.im_size = im_size

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])

    def __getitem__(self, idx):
        proj_list = sorted([n for n in os.listdir(self.im_dir) if n.endswith('_bin.npy')])
        target_file = proj_list[idx]
        proj_name = os.path.join(self.im_dir, target_file)
        target_proj = np.load(proj_name)
        target_proj = (target_proj - np.min(target_proj)) / (np.max(target_proj) - np.min(target_proj))

        vol_num = target_file[:2]
        dvf_name = os.path.join(self.im_dir, f'DVF_{vol_num}_mha.npy')
        target_dvf = np.load(dvf_name)

        vol_name = os.path.join(self.im_dir, 'subCT_06_mha.npy')
        source_vol = np.load(vol_name)
        source_vol = (source_vol - np.min(source_vol)) / (np.max(source_vol) - np.min(source_vol))

        target_vol_name = os.path.join(self.im_dir, f'subCT_{vol_num}_mha.npy')
        target_vol = np.load(target_vol_name)
        target_vol = (target_vol - np.min(target_vol)) / (np.max(target_vol) - np.min(target_vol))

        target_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        target_projections[0, :, :] = np.asarray(target_proj)

        source_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_volumes[0, :, :, :] = np.asarray(source_vol)

        target_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_volumes[0, :, :, :] = np.asarray(target_vol)

        target_flow = np.zeros((3, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_flow[0, :, :, :] = target_dvf[:, :, :, 0]
        target_flow[1, :, :, :] = target_dvf[:, :, :, 1]
        target_flow[2, :, :, :] = target_dvf[:, :, :, 2]

        data = {
            'target_projections': torch.from_numpy(target_projections),
            'source_volumes': torch.from_numpy(source_volumes),
            'target_volumes': torch.from_numpy(target_volumes),
            'target_flow': torch.from_numpy(target_flow)
        }

        return data


# ============================================================================
# Perceptual Loss
# ============================================================================

class PerceptualLoss(torch.nn.Module):
    """VGG-based perceptual loss for 3D volumes using 2D slices"""
    
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.device = device
    
    def extract_features(self, x):
        """Extract features from central slices in all three dimensions"""
        B, C, D, H, W = x.shape
        
        slice_z = x[:, :, D//2, :, :]
        slice_y = x[:, :, :, H//2, :]
        slice_x = x[:, :, :, :, W//2]
        
        slice_z = slice_z.repeat(1, 3, 1, 1)
        slice_y = slice_y.repeat(1, 3, 1, 1)
        slice_x = slice_x.repeat(1, 3, 1, 1)
        
        feat_z = self.vgg(slice_z)
        feat_y = self.vgg(slice_y)
        feat_x = self.vgg(slice_x)
        
        return feat_z, feat_y, feat_x
    
    def forward(self, pred, target):
        pred_feats = self.extract_features(pred)
        target_feats = self.extract_features(target)
        
        loss = 0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.mse_loss(pf, tf)
        
        return loss / 3


# ============================================================================
# Utility Functions
# ============================================================================

def downsample_flow(flow, scale_factor):
    """Downsample flow field and scale values accordingly"""
    downsampled = F.interpolate(flow, scale_factor=scale_factor, mode='trilinear', align_corners=True)
    downsampled = downsampled * scale_factor
    return downsampled


def save_model(model, path):
    """Save model state dict"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def plot_losses(train_losses, val_losses, epoch, hours, filename, title):
    """Plot and save training curves"""
    os.makedirs('plots', exist_ok=True)
    plt.figure()
    plt.title(title)
    plt.plot(np.array(range(1, epoch + 1)), np.array(train_losses), 'b')
    plt.plot(np.array(range(1, epoch + 1)), np.array(val_losses), 'r')
    plt.legend(['Train', 'Validation'])
    plt.ylabel('Loss')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(f'Epochs ({int(hours)} hours)')
    plt.savefig(f'plots/{filename}.png')
    plt.close()


# ============================================================================
# Training Functions
# ============================================================================

def train_motion(model, trainloader, valloader, device, num_epochs, lr, level_weights, args):
    """Train motion estimation pathway"""
    print('\n=== Training Motion Pathway ===')
    
    MSE = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    tic = time.time()
    
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        model.train()
        
        for data in trainloader:
            target_proj = data['target_projections'].to(device)
            source_vol = data['source_volumes'].to(device)
            target_flow = data['target_flow'].to(device)
            
            optimizer.zero_grad()
            
            _, _, flows_at_levels = model(target_proj, source_vol, mode='motion')
            
            loss = 0.0
            for level_idx, pred_flow in enumerate(flows_at_levels):
                scale_factor = pred_flow.shape[2] / target_flow.shape[2]
                target_flow_downsampled = downsample_flow(target_flow, scale_factor)
                level_loss = MSE(target_flow_downsampled, pred_flow)
                loss += level_weights[level_idx] * level_loss
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for valdata in valloader:
                target_proj = valdata['target_projections'].to(device)
                source_vol = valdata['source_volumes'].to(device)
                target_flow = valdata['target_flow'].to(device)
                
                _, _, flows_at_levels = model(target_proj, source_vol, mode='motion')
                
                loss = 0.0
                for level_idx, pred_flow in enumerate(flows_at_levels):
                    scale_factor = pred_flow.shape[2] / target_flow.shape[2]
                    target_flow_downsampled = downsample_flow(target_flow, scale_factor)
                    level_loss = MSE(target_flow_downsampled, pred_flow)
                    loss += level_weights[level_idx] * level_loss
                
                val_loss += loss.item()
        
        time_elapsed = (time.time() - tic) / 3600
        hours = int(np.ceil(time_elapsed))
        
        print(f'Epoch: {epoch} | train: {train_loss/len(trainloader.dataset):.4f} | '
              f'val: {val_loss/len(valloader.dataset):.4f} | time: {hours}h')
        
        train_losses.append(train_loss / len(trainloader.dataset))
        val_losses.append(val_loss / len(valloader.dataset))
        
        if val_loss < min_val_loss:
            save_model(model, f'weights/{args.exp_name}_motion.pth')
            min_val_loss = val_loss
        
        if epoch % 10 == 0:
            plot_losses(train_losses, val_losses, epoch, time_elapsed, 
                       f'{args.exp_name}_motion', f'{args.exp_name} - Motion')
    
    return model, train_losses, val_losses


def train_image(model, trainloader, valloader, device, loss_type, num_epochs, lr, args):
    """Train image synthesis pathway"""
    print(f'\n=== Training Image Pathway ({loss_type} loss) ===')
    
    # Freeze motion pathway
    if hasattr(model, 'motion_encoder'):
        for param in model.motion_encoder.parameters():
            param.requires_grad = False
        for param in model.motion_decoder.parameters():
            param.requires_grad = False
    else:
        for param in model.motion_decoder.parameters():
            param.requires_grad = False
    
    if loss_type == 'l1':
        image_loss_fn = torch.nn.L1Loss()
    elif loss_type == 'perceptual':
        image_loss_fn = PerceptualLoss(device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    min_val_loss = float('inf')
    train_losses, val_losses = [], []
    tic = time.time()
    
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        model.train()
        
        for data in trainloader:
            target_proj = data['target_projections'].to(device)
            source_vol = data['source_volumes'].to(device)
            target_vol = data['target_volumes'].to(device)
            
            optimizer.zero_grad()
            
            pred_vol, _, _ = model(target_proj, source_vol, mode='image')
            loss = image_loss_fn(pred_vol, target_vol)
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for valdata in valloader:
                target_proj = valdata['target_projections'].to(device)
                source_vol = valdata['source_volumes'].to(device)
                target_vol = valdata['target_volumes'].to(device)
                
                pred_vol, _, _ = model(target_proj, source_vol, mode='image')
                loss = image_loss_fn(pred_vol, target_vol)
                val_loss += loss.item()
        
        time_elapsed = (time.time() - tic) / 3600
        hours = int(np.ceil(time_elapsed))
        
        print(f'Epoch: {epoch} | train: {train_loss/len(trainloader.dataset):.4f} | '
              f'val: {val_loss/len(valloader.dataset):.4f} | time: {hours}h')
        
        train_losses.append(train_loss / len(trainloader.dataset))
        val_losses.append(val_loss / len(valloader.dataset))
        
        if val_loss < min_val_loss:
            save_model(model, f'weights/{args.exp_name}_image.pth')
            min_val_loss = val_loss
        
        if epoch % 10 == 0:
            plot_losses(train_losses, val_losses, epoch, time_elapsed,
                       f'{args.exp_name}_image', f'{args.exp_name} - Image ({loss_type})')
    
    return model, train_losses, val_losses


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    print(f'Configuration: {args.exp_name}')
    
    # Load dataset
    dataset = SupervisedDataset(im_dir=args.im_dir, im_size=args.im_size)
    split = [int(len(dataset) * 0.9), int(len(dataset) * 0.1)]
    trainset, valset = torch.utils.data.dataset.random_split(dataset, split)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    
    print(f'Train samples: {len(trainset)}, Val samples: {len(valset)}')
    
    # Initialize model
    if args.model_variant == 'single_encoder':
        model = networks.SingleEncoderDualDecoder(
            args.im_size, int_steps=args.int_steps, num_levels=args.num_levels, 
            skip_connections=args.skip_connections
        )
    elif args.model_variant == 'dual_encoder':
        model = networks.DualEncoderDualDecoder(
            args.im_size, int_steps=args.int_steps, num_levels=args.num_levels,
            skip_connections=args.skip_connections
        )
    elif args.model_variant == 'original':
        model = networks.OriginalModel(
            args.im_size, int_steps=args.int_steps, num_levels=args.num_levels
        )
    else:
        raise ValueError(f"Unknown model variant: {args.model_variant}")
    
    model.to(device)
    
    level_weights = [0.25, 0.5, 0.75, 1.0]
    
    # Training pipeline
    if args.model_variant == 'original':
        print('Training original model with flow supervision...')
        model, _, _ = train_motion(model, trainloader, valloader, device, 
                                   args.motion_epochs, args.lr, level_weights, args)
    else:
        print('Stage 1: Training motion pathway')
        model, _, _ = train_motion(model, trainloader, valloader, device,
                                   args.motion_epochs, args.lr, level_weights, args)
        
        print('Stage 2: Training image pathway')
        model, _, _ = train_image(model, trainloader, valloader, device,
                                  args.image_loss_type, args.image_epochs, args.lr, args)
        
        save_model(model, f'weights/{args.exp_name}_final.pth')
        print(f'\nFinal model saved to weights/{args.exp_name}_final.pth')
    
    print('\nTraining complete!')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 2D-to-3D network variants')
    
    # Model configuration
    parser.add_argument('--model_variant', type=str, default='single_encoder',
                       choices=['single_encoder', 'dual_encoder', 'original'],
                       help='Model architecture variant')
    parser.add_argument('--skip_connections', type=lambda x: x.lower() == 'true', default=False,
                       help='Use skip connections from motion to image decoder')
    parser.add_argument('--image_loss_type', type=str, default='l1',
                       choices=['l1', 'perceptual'],
                       help='Loss function for image synthesis training')
    
    # Training parameters
    parser.add_argument('--motion_epochs', type=int, default=100,
                       help='Number of epochs for motion training')
    parser.add_argument('--image_epochs', type=int, default=100,
                       help='Number of epochs for image training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    
    # Network parameters
    parser.add_argument('--im_size', type=int, default=128,
                       help='Image/volume size')
    parser.add_argument('--int_steps', type=int, default=7,
                       help='Number of flow integration steps')
    parser.add_argument('--num_levels', type=int, default=4,
                       help='Number of resolution levels')
    
    # Data
    parser.add_argument('--im_dir', type=str, default='/srv/shared/SPARE/MC_V_P1_NS_01',
                       help='Directory containing training data')
    
    args = parser.parse_args()
    
    # Generate experiment name
    skip_str = 'skip' if args.skip_connections else 'noskip'
    if args.model_variant == 'original':
        args.exp_name = 'original'
    else:
        args.exp_name = f'{args.model_variant}_{skip_str}_{args.image_loss_type}'
    
    # Create output directories
    os.makedirs('weights', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    main(args)
