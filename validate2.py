import torch
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import argparse


class TestDataset(Dataset):

    def __init__(self, im_dir=None, im_size=None):
        self.im_dir = im_dir
        self.im_size = im_size

    def __len__(self):
        return len([n for n in os.listdir(self.im_dir) if n.startswith('01') & n.endswith('bin.npy')])

    def __getitem__(self, idx):
        # Find target projection
        proj_list = sorted([n for n in os.listdir(self.im_dir) if n.startswith('01') & n.endswith('bin.npy')])
        target_file = proj_list[idx]
        proj_name = os.path.join(self.im_dir, format(target_file))
        target_proj = np.load(proj_name)
        target_proj = (target_proj - np.min(target_proj)) / (np.max(target_proj) - np.min(target_proj))

        # Find target volume
        vol_num = target_file[:2]
        vol_name = os.path.join(self.im_dir, format('subCT_01_mha.npy'))
        target_vol = np.load(vol_name)
        target_vol = (target_vol - np.min(target_vol)) / (np.max(target_vol) - np.min(target_vol))

        # Find target DVF
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

        # Reshape data to correct format
        source_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        source_projections[0, :, :] = np.asarray(source_proj)
        target_projections = np.zeros((1, self.im_size, self.im_size), dtype=np.float32)
        target_projections[0, :, :] = np.asarray(target_proj)

        source_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_volumes[0, :, :, :] = np.asarray(source_vol)
        source_abdomen = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        source_abdomen[0, :, :, :] = np.asarray(source_hull)
        target_volumes = np.zeros((1, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_volumes[0, :, :, :] = np.asarray(target_vol)

        target_flow = np.zeros((3, self.im_size, self.im_size, self.im_size), dtype=np.float32)
        target_flow[0, :, :, :] = target_dvf[:, :, :, 0]
        target_flow[1, :, :, :] = target_dvf[:, :, :, 1]
        target_flow[2, :, :, :] = target_dvf[:, :, :, 2]

        data = {'source_projections': torch.from_numpy(source_projections),
                'target_projections': torch.from_numpy(target_projections),
                'source_volumes': torch.from_numpy(source_volumes),
                'source_abdomen': torch.from_numpy(source_abdomen),
                'target_volumes': torch.from_numpy(target_volumes),
                'target_flow': torch.from_numpy(target_flow)}

        return data


def validate_model(args):
    """Validate a trained model"""
    
    # Generate test loader
    testset = TestDataset(im_dir=args.test_dir, im_size=args.im_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)
    
    # Load model based on architecture
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
    
    # Load weights
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'\n{"="*60}')
    print(f'Validating model: {args.architecture}')
    print(f'Skip connections: {args.skip_connections}')
    print(f'Weights: {args.weights_path}')
    print(f'Test data: {args.test_dir}')
    print(f'Decoder mode: {args.decoder_mode}')
    print(f'{"="*60}\n')
    
    # Get first sample for visualization
    data = next(iter(testloader))
    source_proj = data['source_projections'].to(device)
    target_proj = data['target_projections'].to(device)
    source_vol = data['source_volumes'].to(device)
    source_abdomen = data['source_abdomen'].to(device)
    target_vol = data['target_volumes'].to(device)
    target_flow = data['target_flow'].to(device)
    
    # Prediction
    print('Testing...')
    with torch.no_grad():
        # Use specified decoder mode
        predictions, flows = model.forward(target_proj, source_vol, mode=args.decoder_mode)
        
        # Timing test
        tic = time.time()
        model.forward(target_proj, source_vol, mode=args.decoder_mode)
        toc = time.time()
        print('Inference completed in %d milliseconds' % (1000*(toc - tic)))
    
    # Evaluation metrics
    RMSE = torch.sqrt(torch.mean(torch.square(target_flow - flows)))
    print('RMSE: %.2f' % (RMSE.item()))
    MAE = torch.mean(torch.abs(target_flow - flows))
    print('MAE: %.2f' % (MAE.item()))
    
    # 3D distance error (if available)
    try:
        from voxelmap import losses
        dist3d_mask = losses.dist3d_mask()
        DIST = dist3d_mask.loss(target_flow, flows, source_abdomen)
        print('3D error: %.2f' % (DIST.item()))
    except ImportError:
        print('3D distance metric not available (voxelmap.losses not found)')
    
    # Volume reconstruction metrics
    vol_MAE = torch.mean(torch.abs(target_vol - predictions))
    print('Volume MAE: %.4f' % (vol_MAE.item()))
    vol_RMSE = torch.sqrt(torch.mean(torch.square(target_vol - predictions)))
    print('Volume RMSE: %.4f' % (vol_RMSE.item()))
    
    # Visualization
    pred_flow = flows.cpu().detach().numpy()
    pred_flow = np.asarray(pred_flow[0][:][:][:][:], dtype=np.float32)
    
    target_flow_np = target_flow.cpu().detach().numpy()
    target_flow_np = np.asarray(target_flow_np[0][:][:][:][:], dtype=np.float32)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot DVF comparison
    f = plt.figure(figsize=(12, 8))
    
    # Predicted flow
    f.add_subplot(2, 3, 1)
    lr = pred_flow[1][:][:][:]
    slice_pred = np.squeeze(lr[:, :, round(args.im_size / 2)])
    slice_pred = np.transpose(slice_pred)
    plt.imshow(slice_pred, interpolation='none')
    plt.title('Predicted LR')
    plt.axis('off')
    plt.colorbar()
    
    f.add_subplot(2, 3, 2)
    si = pred_flow[0][:][:][:]
    slice_pred = np.squeeze(si[:, :, round(args.im_size / 2)])
    slice_pred = np.transpose(slice_pred)
    plt.imshow(slice_pred, interpolation='none')
    plt.title('Predicted SI')
    plt.axis('off')
    plt.colorbar()
    
    f.add_subplot(2, 3, 3)
    ap = pred_flow[2][:][:][:]
    slice_pred = np.squeeze(ap[:, :, round(args.im_size / 2)])
    slice_pred = np.transpose(slice_pred)
    plt.imshow(slice_pred, interpolation='none')
    plt.title('Predicted AP')
    plt.axis('off')
    plt.colorbar()
    
    # Target flow
    f.add_subplot(2, 3, 4)
    lr = target_flow_np[1][:][:][:]
    slice_target = np.squeeze(lr[:, :, round(args.im_size / 2)])
    slice_target = np.transpose(slice_target)
    plt.imshow(slice_target, interpolation='none')
    plt.title('Target LR')
    plt.axis('off')
    plt.colorbar()
    
    f.add_subplot(2, 3, 5)
    si = target_flow_np[0][:][:][:]
    slice_target = np.squeeze(si[:, :, round(args.im_size / 2)])
    slice_target = np.transpose(slice_target)
    plt.imshow(slice_target, interpolation='none')
    plt.title('Target SI')
    plt.axis('off')
    plt.colorbar()
    
    f.add_subplot(2, 3, 6)
    ap = target_flow_np[2][:][:][:]
    slice_target = np.squeeze(ap[:, :, round(args.im_size / 2)])
    slice_target = np.transpose(slice_target)
    plt.imshow(slice_target, interpolation='none')
    plt.title('Target AP')
    plt.axis('off')
    plt.colorbar()
    
    plt.suptitle(f'{args.architecture} - {args.decoder_mode} decoder')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.architecture}_{args.decoder_mode}_dvf.png'), dpi=150)
    plt.close()
    
    # Plot volume comparison
    pred_vol = predictions.cpu().detach().numpy()[0, 0]
    target_vol_np = target_vol.cpu().detach().numpy()[0, 0]
    
    f = plt.figure(figsize=(15, 5))
    
    f.add_subplot(1, 3, 1)
    plt.imshow(pred_vol[:, :, args.im_size // 2].T, cmap='gray', interpolation='none')
    plt.title('Predicted Volume')
    plt.axis('off')
    plt.colorbar()
    
    f.add_subplot(1, 3, 2)
    plt.imshow(target_vol_np[:, :, args.im_size // 2].T, cmap='gray', interpolation='none')
    plt.title('Target Volume')
    plt.axis('off')
    plt.colorbar()
    
    f.add_subplot(1, 3, 3)
    diff = np.abs(pred_vol - target_vol_np)
    plt.imshow(diff[:, :, args.im_size // 2].T, cmap='hot', interpolation='none')
    plt.title('Absolute Difference')
    plt.axis('off')
    plt.colorbar()
    
    plt.suptitle(f'{args.architecture} - {args.decoder_mode} decoder - Volume Reconstruction')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.architecture}_{args.decoder_mode}_volume.png'), dpi=150)
    plt.close()
    
    print(f'\nVisualizations saved to: {args.output_dir}')
    print('Validation complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate trained network variants')
    
    # Model specification
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['single_encoder', 'dual_encoder', 'original'],
                        help='Network architecture variant')
    parser.add_argument('--skip_connections', action='store_true',
                        help='Model was trained with skip connections')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to model weights (.pth file)')
    
    # Decoder mode
    parser.add_argument('--decoder_mode', type=str, default='motion',
                        choices=['motion', 'image'],
                        help='Which decoder to use for validation')
    
    # Data parameters
    parser.add_argument('--test_dir', type=str, default='data/MC_V_P1_NS_01',
                        help='Test data directory')
    parser.add_argument('--im_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--int_steps', type=int, default=7,
                        help='Integration steps for diffeomorphic warp')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.decoder_mode == 'image' and args.architecture == 'original':
        raise ValueError("Original architecture only has motion decoder")
    
    validate_model(args)
