# Joint Motion Estimation and Image Synthesis

This codebase implements a joint motion estimation-image synthesis learning scheme for Voxelmap.

## Architecture Variants

### 1. Original
- Single encoder-decoder predicting motion fields
- Baseline architecture for comparison

### 2. Single Encoder, Dual Decoder
- Shared 3D encoder for projection + source volume
- Separate decoders for motion and image synthesis
- Motion decoder trained first, then frozen during image training
- Image decoder uses warped encoder features at all levels
- Optional U-Net skip connections between encoder and image decoder

### 3. Dual Encoder, Dual Decoder
- Independent encoders for motion and image pathways
- Separate decoders for each task
- Sequential training: motion network → image network
- Image encoder features warped by motion predictions at all levels
- Optional skip connections (concatenates motion features with warped image features for conditioning)

All variants support multi-resolution processing with feature warping at each decoder level.

## File Structure

```
utilities/networks.py    # All architecture implementations
train.py                # Training script for any variant
train_all.sh            # Batch training script
```

## Training

### Single variant:
```bash
python train.py \
    --model_variant single_encoder \
    --skip_connections False \
    --image_loss_type l1 \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --im_dir /path/to/data
```

### With perceptual loss:
```bash
python train.py \
    --model_variant dual_encoder \
    --image_loss_type perceptual \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4
```

## Output Structure

```
weights/
├── {variant}_motion.pth        # Motion pathway (stage 1)
├── {variant}_image.pth         # Image pathway (stage 2)
└── {variant}_final.pth         # Complete model

plots/
├── {variant}_motion.png        # Motion training curves
└── {variant}_image.png         # Image training curves

logs/
├── original.log
├── single_encoder_l1.log
├── single_encoder_perceptual.log
├── dual_encoder_l1.log
└── dual_encoder_perceptual.log
```

## Data Requirements

Expected files in `im_dir`:
- `XX_bin.npy`: Target projections (XX=phase)
- `DVF_XX_mha.npy`: Target deformation fields (3D, 3 channels)
- `subCT_XX_mha.npy`: Target volumes
- `subCT_06_mha.npy`: Source volume (reference)

All data normalized to [0, 1].

## Key Parameters

**Model:**
- `--model_variant`: `single_encoder`, `dual_encoder`, `original`
- `--skip_connections`: Enable skip connections (default: False)
- `--image_loss_type`: `l1`, `perceptual`

**Training:**
- `--motion_epochs`: Epochs for motion training (default: 100)
- `--image_epochs`: Epochs for image training (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 8)

**Network:**
- `--int_steps`: Flow integration steps (default: 7)
- `--im_size`: Resolution (default: 128)
- `--num_levels`: Multi-resolution levels (default: 4)

## Training Details

### Two-Stage Training
For dual decoder variants:

**Stage 1 - Motion:**
- Train motion encoder/decoder with multi-level MSE on flow fields
- Level weights: [0.25, 0.5, 0.75, 1.0] (coarse to fine)
- Best model saved as `{variant}_motion.pth`

**Stage 2 - Image:**
- Freeze motion pathway
- Train image encoder/decoder with L1 or perceptual loss
- Features warped at each level using predicted motion
- Best model saved as `{variant}_image.pth`

Final complete model: `{variant}_final.pth`

### Loss Functions

**Motion Loss:** Multi-level MSE between predicted and ground truth flow

**Image Loss:**
- **L1**: Pixel-wise loss between predicted and target volumes
- **Perceptual**: VGG16-based features from 3 orthogonal slice planes

### Multi-Resolution Processing

Flow predictions at 4 levels (for im_size=128):
- Level 0: 16×16×16
- Level 1: 32×32×32
- Level 2: 64×64×64
- Level 3: 128×128×128

Features warped at each level before upsampling.

## Model Loading

```python
from utilities import network_variants
import torch

model = network_variants.SingleEncoderDualDecoder(
    im_size=128, int_steps=7, num_levels=4, skip_connections=False
)
model.load_state_dict(torch.load('weights/single_encoder_noskip_l1_final.pth'))
model.eval()

# Motion prediction
with torch.no_grad():
    warped_vol, flow, multi_flows = model(target_proj, source_vol, mode='motion')

# Image synthesis
with torch.no_grad():
    synth_vol, flow, multi_flows = model(target_proj, source_vol, mode='image')
```

## Implementation Notes

### Skip Connections
- **Single Encoder**: Fuses warped encoder features with motion decoder features
- **Dual Encoder**: Concatenates motion decoder features with warped image encoder features
  - Semantics: motion features provide conditioning for image synthesis
  - Recommend experimentation to validate effectiveness

### Feature Warping
- Features warped at decoder levels before upsampling
- Maintains spatial correspondence during reconstruction
- Applied at all resolution levels for maximum motion guidance

## Installation

```bash
pip install torch torchvision numpy matplotlib

# Required utilities (from your codebase):
# - utilities.layers: VecInt, SpatialTransformer
# - utilities.losses: MSE
# - utilities.modelio: LoadableModel, store_config_args
```
