# Joint Motion Estimation and Image Synthesis

This codebase implements joint training of motion estimation and image synthesis networks Volxelmap.

## Architecture Variants

### 1. Original (Baseline)
- Single encoder-decoder architecture
- Predicts motion fields only
- Uses L2 loss on flow predictions

### 2. Single Encoder, Dual Decoder
- Shared 3D encoder for projection + source volume
- Separate decoders for motion and image synthesis
- Motion decoder features concatenated with encoder features for image decoder
- Optional U-Net skip connections between encoder and decoders

### 3. Dual Encoder, Dual Decoder
- Independent encoders for motion and image pathways
- Separate decoders for each task
- Motion decoder features concatenated with image encoder features
- Optional skip connections for additional feature fusion

All dual decoder variants use **joint training** with:
- **Motion loss**: L2 (MSE) on predicted flow fields
- **Image loss**: L1 on reconstructed volumes, weighted by λ (default: 10.0)

## File Structure

```
utilities/networks.py    # Architecture implementations
train.py                # Training script with joint training
train_all.sh            # Batch training for all variants
```

## Quick Start

### Train single variant:
```bash
python train.py \
    --architecture single_encoder \
    --skip_connections \
    --image_loss_weight 10.0 \
    --epochs 100 \
    --lr 1e-5 \
    --im_dir /path/to/data
```

### Train all variants:
```bash
bash train_all.sh
```

This trains:
- `original` (baseline)
- `single_encoder` (without skip connections)
- `single_encoder --skip_connections`
- `dual_encoder` (without skip connections)
- `dual_encoder --skip_connections`

## Output Structure

```
outputs/
├── original_lambda10.0/
│   ├── weights/
│   │   └── best_model.pth
│   └── plots/
│       └── training_curve.png
├── single_encoder_lambda10.0/
│   ├── weights/
│   │   └── best_model.pth
│   └── plots/
│       └── training_curve.png
├── single_encoder_skip_lambda10.0/
│   └── ...
├── dual_encoder_lambda10.0/
│   └── ...
└── dual_encoder_skip_lambda10.0/
    └── ...
```

## Training Curves

For dual decoder variants, plots show:
- **Total loss** (solid lines): Combined motion + image loss
- **Motion loss** (dashed lines): L2 flow prediction component
- **Image loss** (dotted lines): L1 volume reconstruction component

## Data Requirements

Expected files in `im_dir`:
- `XX_YY_bin.npy`: Projections (XX=phase, YY=projection number)
- `DVF_XX_mha.npy`: Ground truth deformation fields (shape: [H,W,D,3])
- `subCT_XX_mha.npy`: Target volumes
- `subCT_06_mha.npy`: Source volume (reference phase)
- `sub_Abdomen_mha.npy`: Abdomen mask

All data normalized to [0, 1].

## Key Parameters

**Model:**
- `--architecture`: `single_encoder`, `dual_encoder`, `original`
- `--skip_connections`: Enable U-Net skip connections (dual decoder only)
- `--image_loss_weight`: Weight λ for image loss (default: 10.0)

**Training:**
- `--epochs`: Training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-5)
- `--batch_size`: Batch size (default: 8)

**Network:**
- `--int_steps`: Flow integration steps (default: 7)
- `--im_size`: Volume resolution (default: 128)

## Joint Training Details

Both motion and image decoders train simultaneously:

1. **Forward passes**: 
   - Motion mode: encoder → motion decoder → L2 loss
   - Image mode: encoder → motion decoder (frozen) → image decoder → L1 loss

2. **Combined loss**: `L_total = L2(flow) + λ * L1(volume)`

3. **Gradient updates**: All trainable parameters updated together

4. **Feature passing**: Motion decoder features concatenated with encoder/image features at each resolution level

## Loss Functions

**Motion Loss (L2):**
```python
motion_loss = MSE(predicted_flow, target_flow)
```

**Image Loss (L1):**
```python
image_loss = L1(reconstructed_volume, target_volume)
```

**Total Loss:**
```python
total_loss = motion_loss + λ * image_loss
```

## Model Loading & Inference

```python
from utilities.networks import SingleEncoderDualDecoder
import torch

# Load model
model = SingleEncoderDualDecoder(
    im_size=128, 
    skip_connections=True
)
checkpoint = torch.load('outputs/single_encoder_skip_lambda10.0/weights/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Motion prediction
with torch.no_grad():
    warped_vol, flow = model(target_proj, source_vol, mode='motion')

# Image synthesis
with torch.no_grad():
    synth_vol, flow = model(target_proj, source_vol, mode='image')
```

## Architecture Details

### Skip Connections

**Single Encoder:**
- Concatenates encoder features + motion decoder features for image decoder input
- Applied at each resolution level during upsampling

**Dual Encoder:**
- Concatenates image encoder features + motion decoder features
- Allows motion features to condition image synthesis

### Feature Concatenation

Motion decoder features pass to image decoder at matching resolution levels:
- No warping required (features in same space)
- Concatenation before each upsampling block
- Channel dimensions adjusted automatically

### Flow Integration

All architectures use diffeomorphic flow integration:
- `int_steps=0`: No integration (velocity field output)
- `int_steps>0`: Scaling and squaring for diffeomorphic deformation

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib

# Required from your codebase:
# - utilities.layers: VecInt, SpatialTransformer
# - utilities.modelio: LoadableModel, store_config_args
```

## Hyperparameter Tuning

Key hyperparameters to experiment with:

1. **Image loss weight (λ)**: Balance motion vs image objectives
   - Higher λ: prioritize image quality
   - Lower λ: prioritize motion accuracy
   
2. **Skip connections**: May improve feature propagation
   - Test both enabled/disabled for your data

3. **Learning rate**: Start with 1e-5, adjust based on convergence