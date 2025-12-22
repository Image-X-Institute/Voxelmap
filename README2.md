# Joint Motion Estimation and Image Synthesis

This codebase implements joint training of motion estimation and image synthesis networks for Voxelmap.

## Learning Modes

### Supervised Learning
- Motion module trained with **L2 loss on flow fields** (requires ground truth DVFs)
- Direct supervision on deformation field accuracy

### Unsupervised Learning (Default)
- Motion module trained with **L1 loss on warped volumes** (no DVF required)
- Learns motion by warping source volume to match target volume
- More practical for real clinical scenarios without ground truth motion

Both modes use **L1 loss** for image synthesis module.

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
- **Motion loss**: L2 (supervised) or L1 (unsupervised) 
- **Image loss**: L1 on reconstructed volumes, weighted by λ (default: 10.0)

## File Structure

```
utilities/networks.py    # Architecture implementations
train.py                # Training script with joint training
train_all.sh            # Batch training for all variants
```

## Quick Start

### Train single variant (unsupervised):
```bash
python train.py \
    --architecture single_encoder \
    --skip_connections \
    --image_loss_weight 10.0 \
    --epochs 100 \
    --lr 1e-5 \
    --im_dir /path/to/data
```

### Train with supervised learning:
```bash
python train.py \
    --architecture single_encoder \
    --supervised \
    --epochs 100 \
    --lr 1e-5 \
    --im_dir /path/to/data
```

### Train all variants:
```bash
bash train_all.sh
```

This trains 10 total variants:
- `original` (unsupervised + supervised)
- `single_encoder` × 2 skip settings × 2 learning modes = 4 variants
- `dual_encoder` × 2 skip settings × 2 learning modes = 4 variants

## Output Structure

```
outputs/
├── original_lambda10.0/
│   ├── weights/best_model.pth
│   └── plots/training_curve.png
├── original_supervised_lambda10.0/
│   └── ...
├── single_encoder_lambda10.0/
│   └── ...
├── single_encoder_supervised_lambda10.0/
│   └── ...
├── single_encoder_skip_lambda10.0/
│   └── ...
├── single_encoder_skip_supervised_lambda10.0/
│   └── ...
└── (similarly for dual_encoder variants)
```

## Training Curves

For dual decoder variants, plots show:
- **Total loss** (solid lines): Combined motion + image loss
- **Motion loss** (dashed lines): L2 flow prediction component
- **Image loss** (dotted lines): L1 volume reconstruction component

## Data Requirements

Expected files in `im_dir`:
- `XX_YY_bin.npy`: Projections (XX=phase, YY=projection number)
- `subCT_XX_mha.npy`: Target volumes (required for both supervised and unsupervised)
- `subCT_06_mha.npy`: Source volume (reference phase)
- `DVF_XX_mha.npy`: Ground truth deformation fields (shape: [H,W,D,3]) - **only required for supervised learning**
- `sub_Abdomen_mha.npy`: Abdomen mask

All data normalized to [0, 1].

## Key Parameters

**Model:**
- `--architecture`: `single_encoder`, `dual_encoder`, `original`
- `--skip_connections`: Enable U-Net skip connections (dual decoder only)
- `--supervised`: Use supervised learning (L2 on flow). Default is unsupervised (L1 on volume)
- `--image_loss_weight`: Weight λ for image loss (default: 10.0)

**Training:**
- `--epochs`: Training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-5)
- `--batch_size`: Batch size (default: 8)

**Network:**
- `--int_steps`: Flow integration steps (default: 10)
- `--im_size`: Volume resolution (default: 128)

## Joint Training Details

Both motion and image decoders train simultaneously:

1. **Forward passes**: 
   - Motion mode: encoder → motion decoder
     - Supervised: L2 loss on predicted flow vs ground truth flow
     - Unsupervised: L1 loss on warped volume vs target volume
   - Image mode: encoder → motion decoder (frozen) → image decoder → L1 loss

2. **Combined loss**: 
   - Supervised: `L_total = L2(flow) + λ * L1(volume)`
   - Unsupervised: `L_total = L1(warped_volume) + λ * L1(synthesized_volume)`

3. **Gradient updates**: All trainable parameters updated together

4. **Feature passing**: Motion decoder features concatenated with encoder/image features at each resolution level

## Loss Functions

**Supervised Motion Loss (L2):**
```python
motion_loss = MSE(predicted_flow, target_flow)
```

**Unsupervised Motion Loss (L1):**
```python
motion_loss = L1(warped_source_volume, target_volume)
```

**Image Loss (L1):**
```python
image_loss = L1(reconstructed_volume, target_volume)
```

**Total Loss:**
```python
# Supervised
total_loss = L2(flow) + λ * L1(volume)

# Unsupervised (default)
total_loss = L1(warped_volume) + λ * L1(synthesized_volume)
```

## Model Loading & Inference

```python
from utilities.networks import SingleEncoderDualDecoder
import torch

# Load model
model = SingleEncoderDualDecoder(
    im_size=128, 
    int_steps=7, 
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
