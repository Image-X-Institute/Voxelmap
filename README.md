# Voxelmap network variants for MRI patient data study

This code base implements multiple architecture variants for Voxelmap, supporting both supervised (with DVF ground truth) and unsupervised learning, with a specific focus on MRI data.

## Overview

This framework implements deformable registration between:
- **Input**: Two 2D slices (coronal & sagittal) + source 3D volume
- **Output**: Deformation field that warps source volume to match target slices.

## Architecture Variants

Four architectures available for ablation studies:

| Architecture | Description | Encoder | Fusion |
|-------------|-------------|---------|--------|
| `original_mri` | Separate 2D encoders → 2D-to-3D transform | 2D (dual stream) | Bottleneck concatenation |
| `simple_3d` | Embed projections, single 3D encoder-decoder | 3D (single) | Early fusion |
| `dual_stream_2d` | Separate 2D encoders with early fusion | 2D (dual stream) | Pre-transform concatenation |
| `hybrid` | 2D encoders for projections + 3D for volume | Mixed | Late fusion |

## Installation

```bash
pip install torch numpy matplotlib
```

Required utilities modules:
- `utilities.layers` - VecInt, SpatialTransformer
- `utilities.losses` - loss functions
- `utilities.modelio` - LoadableModel base class

## Usage

### Training

**Supervised (with DVF ground truth):**
```bash
python train.py \
    --architecture original_mri \
    --supervised \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5
```

**Unsupervised (image-based loss):**
```bash
python train.py \
    --architecture simple_3d \
    --skip_connections \
    --epochs 50
```

### Ablation Study

Test all architectures:
```bash
for arch in original_mri simple_3d dual_stream_2d hybrid; do
    python train.py --architecture $arch --supervised --epochs 50
    python train.py --architecture $arch --epochs 50
done
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--architecture` | `original_mri` | Network architecture |
| `--supervised` | False | Use DVF ground truth |
| `--skip_connections` | False | Enable U-Net style skip connections |
| `--int_steps` | 7 | Diffeomorphic integration steps (0=off) |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-5 | Learning rate |

## Data Format

### Directory Structure
```
data/combat/train/
├── 01_s.npy              # Target projections [2, H, W]
├── 06_s.npy              # Source projections [2, H, W]
├── sub_Abdomen_mha.npy   # Source volume [D, H, W]
└── sub_DVF_01_mha.npy    # DVF ground truth [D, H, W, 3] (supervised only)
```

### Data Files
- **Projections** (`*_s.npy`): Shape `[2, H, W]` - index 0=coronal, 1=sagittal
- **Volume** (`sub_Abdomen_mha.npy`): Shape `[D, H, W]` - 3D volume
- **DVF** (`sub_DVF_*_mha.npy`): Shape `[D, H, W, 3]` - displacement field (optional)

## Model Architecture

### Core Components

**Encoder Options:**
- `Encoder2D`: Downsampling with residual blocks (2D projections)
- `Encoder3D`: Downsampling with residual blocks (3D volumes)

**Decoder:**
- `Decoder3D`: Upsampling with optional skip connections
- Outputs 3-channel displacement field

**Integration:**
- `VecInt`: Diffeomorphic flow integration (stationary velocity field)
- `SpatialTransformer`: Applies deformation field to volume

### Forward Pass

```python
model = Proj2VolRegistration(
    im_size=128,
    architecture='original_mri',
    int_steps=7,
    skip_connections=False
)

y_source, flow = model(source_c, source_s, target_c, target_s, source_vol)
```

**Inputs:**
- `source_c, source_s`: Source projections `[B, 1, H, W]`
- `target_c, target_s`: Target projections `[B, 1, H, W]`
- `source_vol`: Source volume `[B, 1, D, H, W]`

**Outputs:**
- `y_source`: Warped volume `[B, 1, D, H, W]`
- `flow`: Displacement field `[B, 3, D, H, W]`

## Loss Functions

### Supervised
```python
loss = flow_mask.loss(target_flow, pred_flow, mask)
```

### Unsupervised
Requires implementation in `utilities.losses`:
```python
class UnsupervisedLoss:
    def loss(self, target_c, target_s, warped_vol, flow):
        # Image similarity (MSE/NCC on DRRs from warped volume)
        # + flow regularization (gradient penalty)
        pass
```

## Output Files

### Weights
```
weights/
└── original_mri_sup_noskip_int7.pth
```

### Training Plots
```
plots/
└── original_mri_sup_noskip_int7.png
```

Filenames encode: `{architecture}_{mode}_{skip}_{integration}`
## License

[Specify your license here]
