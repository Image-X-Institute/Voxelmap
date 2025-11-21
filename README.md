# Network Architecture Variants

This codebase implements multiple architectural variants for 2D-to-3D deformation field prediction with optional FiLM conditioning.

## Architecture Variants

### 1. Concatenated (Original)
- Concatenates source and target projections
- Single 2D encoding path
- Full depth encoding (original number of blocks)

### 2. Dual Encoder
- Separate identical encoders for source and target projections
- Features concatenated after encoding
- One fewer residual block than original

### 3. Separate Projection-Volume Encoder
- Independent 2D encoder for projections
- Independent 3D encoder for volume
- Features combined after encoding
- One fewer residual block than original

### 4. Broadcast Encoder
- Projects 2D features to 3D by broadcasting
- Single 3D residual block processes projected features
- Combined with volume in single 3D encoding path
- One fewer residual block than original

### FiLM Conditioning
Each architecture has a variant with Feature-wise Linear Modulation (FiLM) that conditions on gantry angle:
- Requires `Angles.csv` in image directory with columns: `filename`, `angle`
- FiLM layers inserted in each encoding block
- Learns angle-dependent affine transformations: `γ(θ) * x + β(θ)`

## File Structure

```
network_variants.py      # All architecture implementations
train_variants.py        # Training script for any variant
validate_model.py        # Validation and visualization
train_all_variants.sh    # Batch training script
```

## Training

### Single architecture:
```bash
python train_variants.py \
    --architecture concatenated \
    --im_dir data/xcat/train \
    --im_size 128 \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-5
```

### With FiLM:
```bash
python train_variants.py \
    --architecture concatenated \
    --use_film \
    --im_dir data/xcat/train \
    --im_size 128 \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-5
```

### All variants:
```bash
bash train_all_variants.sh --epochs 50
```

## Validation

```bash
python validate_model.py \
    --checkpoint outputs/concatenated/weights/best_model.pth \
    --im_dir data/xcat/train \
    --phase 01 \
    --slice_idx 64
```

Generates two figures:
1. **DVF comparison**: Target vs predicted deformation fields (X, Y, Z components + magnitude + errors)
2. **Volume comparison**: Source, target, predicted volumes with error maps

## Output Structure

```
outputs/
├── concatenated/
│   ├── weights/
│   │   └── best_model.pth
│   ├── plots/
│   │   └── training_curve.png
│   └── validation/
│       ├── dvf_comparison_phase_01.png
│       └── volume_comparison_phase_01.png
├── concatenated_film/
├── dual/
├── dual_film/
├── separate/
├── separate_film/
├── broadcast/
└── broadcast_film/
```

## Data Requirements

Expected files in `im_dir`:
- `XX_YY_bin.npy`: Projections (XX=phase, YY=angle)
- `sub_DVF_XX_mha.npy`: Target deformation fields
- `sub_CT_XX_mha.npy`: Target volumes
- `sub_CT_06_mha.npy`: Source volume (reference phase)
- `sub_Abdomen_mha.npy`: Abdomen mask
- `Angles.csv`: Gantry angles (required only for FiLM variants)

## Key Parameters

- `--architecture`: `concatenated`, `dual`, `separate`, `broadcast`
- `--use_film`: Enable angle conditioning
- `--int_steps`: Integration steps for diffeomorphic transform (default: 10)
- `--im_size`: Input/output resolution (default: 128)
- `--epochs`: Training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-5)

## Metrics

Validation reports:
- **DVF**: MSE, MAE
- **Volume**: MSE, MAE, PSNR
