# Respiratory Motion Estimation from Fluoroscopy — Ablation Study

Deep-learning pipeline for volumetric respiratory motion estimation from 2-D fluoroscopic projections. A source CT volume is warped toward a target respiratory phase by predicting a dense deformation vector field (DVF) conditioned on projection data. The repo implements an ablation study over four network variants, trained in a leave-one-volume-out fashion.

## Overview

Given a fixed source CT volume (phase `06`) and a target 2-D projection, each model predicts a DVF that warps the source volume to match the target phase. Diffeomorphic integration (scaling-and-squaring) and spatial transformation are applied to produce the warped output. Dual variants add a second image-decoder "cycle" branch with learned uncertainty weighting between the two reconstruction losses.

## Model Variants

| Variant | Inputs | Cycle branch | Notes |
|---|---|---|---|
| `proj-single` | source proj, target proj, source vol | — | 2-D projection pair → DVF → warped volume |
| `proj-dual` | source proj, target proj, source vol | ✓ | `proj-single` + image-decoder cycle branch |
| `vol-dual` | source vol, target proj | ✓ | Volume + target projection → DVF (no z-coordinate) |
| `vol-dual-z` | source vol, target proj | ✓ | `vol-dual` + learned z-coordinate channel |

All variants use a 3-D residual encoder/decoder. Projection features are embedded in 2-D and broadcast along the depth axis before fusion with volumetric features.

## Architecture

- **Embedders** — 2-D / 3-D residual blocks (`ResBlock2D`, `ResBlock3D`) with GroupNorm.
- **Encoder** (`Encoder3D`) — strided residual downsampling; channel widths scale with `im_size`.
- **Decoders** (`Decoder3D`) — transposed-conv upsampling arm followed by tanh "extra" refinement blocks; output 3-channel DVF.
- **Integration** (`layers.VecInt`) — diffeomorphic vector integration over `int_steps`.
- **Warping** (`layers.SpatialTransformer`) — applies the integrated DVF to the source volume.

Dual variants expose two learnable parameters, `log_var_dvf` and `log_var_img`, for uncertainty-weighted multi-task loss balancing.

## Repository Layout

```
network.py            Model definitions + build_model() factory
train.py              Trains a single (variant, excl_vol) combination
utilities/
  layers.py           VecInt, SpatialTransformer
  modelio.py          LoadableModel, store_config_args
weights/              Best checkpoints (created at runtime)
plots/                Per-run loss / uncertainty plots
logs/                 Per-run training logs
```

## Data

Set via `IM_DIR` in `train.py` (default `/srv/shared/data/pixelprint`). Expected files:

- `sub_CT_{phase}_mha.npy` — CT volume per phase, reshaped to `128³`.
- `{phase}_proj_{NNNNN}_bin.npy` — 2-D projections, indexed `1 … 397` per phase.

Eight respiratory phases (`01`–`08`); phase `06` is the fixed source. Global min/max normalisation stats are computed per run over the training phases (excluding the held-out volume).

## Usage

Train a single variant with one volume held out (leave-one-out):

```bash
python train.py --variant proj-single --excl_vol 01 --gpu 0
python train.py --variant proj-dual   --excl_vol 03 --gpu 1
python train.py --variant vol-dual    --excl_vol 05 --gpu 2
python train.py --variant vol-dual-z  --excl_vol 07 --gpu 3
```

**Arguments**

- `--variant` — one of `proj-single`, `proj-dual`, `vol-dual`, `vol-dual-z`.
- `--excl_vol` — volume held out from training (`01`–`08`).
- `--gpu` — CUDA device index (default `0`).

Outputs per run: a best checkpoint in `weights/`, a loss/uncertainty plot in `plots/`, and a tee'd log in `logs/`.

## Training Configuration

Defined in `train.py` (`TRAIN_CONFIG`):

| Setting | Value |
|---|---|
| Epochs | 80 |
| Learning rate | 1e-5 |
| Batch size | 4 |
| Steps per epoch | 3000 |
| Image size | 128³ |
| Integration steps | 7 |
| Optimizer | Adam |

The dataset is re-split 90/10 (train/val) each epoch. Checkpoints are saved on best validation metric (`warp + cycle` L1).

## Loss

- `proj-single` — L1 between warped and target volume.
- Dual variants — uncertainty-weighted L1 over both the warp output and the cycle output:

  ```
  L = 0.5·exp(−log_var_dvf)·L1_warp + 0.5·log_var_dvf
    + 0.5·exp(−log_var_img)·L1_cycle + 0.5·log_var_img
  ```

  Log-variances are clamped at `LV_CLAMP = -3.0`; a warning is logged when they exceed `LV_WARN = 2.0`.

## Requirements

- Python 3.x
- PyTorch (CUDA-enabled)
- NumPy
- Matplotlib

Requires the `utilities` package (`layers`, `modelio`) on the path.

## Notes

- `build_model(variant, im_size=128, int_steps=7)` is the single entry point for constructing any variant.
- Setting `int_steps=0` disables diffeomorphic integration.
