#!/bin/bash
# Train all network variants with and without skip connections

echo "=========================================="
echo "Training All Network Variants"
echo "=========================================="

# Create output directories
mkdir -p weights
mkdir -p plots
mkdir -p logs

# Original Model (baseline)
echo ""
echo ">>> Training Original Model"
python3 train.py \
    --model_variant original \
    --motion_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/original.log

# Single Encoder, Dual Decoder - No Skip - L1 Loss
echo ""
echo ">>> Training Single Encoder Dual Decoder (No Skip, L1)"
python3 train.py \
    --model_variant single_encoder \
    --skip_connections false \
    --image_loss_type l1 \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/single_encoder_noskip_l1.log

# Single Encoder, Dual Decoder - No Skip - Perceptual Loss
echo ""
echo ">>> Training Single Encoder Dual Decoder (No Skip, Perceptual)"
python3 train.py \
    --model_variant single_encoder \
    --skip_connections false \
    --image_loss_type perceptual \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/single_encoder_noskip_perceptual.log

# Single Encoder, Dual Decoder - With Skip - L1 Loss
echo ""
echo ">>> Training Single Encoder Dual Decoder (Skip, L1)"
python3 train.py \
    --model_variant single_encoder \
    --skip_connections true \
    --image_loss_type l1 \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/single_encoder_skip_l1.log

# Single Encoder, Dual Decoder - With Skip - Perceptual Loss
echo ""
echo ">>> Training Single Encoder Dual Decoder (Skip, Perceptual)"
python3 train.py \
    --model_variant single_encoder \
    --skip_connections true \
    --image_loss_type perceptual \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/single_encoder_skip_perceptual.log

# Dual Encoder, Dual Decoder - No Skip - L1 Loss
echo ""
echo ">>> Training Dual Encoder Dual Decoder (No Skip, L1)"
python3 train.py \
    --model_variant dual_encoder \
    --skip_connections false \
    --image_loss_type l1 \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/dual_encoder_noskip_l1.log

# Dual Encoder, Dual Decoder - No Skip - Perceptual Loss
echo ""
echo ">>> Training Dual Encoder Dual Decoder (No Skip, Perceptual)"
python3 train.py \
    --model_variant dual_encoder \
    --skip_connections false \
    --image_loss_type perceptual \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/dual_encoder_noskip_perceptual.log

# Dual Encoder, Dual Decoder - With Skip - L1 Loss
echo ""
echo ">>> Training Dual Encoder Dual Decoder (Skip, L1)"
python3 train.py \
    --model_variant dual_encoder \
    --skip_connections true \
    --image_loss_type l1 \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/dual_encoder_skip_l1.log

# Dual Encoder, Dual Decoder - With Skip - Perceptual Loss
echo ""
echo ">>> Training Dual Encoder Dual Decoder (Skip, Perceptual)"
python3 train.py \
    --model_variant dual_encoder \
    --skip_connections tru
    --image_loss_type perceptual \
    --motion_epochs 100 \
    --image_epochs 100 \
    --lr 1e-4 \
    --int_steps 7 \
    2>&1 | tee logs/dual_encoder_skip_perceptual.log

echo ""
echo "=========================================="
echo "All Training Complete!"
echo "=========================================="
echo "Models saved in: weights/"
echo "Plots saved in: plots/"
echo "Logs saved in: logs/"
echo ""
echo "Total variants trained: 9"
echo "  - 1 Original baseline"
echo "  - 4 Single Encoder (skip/noskip × L1/perceptual)"
echo "  - 4 Dual Encoder (skip/noskip × L1/perceptual)"
