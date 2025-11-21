#!/bin/bash

# Script to train all network variants
# Usage: bash train_all.sh [optional arguments]

# Default parameters (can be overridden with command line args)
IM_DIR="data/xcat/train"
IM_SIZE=128
BATCH_SIZE=8
EPOCHS=50
LR=1e-5
INT_STEPS=10

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --im_dir)
            IM_DIR="$2"
            shift 2
            ;;
        --im_size)
            IM_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training all network variants with:"
echo "  Image directory: $IM_DIR"
echo "  Image size: $IM_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

# Array of architectures
ARCHITECTURES=("concatenated" "dual" "separate" "broadcast")

# Train each architecture with and without FiLM
for arch in "${ARCHITECTURES[@]}"; do
    echo "========================================="
    echo "Training: $arch (without FiLM)"
    echo "========================================="
    python train.py \
        --architecture $arch \
        --im_dir $IM_DIR \
        --im_size $IM_SIZE \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --int_steps $INT_STEPS
    
    echo ""
    echo "========================================="
    echo "Training: $arch (with FiLM)"
    echo "========================================="
    python train.py \
        --architecture $arch \
        --use_film \
        --im_dir $IM_DIR \
        --im_size $IM_SIZE \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        --int_steps $INT_STEPS
    
    echo ""
done

echo "========================================="
echo "All training complete!"
echo "========================================="
