#!/bin/bash
# Script to train all architecture variants for ablation study
# Usage: bash train_all.sh [optional arguments]

# Default parameters (can be overridden with command line args)
DATA_DIR="data/combat/train"
IM_SIZE=128
BATCH_SIZE=8
EPOCHS=50
LR=1e-5
INT_STEPS=7
TRAIN_SPLIT=0.9
SUPERVISED=false
SKIP=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
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
        --int_steps)
            INT_STEPS="$2"
            shift 2
            ;;
        --train_split)
            TRAIN_SPLIT="$2"
            shift 2
            ;;
        --supervised)
            SUPERVISED=true
            shift
            ;;
        --skip_connections)
            SKIP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Available options:"
            echo "  --data_dir <path>"
            echo "  --im_size <int>"
            echo "  --batch_size <int>"
            echo "  --epochs <int>"
            echo "  --lr <float>"
            echo "  --int_steps <int>"
            echo "  --train_split <float>"
            echo "  --supervised"
            echo "  --skip_connections"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "ABLATION STUDY - Training All Variants"
echo "========================================="
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Image size: $IM_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Integration steps: $INT_STEPS"
echo "  Mode: $([ "$SUPERVISED" = true ] && echo "Supervised" || echo "Unsupervised")"
echo "  Skip connections: $([ "$SKIP" = true ] && echo "Enabled" || echo "Disabled")"
echo "========================================="
echo ""

# Array of architectures
ARCHITECTURES=("original_mri" "simple_3d" "dual_stream_2d" "hybrid")

# Build base command
BASE_CMD="python train.py \
    --data_dir $DATA_DIR \
    --im_size $IM_SIZE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --int_steps $INT_STEPS \
    --train_split $TRAIN_SPLIT"

# Add supervised flag if set
if [ "$SUPERVISED" = true ]; then
    BASE_CMD="$BASE_CMD --supervised"
fi

# Add skip connections flag if set
if [ "$SKIP" = true ]; then
    BASE_CMD="$BASE_CMD --skip_connections"
fi

# Train each architecture
for arch in "${ARCHITECTURES[@]}"; do
    echo "========================================="
    echo "Training: $arch"
    echo "========================================="
    
    $BASE_CMD --architecture $arch
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for $arch"
        exit 1
    fi
    
    echo ""
done

echo "========================================="
echo "All training complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  Weights: weights/"
echo "  Plots: plots/"
echo ""
echo "Trained architectures:"
for arch in "${ARCHITECTURES[@]}"; do
    MODE=$([ "$SUPERVISED" = true ] && echo "sup" || echo "unsup")
    SKIP_STR=$([ "$SKIP" = true ] && echo "skip" || echo "noskip")
    echo "  - ${arch}_${MODE}_${SKIP_STR}_int${INT_STEPS}"
done
