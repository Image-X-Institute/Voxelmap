#!/bin/bash
# Script to train all network variants with joint training
# Usage: bash train_all.sh [optional arguments]

# Default parameters (can be overridden with command line args)
IM_DIR="/srv/shared/SPARE/MC_V_P1_NS_01"
IM_SIZE=128
BATCH_SIZE=8
EPOCHS=50
LR=1e-5
INT_STEPS=7
IMAGE_LOSS_WEIGHT=10.0

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
        --int_steps)
            INT_STEPS="$2"
            shift 2
            ;;
        --image_loss_weight)
            IMAGE_LOSS_WEIGHT="$2"
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
echo "  Integration steps: $INT_STEPS"
echo "  Image loss weight: $IMAGE_LOSS_WEIGHT"
echo ""

# Array of architectures
ARCHITECTURES=("single_encoder" "dual_encoder" "original")

# Train each architecture variant
for arch in "${ARCHITECTURES[@]}"; do
    
    # For original architecture, only train baseline
    if [ "$arch" == "original" ]; then
        # Unsupervised
        echo "========================================="
        echo "Training: $arch (unsupervised)"
        echo "========================================="
        python train2.py \
            --architecture $arch \
            --im_dir $IM_DIR \
            --im_size $IM_SIZE \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LR \
            --int_steps $INT_STEPS
        
        echo ""
        
        # Supervised
        echo "========================================="
        echo "Training: $arch (supervised)"
        echo "========================================="
        python train2.py \
            --architecture $arch \
            --supervised \
            --im_dir $IM_DIR \
            --im_size $IM_SIZE \
            --batch_size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LR \
            --int_steps $INT_STEPS
        
        echo ""
        continue
    fi
    
    # For dual decoder architectures, train with/without skip connections and supervised/unsupervised
    for skip_flag in "" "--skip_connections"; do
        skip_name=""
        if [ ! -z "$skip_flag" ]; then
            skip_name=" + Skip"
        fi
        
        for supervised_flag in "" "--supervised"; do
            supervised_name=""
            if [ ! -z "$supervised_flag" ]; then
                supervised_name=" (supervised)"
            else
                supervised_name=" (unsupervised)"
            fi
            
            echo "========================================="
            echo "Training: $arch${skip_name}${supervised_name}"
            echo "========================================="
            python train2.py \
                --architecture $arch \
                $skip_flag \
                $supervised_flag \
                --im_dir $IM_DIR \
                --im_size $IM_SIZE \
                --batch_size $BATCH_SIZE \
                --epochs $EPOCHS \
                --lr $LR \
                --int_steps $INT_STEPS \
                --image_loss_weight $IMAGE_LOSS_WEIGHT
            
            echo ""
        done
    done
done

echo "========================================="
echo "All training complete!"
echo "========================================="
