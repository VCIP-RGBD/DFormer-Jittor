#!/bin/bash

# DFormer Jittor Training Script
# Adapted from PyTorch version for Jittor framework

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1"

# Training configuration
GPUS=2
CONFIG="local_configs.NYUDepthv2.DFormerv2_S"
BATCH_SIZE=8
EPOCHS=500
LR=6e-5

# Run training
python utils/train.py \
    --config=$CONFIG \
    --gpus=$GPUS \
    --batch_size=$BATCH_SIZE \
    --epochs=$EPOCHS \
    --lr=$LR \
    --syncbn \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD

# Available configurations for DFormers on NYUDepthv2:
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormerv2_S
# local_configs.NYUDepthv2.DFormerv2_B
# local_configs.NYUDepthv2.DFormerv2_L

# Available configurations for DFormers on SUNRGBD:
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormerv2_S
# local_configs.SUNRGBD.DFormerv2_B
# local_configs.SUNRGBD.DFormerv2_L
