#!/bin/bash

# DFormer Jittor Training Script - Fixed Version
# This version includes fixes for eval hanging issues

GPUS=1
export CUDA_VISIBLE_DEVICES="0"

# Memory optimization environment variables
export JT_SYNC=0
export JT_LAZY=1
export CUDA_LAUNCH_BLOCKING=0

echo "Starting DFormer training with eval fixes..."
echo "Fixes applied:"
echo "- Reduced num_workers to 0 for validation (no multiprocessing)"
echo "- Reduced num_workers to 2 for training"
echo "- Disabled MSF evaluation in early epochs"
echo "- Reduced batch sizes"
echo "- Optimized memory cleanup"
echo "- Reduced sync frequency"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python utils/train.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S \
    --gpus=$GPUS \
    --no-sliding \
    --no-syncbn \
    --no-mst \
    --no-amp \
    --no-val_amp \
    --no-pad_SUNRGBD \
    --no-use_seed
