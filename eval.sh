#!/bin/bash

# DFormer Jittor Evaluation Script
# Adapted from PyTorch version for Jittor framework

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Evaluation configuration
GPUS=8
CONFIG="local_configs.NYUDepthv2.DFormerv2_S"
CHECKPOINT="checkpoints/trained/DFormerv2_Small_NYU.pkl"

# Run evaluation
python utils/eval.py \
    --config=$CONFIG \
    --gpus=$GPUS \
    --sliding \
    --syncbn \
    --amp \
    --pad_SUNRGBD \
    --continue_fpath=$CHECKPOINT

# Available configurations and checkpoints:

# NYUv2 DFormers
# --config=local_configs.NYUDepthv2.DFormer_Large
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Large.pkl
# --config=local_configs.NYUDepthv2.DFormer_Base  
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Base.pkl
# --config=local_configs.NYUDepthv2.DFormer_Small
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Small.pkl
# --config=local_configs.NYUDepthv2.DFormer_Tiny
# --continue_fpath=checkpoints/trained/NYUv2_DFormer_Tiny.pkl

# NYUv2 DFormerv2
# --config=local_configs.NYUDepthv2.DFormerv2_L
# --continue_fpath=checkpoints/trained/DFormerv2_Large_NYU.pkl
# --config=local_configs.NYUDepthv2.DFormerv2_B
# --continue_fpath=checkpoints/trained/DFormerv2_Base_NYU.pkl
# --config=local_configs.NYUDepthv2.DFormerv2_S
# --continue_fpath=checkpoints/trained/DFormerv2_Small_NYU.pkl

# SUNRGBD DFormers
# --config=local_configs.SUNRGBD.DFormer_Large
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Large.pkl
# --config=local_configs.SUNRGBD.DFormer_Base
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Base.pkl
# --config=local_configs.SUNRGBD.DFormer_Small
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Small.pkl
# --config=local_configs.SUNRGBD.DFormer_Tiny
# --continue_fpath=checkpoints/trained/SUNRGBD_DFormer_Tiny.pkl

# SUNRGBD DFormerv2
# --config=local_configs.SUNRGBD.DFormerv2_L
# --continue_fpath=checkpoints/trained/DFormerv2_Large_SUNRGBD.pkl
# --config=local_configs.SUNRGBD.DFormerv2_B
# --continue_fpath=checkpoints/trained/DFormerv2_Base_SUNRGBD.pkl
# --config=local_configs.SUNRGBD.DFormerv2_S
# --continue_fpath=checkpoints/trained/DFormerv2_Small_SUNRGBD.pkl
