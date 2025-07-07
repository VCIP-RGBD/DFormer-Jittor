# DFormer for RGBD Semantic Segmentation (Jittor Implementation)

This is the Jittor implementation of DFormer and DFormerv2 for RGBD semantic segmentation.

## Original Papers

> DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation<br/>
> ICLR 2024

> DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation<br/>
> CVPR 2025

## Installation

```bash
conda create -n dformer_jittor python=3.8 -y
conda activate dformer_jittor

# Install Jittor
pip install jittor
```

## Usage

### Train
```bash
python train.py --config configs/nyu_dformer_base.py
```

### Test
```bash
python test.py --config configs/nyu_dformer_base.py --checkpoint checkpoints/best.pkl
```

## Model Zoo

Coming soon...

## Acknowledgments

This implementation is based on the original PyTorch implementation of DFormer. 