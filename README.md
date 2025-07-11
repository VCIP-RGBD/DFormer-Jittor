# <p align=center>`DFormer for RGBD Semantic Segmentation (Jittor Implementation)`</p>

<p align="center">
    <img src="https://img.shields.io/badge/Framework-Jittor-brightgreen" alt="Framework">
    <img src="https://img.shields.io/badge/License-Non--Commercial-red" alt="License">
    <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
</p>

This is the Jittor implementation of DFormer and DFormerv2 for RGBD semantic segmentation. Developed based on the Jittor deep learning framework, it provides efficient solutions for training and inference.

This repository contains the official Jittor implementation of the following papers:

> DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation<br/>
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=en),
> [Xuying Zhang](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=en),
> [Li Liu](https://scholar.google.com/citations?hl=en&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en) <br/>
> ICLR 2024. 
>[Paper Link](https://arxiv.org/abs/2309.09668) |
>[Homepage](https://yinbow.github.io/Projects/DFormer/index.html) |
>[PyTorch Version](https://github.com/VCIP-RGBD/DFormer)

> DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation<br/>
> [Bo-Wen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=en),
> [Jiao-Long Cao](https://github.com/caojiaolong),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=en&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en)<br/>
> CVPR 2025. 
> [Paper Link](https://arxiv.org/abs/2504.04701) |
> [Chinese Version](https://mftp.mmcheng.net/Papers/25CVPR_RGBDSeg-CN.pdf) |
> [PyTorch Version](https://github.com/VCIP-RGBD/DFormer)

## 🚀 Getting Started

### Environment Setup

```bash
# Create a conda environment
conda create -n dformer_jittor python=3.8 -y
conda activate dformer_jittor

# Install Jittor
pip install jittor

# Install other dependencies
pip install opencv-python pillow numpy scipy tqdm tensorboardX tabulate easydict
```

### Dataset Preparation

Supported datasets:
- **NYUDepthv2**: An indoor RGBD semantic segmentation dataset.
- **SUNRGBD**: A large-scale dataset for indoor scene understanding.

Download links:
| Dataset | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|

### Pre-trained Models

| Model | Dataset | mIoU | Download Link |
|------|--------|------|----------|
| DFormer-Small | NYUDepthv2 | 52.3 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormer-Base | NYUDepthv2 | 54.1 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormer-Large | NYUDepthv2 | 55.8 | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn) |
| DFormerv2-Small | NYUDepthv2 | 53.7 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |
| DFormerv2-Base | NYUDepthv2 | 55.3 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |
| DFormerv2-Large | NYUDepthv2 | 57.1 | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym) |

### Directory Structure

```
DFormer-Jittor/
├── checkpoints/              # Directory for pre-trained models
│   ├── pretrained/          # ImageNet pre-trained models
│   └── trained/             # Trained models
├── datasets/                # Directory for datasets
│   ├── NYUDepthv2/         # NYU dataset
│   └── SUNRGBD/            # SUNRGBD dataset
├── local_configs/          # Configuration files
├── models/                 # Model definitions
├── utils/                  # Utility functions
├── train.sh               # Training script
├── eval.sh                # Evaluation script
└── infer.sh               # Inference script
```

## 📖 Usage

### Training

Use the provided training script:
```bash
bash train.sh
```

Or use the Python command directly:
```bash
python utils/train.py --config local_configs.NYUDepthv2.DFormer_Base
```

### Evaluation

```bash
bash eval.sh
```

Alternatively:
```bash
python utils/eval.py --config local_configs.NYUDepthv2.DFormer_Base --checkpoint checkpoints/trained/NYUDepthv2/DFormer_Base/best.pkl
```

### Inference/Visualization

```bash
bash infer.sh
```

## 🎯 Performance

### NYUDepthv2 Dataset

| Method | Backbone | mIoU | Params | FLOPs |
|------|----------|------|--------|-------|
| DFormer-T | DFormer-Tiny | 48.5 | 5.0M | 15.2G |
| DFormer-S | DFormer-Small | 52.3 | 13.1M | 28.4G |
| DFormer-B | DFormer-Base | 54.1 | 35.4M | 75.0G |
| DFormer-L | DFormer-Large | 55.8 | 62.3M | 132.8G |
| DFormerv2-S | DFormerv2-Small | 53.7 | 13.1M | 28.4G |
| DFormerv2-B | DFormerv2-Base | 55.3 | 35.4M | 75.0G |
| DFormerv2-L | DFormerv2-Large | 57.1 | 62.3M | 132.8G |

### SUNRGBD Dataset

| Method | Backbone | mIoU | Params | FLOPs |
|------|----------|------|--------|-------|
| DFormer-T | DFormer-Tiny | 46.2 | 5.0M | 15.2G |
| DFormer-S | DFormer-Small | 49.8 | 13.1M | 28.4G |
| DFormer-B | DFormer-Base | 51.6 | 35.4M | 75.0G |
| DFormer-L | DFormer-Large | 53.4 | 62.3M | 132.8G |
| DFormerv2-S | DFormerv2-Small | 51.2 | 13.1M | 28.4G |
| DFormerv2-B | DFormerv2-Base | 52.8 | 35.4M | 75.0G |
| DFormerv2-L | DFormerv2-Large | 54.5 | 62.3M | 132.8G |

## 🔧 Configuration

The project uses Python configuration files located in the `local_configs/` directory:

```python
# local_configs/NYUDepthv2/DFormer_Base.py
class C:
    # Dataset configuration
    dataset_name = "NYUDepthv2"
    dataset_dir = "datasets/NYUDepthv2"
    num_classes = 40
    
    # Model configuration
    backbone = "DFormer_Base"
    pretrained_model = "checkpoints/pretrained/DFormer_Base.pth"
    
    # Training configuration
    batch_size = 8
    nepochs = 500
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    
    # Other configurations
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
```

## 📊 Benchmarking

### FLOPs and Parameters

```bash
python benchmark.py --config local_configs.NYUDepthv2.DFormer_Base
```

### Inference Speed

```bash
python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Base
```

## 🤝 Contributing

We welcome all forms of contributions:

1. **Bug Reports**: Report issues in GitHub Issues.
2. **Feature Requests**: Suggest new features.
3. **Code Contributions**: Submit Pull Requests.
4. **Documentation Improvements**: Improve README and code comments.


## 📞 Contact

If you have any questions about our work, feel free to contact us:

- Email: bowenyin@mail.nankai.edu.cn, caojiaolong@mail.nankai.edu.cn
- GitHub Issues: [Submit an issue](https://github.com/VCIP-RGBD/DFormer-Jittor/issues)

## 📚 Citation

If you use our work in your research, please cite the following papers:

```bibtex
@inproceedings{yin2024dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhong-Yu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{yin2025dformerv2,
  title={DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation},
  author={Yin, Bo-Wen and Cao, Jiao-Long and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19345--19355},
  year={2025}
}
```

## 🙏 Acknowledgements

Our implementation is mainly based on the following open-source projects:

- [Jittor](https://github.com/Jittor/jittor): A deep learning framework.
- [DFormer](https://github.com/VCIP-RGBD/DFormer): The original PyTorch implementation.
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation): A semantic segmentation toolbox.

Thanks to all the contributors for their efforts!

## 📄 License

This project is for non-commercial use only. See the [LICENSE](LICENSE) file for details.

---

