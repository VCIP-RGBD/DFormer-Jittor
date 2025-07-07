"""
Data loader utilities for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import cv2
import jittor as jt
import numpy as np
from jittor.dataset import Dataset
import random

from utils.transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
)


def random_mirror(rgb, gt, modal_x):
    """Apply random horizontal mirroring to RGB, ground truth, and modal data."""
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)
    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scales):
    """Apply random scaling to RGB, ground truth, and modal data."""
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return rgb, gt, modal_x, scale


class TrainPre(object):
    """Training data preprocessing class."""
    
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        """Apply preprocessing to training data."""
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(rgb, gt, modal_x, self.config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])
        else:
            modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        return rgb, gt, modal_x


class ValPre(object):
    """Validation data preprocessing class."""
    
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        """Apply preprocessing to validation data."""
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])
        else:
            modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        rgb = rgb.transpose(2, 0, 1)
        modal_x = modal_x.transpose(2, 0, 1)

        return rgb, gt, modal_x


def get_train_loader(engine, dataset_cls, config):
    """Create training data loader."""
    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'x_root': config.x_root_folder,
        'x_format': config.x_format,
        'x_single_channel': config.x_is_single_channel,
        'class_names': config.class_names,
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'class_names': config.class_names
    }
    
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)
    
    train_dataset = dataset_cls(data_setting, "train", train_preprocess, config.batch_size)
    
    train_sampler = None  # TODO: Implement distributed sampler if needed
    
    train_loader = train_dataset.set_attrs(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    return train_loader, train_sampler


def get_val_loader(engine, dataset_cls, config, val_batch_size=None):
    """Create validation data loader."""
    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'x_root': config.x_root_folder,
        'x_format': config.x_format,
        'x_single_channel': config.x_is_single_channel,
        'class_names': config.class_names,
        'train_source': config.train_source,
        'eval_source': config.eval_source,
        'class_names': config.class_names
    }
    
    val_preprocess = ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)
    
    val_dataset = dataset_cls(data_setting, "val", val_preprocess, val_batch_size or config.batch_size)
    
    val_sampler = None  # TODO: Implement distributed sampler if needed
    
    val_loader = val_dataset.set_attrs(
        batch_size=val_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    )
    
    return val_loader, val_sampler
