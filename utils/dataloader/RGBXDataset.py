"""
RGBX Dataset for DFormer Jittor implementation
Adapted from PyTorch version for Jittor framework
"""

import os
import cv2
import jittor as jt
import numpy as np
from jittor.dataset import Dataset


def get_path(
    dataset_name,
    _rgb_path,
    _rgb_format,
    _x_path,
    _x_format,
    _gt_path,
    _gt_format,
    x_modal,
    item_name,
):
    """Get file paths for RGB, depth/modal, and ground truth data."""
    if dataset_name == "StanFord2D3D":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("/rgb/", "/depth/").replace("_rgb", "_newdepth")
            + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "")
            .replace(".png", "")
            .replace("/rgb/", "/semantic/")
            .replace("_rgb", "_newsemantic")
            + _gt_format,
        )
    elif dataset_name == "new_stanford":
        area = item_name.split(" ")[0]
        name = item_name.split(" ")[1]
        rgb_path = os.path.join(_rgb_path, area + "/image/" + name + _rgb_format)
        d_path = os.path.join(_x_path, area + "/hha/" + name + _x_format)
        gt_path = os.path.join(_gt_path, area + "/label/" + name + _gt_format)
    elif dataset_name == "KITTI-360":
        rgb_path = os.path.join(_rgb_path, item_name.split(" ")[0])
        d_path = os.path.join(
            _x_path,
            item_name.split(" ")[0]
            .replace("data_2d_raw", "data_3d_rangeview")
            .replace("image_00/data_rect", "velodyne_points/data"),
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.split(" ")[1].replace("data_2d_semantics", "data_2d_semantics_trainID"),
        )
    elif dataset_name == "Scannet":
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_depth") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "").replace("color", "convert_label") + _gt_format,
        )
    elif dataset_name == "MFNet":
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(_x_path, item_name + _x_format)
        gt_path = os.path.join(_gt_path, item_name + _gt_format)
    elif dataset_name == "EventScape":
        item_name = item_name.split(".png")[0]
        img_name = item_name.split("/")[-1]
        img_id = img_name.replace("_image", "").split("_")[-1]
        rgb_path = os.path.join(_rgb_path, item_name + _rgb_format)
        d_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "depth").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        e_path = os.path.join(
            _x_path,
            item_name.replace("rgb", "events").replace("data", "frames").replace(img_name, img_id) + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace("rgb", "semantic").replace("image", "gt_labelIds") + _gt_format,
        )
    else:
        # Default case for NYUDepthv2 and SUNRGBD
        item_name = item_name.split("/")[1].split(".jpg")[0] if "/" in item_name else item_name.split(".jpg")[0]
        rgb_path = os.path.join(
            _rgb_path,
            item_name.replace(".jpg", "").replace(".png", "") + _rgb_format,
        )
        d_path = os.path.join(
            _x_path,
            item_name.replace(".jpg", "").replace(".png", "") + _x_format,
        )
        gt_path = os.path.join(
            _gt_path,
            item_name.replace(".jpg", "").replace(".png", "") + _gt_format,
        )

    return rgb_path, d_path, gt_path


class RGBXDataset(Dataset):
    """RGBX Dataset class for multi-modal semantic segmentation."""
    
    def __init__(self, setting, split_name, preprocess=None, batch_size=1):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._class_names = setting['class_names']
        self.preprocess = preprocess
        self.batch_size = batch_size

        # Determine dataset name from path
        if 'NYU' in self._rgb_path or 'nyu' in self._rgb_path:
            self.dataset_name = 'NYUDepthv2'
        elif 'SUNRGBD' in self._rgb_path or 'sunrgbd' in self._rgb_path:
            self.dataset_name = 'SUNRGBD'
        else:
            self.dataset_name = 'Unknown'

        # Load file lists
        if split_name == 'train':
            with open(self._train_source, 'r') as f:
                self.data_list = f.read().splitlines()
        else:
            with open(self._eval_source, 'r') as f:
                self.data_list = f.read().splitlines()

        self.total_len = len(self.data_list)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        """Get a single data sample."""
        if index >= self.total_len:
            index = index % self.total_len

        item_name = self.data_list[index]
        
        # Get file paths
        rgb_path, x_path, gt_path = get_path(
            self.dataset_name,
            self._rgb_path,
            self._rgb_format,
            self._x_path,
            self._x_format,
            self._gt_path,
            self._gt_format,
            'depth',  # x_modal
            item_name,
        )

        # Load RGB image
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth/modal image
        if self._x_single_channel:
            modal_x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
            if modal_x is None:
                raise FileNotFoundError(f"Modal image not found: {x_path}")
            modal_x = np.expand_dims(modal_x, axis=2)
            modal_x = np.concatenate([modal_x, modal_x, modal_x], axis=2)
        else:
            modal_x = cv2.imread(x_path, cv2.IMREAD_COLOR)
            if modal_x is None:
                raise FileNotFoundError(f"Modal image not found: {x_path}")
            modal_x = cv2.cvtColor(modal_x, cv2.COLOR_BGR2RGB)

        # Load ground truth
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")

        # Apply transformations if specified
        if self._transform_gt:
            gt = self._transform_gt(gt)

        # Apply preprocessing
        if self.preprocess is not None:
            rgb, gt, modal_x = self.preprocess(rgb, gt, modal_x)

        # Convert to Jittor tensors
        rgb = jt.array(rgb).float32()
        modal_x = jt.array(modal_x).float32()
        gt = jt.array(gt).int64()

        return rgb, modal_x, gt
