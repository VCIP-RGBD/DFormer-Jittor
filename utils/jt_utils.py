"""
Jittor utilities for DFormer implementation
Adapted from PyTorch pyt_utils for Jittor framework
"""

import os
import sys
import time
import random
import argparse
import logging
from collections import OrderedDict, defaultdict

import jittor as jt
import numpy as np


def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def link_file(src, target):
    """Create symbolic link."""
    if os.path.islink(target) or os.path.exists(target):
        os.remove(target)
    os.symlink(src, target)


def extant_file(x):
    """Check if file exists."""
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def parse_devices(input_devices):
    """Parse device string."""
    if input_devices.endswith('*'):
        devices = list(range(jt.flags.use_cuda))
        return devices
    else:
        devices = [int(d) for d in input_devices.split(',')]
        return devices


def load_model(model, model_file, is_restore=False):
    """Load model from checkpoint file."""
    t_start = time.time()
    
    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = jt.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    print("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))

    return model


def save_model(model, model_file):
    """Save model to checkpoint file."""
    ensure_dir(os.path.dirname(model_file))
    jt.save(model.state_dict(), model_file)
    print("Model saved to {}".format(model_file))


def all_reduce_tensor(tensor, op=None, world_size=1):
    """All reduce tensor across processes."""
    # For single GPU, just return the tensor
    return tensor


def reduce_tensor(tensor):
    """Reduce tensor across processes."""
    # For single GPU, just return the tensor
    return tensor


def get_world_size():
    """Get world size."""
    return 1


def get_rank():
    """Get current rank."""
    return 0


def is_main_process():
    """Check if current process is main process."""
    return True


def synchronize():
    """Synchronize all processes."""
    pass


def time_synchronized():
    """Get synchronized time."""
    jt.sync_all()
    return time.time()


class AverageMeter(object):
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Timer utility."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self, average=True):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def format_time(seconds):
    """Format time in seconds to readable string."""
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
