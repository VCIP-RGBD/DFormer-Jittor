"""
Non-Local Neural Networks decoder converted to Jittor.
Simplified implementation without mmcv/mmseg dependencies.
"""

import jittor as jt
from jittor import nn
from abc import ABCMeta
from typing import Dict, Optional


def normal_init(module, std=0.01):
    """Normal initialization."""
    if hasattr(module, 'weight'):
        nn.init.gauss_(module.weight, 0, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)


def constant_init(module, val):
    """Constant initialization.""" 
    if hasattr(module, 'weight'):
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, val)


class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module."""
    # ... rest of implementation stays the same