"""
Utility functions for Jittor that replace mmcv and mmseg functionality.
All implementations are pure Jittor without PyTorch dependencies.
"""

import jittor as jt
from jittor import nn
import jittor.nn as F
import math
from typing import Dict, List, Optional, Union, Tuple


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Initialize tensor with truncated normal distribution."""
    # TODO: Implement proper truncated normal initialization
    # For now, use standard normal initialization
    nn.init.gauss_(tensor, mean, std)
    return tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def execute(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + jt.rand([x.shape[0]] + [1] * (x.ndim - 1), dtype=x.dtype)
        random_tensor = random_tensor.floor()  # binarize
        
        if self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        
        return x * random_tensor


def build_norm_layer(norm_cfg, num_features):
    """Build normalization layer."""
    if norm_cfg is None:
        return None, nn.Identity()
    
    norm_type = norm_cfg.get('type', 'BN')
    if norm_type in ['BN', 'BatchNorm2d']:
        return None, nn.BatchNorm2d(num_features)
    elif norm_type in ['SyncBN', 'SyncBatchNorm2d']:
        return None, nn.BatchNorm2d(num_features)  # Jittor handles sync automatically
    elif norm_type in ['GN', 'GroupNorm']:
        num_groups = norm_cfg.get('num_groups', 32)
        return None, nn.GroupNorm(num_groups, num_features)
    elif norm_type in ['LN', 'LayerNorm']:
        return None, nn.LayerNorm(num_features)
    else:
        return None, nn.BatchNorm2d(num_features)


def build_activation_layer(act_cfg):
    """Build activation layer."""
    if act_cfg is None:
        return nn.Identity()
    
    act_type = act_cfg.get('type', 'ReLU')
    if act_type == 'ReLU':
        inplace = act_cfg.get('inplace', True)
        return nn.ReLU()
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'LeakyReLU':
        negative_slope = act_cfg.get('negative_slope', 0.01)
        return nn.LeakyReLU(negative_slope)
    elif act_type == 'Swish':
        return nn.Swish()
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    else:
        return nn.ReLU()


def build_dropout(dropout_cfg):
    """Build dropout layer."""
    if dropout_cfg is None:
        return nn.Identity()
    
    if isinstance(dropout_cfg, dict):
        dropout_type = dropout_cfg.get('type', 'Dropout')
        if dropout_type == 'DropPath':
            return DropPath(dropout_cfg.get('drop_prob', 0.0))
        elif dropout_type == 'Dropout':
            return nn.Dropout(dropout_cfg.get('p', 0.5))
        elif dropout_type == 'Dropout2d':
            return nn.Dropout2d(dropout_cfg.get('p', 0.5))
        else:
            return nn.Identity()
    else:
        return dropout_cfg if dropout_cfg is not None else nn.Identity()


def build_conv_layer(conv_cfg, *args, **kwargs):
    """Build convolution layer."""
    if conv_cfg is None:
        conv_cfg = dict(type='Conv2d')
    
    conv_type = conv_cfg.get('type', 'Conv2d')
    if conv_type == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    elif conv_type == 'Conv1d':
        return nn.Conv1d(*args, **kwargs)
    elif conv_type == 'Conv3d':
        return nn.Conv3d(*args, **kwargs)
    else:
        return nn.Conv2d(*args, **kwargs)


class ConvModule(nn.Module):
    """Conv-Norm-Act module."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        inplace=True,
        with_spectral_norm=False,
        padding_mode='zeros',
        order=('conv', 'norm', 'act'),
        **kwargs
    ):
        super(ConvModule, self).__init__()
        
        # Build conv layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Build norm layer
        self.norm_name, self.norm = build_norm_layer(norm_cfg, out_channels)
        
        # Build activation layer
        self.activate = build_activation_layer(act_cfg)
        
        self.order = order
    
    def execute(self, x):
        """Execute function."""
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.norm is not None:
                x = self.norm(x)
            elif layer == 'act' and self.activate is not None:
                x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dw_norm_cfg=None,
        dw_act_cfg=dict(type='ReLU'),
        pw_norm_cfg=None,
        pw_act_cfg=dict(type='ReLU'),
    ):
        super(DepthwiseSeparableConvModule, self).__init__()
        
        # Depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg or norm_cfg,
            act_cfg=dw_act_cfg,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg or norm_cfg,
            act_cfg=pw_act_cfg,
            bias=False
        )

    def execute(self, x):
        """Execute function."""
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class BaseModule(nn.Module):
    """Base module for all models."""
    
    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self.init_cfg = init_cfg
        self._is_init = False
    
    def init_weights(self):
        """Initialize weights."""
        if self.init_cfg is not None:
            # TODO: Implement weight initialization based on config
            pass
        self._is_init = True


class FFN(nn.Module):
    """Feed-forward network."""
    
    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        num_fcs=2,
        act_cfg=dict(type='ReLU'),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None
    ):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        self.add_identity = add_identity
        
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            if self.activate is not None:
                layers.append(self.activate)
            if ffn_drop > 0:
                layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        
        # Final layer
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        if ffn_drop > 0:
            layers.append(nn.Dropout(ffn_drop))
        
        self.layers = nn.Sequential(*layers)
        
        # Dropout layer
        self.dropout_layer = build_dropout(dropout_layer)
    
    def execute(self, x, identity=None):
        """Execute function."""
        out = self.dropout_layer(self.layers(x))
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


def load_state_dict(model, state_dict, strict=True):
    """Load state dict to model."""
    missing_keys = []
    unexpected_keys = []
    
    model_state_dict = model.state_dict()
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            try:
                model_state_dict[key].assign(value)
            except Exception as e:
                print(f"Failed to load parameter {key}: {e}")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"Error loading state dict: missing {len(missing_keys)} keys, unexpected {len(unexpected_keys)} keys")


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from file."""
    try:
        # TODO: Implement proper checkpoint loading for Jittor
        checkpoint = jt.load(filename)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        load_state_dict(model, state_dict, strict)
        return checkpoint
    except Exception as e:
        print(f"Failed to load checkpoint from {filename}: {e}")
        return None


def resize(
    input,
    size=None,
    scale_factor=None,
    mode='nearest',
    align_corners=None,
    warning=True
):
    """Resize a tensor."""
    if size is not None:
        return F.interpolate(input, size=size, mode=mode, align_corners=align_corners)
    else:
        return F.interpolate(input, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class Scale(nn.Module):
    """Scale layer."""
    
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = jt.array([scale])
    
    def execute(self, x):
        return x * self.scale


def constant_init(module, val, bias=0):
    """Initialize module parameters with constant values."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """Initialize module parameters with normal distribution."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Initialize module parameters with Xavier initialization."""
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_gauss_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    """Initialize module parameters with Kaiming initialization."""
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.relu_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.relu_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# Alias for compatibility
trunc_normal_init = trunc_normal_ 