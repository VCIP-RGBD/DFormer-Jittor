import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import jittor as jt
from jittor import nn
import jittor.nn as F
import math


def build_norm_layer(norm_cfg, num_features):
    """Build normalization layer."""
    if norm_cfg is None:
        return None, nn.Identity()
    
    norm_type = norm_cfg.get('type', 'BN')
    if norm_type in ['BN', 'BatchNorm2d']:
        return None, nn.BatchNorm2d(num_features)
    elif norm_type in ['SyncBN', 'SyncBatchNorm2d']:
        return None, nn.BatchNorm2d(num_features)
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
        return nn.ReLU()
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(act_cfg.get('negative_slope', 0.01))
    elif act_type == 'Swish':
        return nn.Swish()
    else:
        return nn.ReLU()


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
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
        dropout=0.0,
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
            layers.append(self.activate)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = feedforward_channels
        
        # Final layer
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
    
    def execute(self, x):
        """Execute function."""
        out = self.layers(x)
        if self.add_identity:
            return x + out
        return out


def load_state_dict(model, state_dict, strict=True):
    """Load state dict to model."""
    # TODO: Implement proper state dict loading for Jittor
    missing_keys = []
    unexpected_keys = []
    
    model_state_dict = model.state_dict()
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            try:
                model_state_dict[key].assign(value)
            except:
                print(f"Failed to load parameter {key}")
                missing_keys.append(key)
        else:
            unexpected_keys.append(key)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(f"Error loading state dict: missing {len(missing_keys)} keys, unexpected {len(unexpected_keys)} keys")


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Initialize tensor with truncated normal distribution."""
    # TODO: Implement proper truncated normal initialization
    nn.init.gauss_(tensor, mean, std)
    return tensor


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = jt.ones(normalized_shape)
        self.bias = jt.zeros(normalized_shape)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def execute(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / jt.sqrt(s + self.eps)
            x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return x


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
        **kwargs
    ):
        super(ConvModule, self).__init__()
        
        # Build conv layer
        self.conv = nn.Conv2d(
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
    
    def execute(self, x):
        """Execute function."""
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class MLP(nn.Module):
    """MLP module with depthwise convolution for position encoding."""

    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type="SyncBN", requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def execute(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)
        return x


class attention(nn.Module):
    """Attention module for DFormer."""

    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type="SyncBN", requires_grad=True), drop_depth=False):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim // 2)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2)
        self.e_fore = nn.Linear(dim // 2, dim // 2)
        self.e_back = nn.Linear(dim // 2, dim // 2)

        self.proj = nn.Linear(dim // 2 * 3, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim // 2 * 3, dim // 2)
        if window != 0:
            self.short_cut_linear = nn.Linear(dim // 2 * 3, dim // 2)
            self.kv = nn.Linear(dim, dim)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            self.proj = nn.Linear(dim * 2, dim)
            if not drop_depth:
                self.proj_e = nn.Linear(dim * 2, dim // 2)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim // 2, eps=1e-6, data_format="channels_last")
        self.drop_depth = drop_depth

    def execute(self, x, x_e):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        if self.window != 0:
            short_cut = jt.concat([x, x_e], dim=3)
            short_cut = short_cut.permute(0, 3, 1, 2)

        q = self.q(x)
        cutted_x = self.q_cut(x)
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)

        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            b = x.permute(0, 2, 3, 1)
            kv = self.kv(b)
            kv = kv.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            short_cut = self.pool(short_cut).permute(0, 2, 3, 1)
            short_cut = self.short_cut_linear(short_cut)
            short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
            m = short_cut
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = (
                (attn @ v)
                .reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2)
                .permute(0, 1, 4, 2, 3)
                .reshape(B, C // 2, self.window, self.window)
            )
            attn = F.interpolate(attn, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        cutted_x = cutted_x * x_e
        x = q * a

        if self.window != 0:
            x = jt.concat([x, attn, cutted_x], dim=3)
        else:
            x = jt.concat([x, cutted_x], dim=3)
        if not self.drop_depth:
            x_e = self.proj_e(x)
        x = self.proj(x)

        return x, x_e


class Block(nn.Module):
    """Basic block for DFormer."""

    def __init__(
        self,
        index,
        dim,
        num_head,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        mlp_ratio=4.0,
        block_index=0,
        last_block_index=50,
        window=7,
        dropout_layer=None,
        drop_depth=False,
    ):
        super().__init__()

        self.index = index
        layer_scale_init_value = 1e-6
        if block_index > last_block_index:
            window = 0
        self.attn = attention(dim, num_head, window=window, norm_cfg=norm_cfg, drop_depth=drop_depth)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.dropout_layer = build_dropout(dropout_layer)

        self.layer_scale_1 = jt.ones((dim,)) * layer_scale_init_value
        self.layer_scale_2 = jt.ones((dim,)) * layer_scale_init_value

        if not drop_depth:
            self.layer_scale_1_e = jt.ones((dim // 2,)) * layer_scale_init_value
            self.layer_scale_2_e = jt.ones((dim // 2,)) * layer_scale_init_value
            self.mlp_e2 = MLP(dim // 2, mlp_ratio)
        self.drop_depth = drop_depth

    def execute(self, x, x_e):
        res_x, res_e = x, x_e
        x, x_e = self.attn(x, x_e)

        x = res_x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x)
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))
        
        if not self.drop_depth:
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))

        return x, x_e


class DFormer(BaseModule):
    """DFormer encoder for RGBD semantic segmentation."""

    def __init__(
        self,
        in_channels=4,
        depths=(2, 2, 8, 2),
        dims=(32, 64, 128, 256),
        out_indices=(0, 1, 2, 3),
        windows=[7, 7, 7, 7],
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        mlp_ratios=[8, 8, 4, 4],
        num_heads=(2, 4, 10, 16),
        last_block=[50, 50, 50, 50],
        drop_path_rate=0.1,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        print(drop_path_rate)
        self.depths = depths
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        
        # Build downsample layers (stems and downsampling)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
        )

        self.downsample_layers_e = nn.ModuleList()
        stem_e = nn.Sequential(
            nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 4),
            nn.GELU(),
            nn.Conv2d(dims[0] // 4, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
        )

        self.downsample_layers.append(stem)
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims) - 1):
            stride = 2
            downsample_layer = nn.Sequential(
                build_norm_layer(norm_cfg, dims[i])[1],
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                build_norm_layer(norm_cfg, dims[i] // 2)[1],
                nn.Conv2d(dims[i] // 2, dims[i + 1] // 2, kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        # Build stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(*[
                Block(
                    index=cur + j,
                    dim=dims[i],
                    window=windows[i],
                    dropout_layer=dict(type="DropPath", drop_prob=dp_rates[cur + j]),
                    num_head=num_heads[i],
                    norm_cfg=norm_cfg,
                    block_index=depths[i] - j,
                    last_block_index=last_block[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_depth=((i == 3) & (j == depths[i] - 1)),
                )
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone."""
        if pretrained is not None:
            try:
                # Load pretrained weights
                print(f"Loading pretrained weights from {pretrained}")
                # TODO: Implement proper weight loading
                pass
            except Exception as e:
                print(f"Failed to load pretrained weights: {e}")
        else:
            # Initialize weights using default initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def execute(self, x, x_e):
        """Forward function."""
        if x_e is None:
            x_e = x
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(x_e.shape) == 3:
            x_e = x_e.unsqueeze(0)

        x_e = x_e[:, 0, :, :].unsqueeze(1)

        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)

            x = x.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)
            for blk in self.stages[i]:
                x, x_e = blk(x, x_e)
            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
            outs.append(x)
        return outs, None


def load_model_weights(model, model_type, kwargs):
    """Load model weights - placeholder for compatibility."""
    print(f"Loading {model_type} weights")
    return model


def DFormer_Tiny(pretrained=False, **kwargs):
    """DFormer Tiny model."""
    model = DFormer(
        dims=[32, 64, 128, 256],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 3, 5, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        **kwargs,
    )
    if pretrained:
        model = load_model_weights(model, "DFormer_Tiny", kwargs)
    return model


def DFormer_Small(pretrained=False, **kwargs):
    """DFormer Small model."""
    model = DFormer(
        dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 4, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        **kwargs,
    )
    if pretrained:
        model = load_model_weights(model, "DFormer_Small", kwargs)
    return model


def DFormer_Base(pretrained=False, drop_path_rate=0.1, **kwargs):
    """DFormer Base model."""
    model = DFormer(
        dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 3, 12, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    if pretrained:
        model = load_model_weights(model, "DFormer_Base", kwargs)
    return model


def DFormer_Large(pretrained=False, drop_path_rate=0.1, **kwargs):
    """DFormer Large model."""
    model = DFormer(
        dims=[96, 192, 288, 576],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 3, 12, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    if pretrained:
        model = load_model_weights(model, "DFormer_Large", kwargs)
    return model 