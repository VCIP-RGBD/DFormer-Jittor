"""
DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
Code: https://github.com/VCIP-RGBD/DFormer

Author: yinbow
Email: bowenyin@mail.nankai.edu.cn

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import jittor as jt
from jittor import nn
import jittor.nn as F
import math
from typing import List, Tuple
from collections import OrderedDict


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Initialize tensor with truncated normal distribution."""
    # TODO: Implement proper truncated normal initialization
    nn.init.gauss_(tensor, mean, std)
    return tensor


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


def load_state_dict(model, state_dict, strict=True):
    """Load state dict to model."""
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


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    """Load checkpoint from file."""
    try:
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


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D tensors."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def execute(self, x):
        """
        input shape (b c h w)
        """
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    """Depthwise convolution 2d."""

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def execute(self, x):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def execute(self, x):
        """
        x: B H W C
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


def angle_transform(x, sin, cos):
    """Apply angle transformation for positional encoding."""
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (jt.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    """Geometry Prior Generator for depth-aware attention."""

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** jt.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = jt.ones(2, 1, 1, 1)
        decay = jt.log(
            1 - 2 ** (-initial_value - heads_range * jt.arange(num_heads, dtype=jt.float32) / num_heads)
        )
        # Register as buffers (non-trainable parameters)
        self.angle = angle
        self.decay = decay

    def generate_depth_decay(self, H, W, depth_grid):
        """Generate 2d depth decay mask."""
        B, C, H_grid, W_grid = depth_grid.shape
        # Take the first channel if multiple channels exist
        if C > 1:
            depth_grid = depth_grid[:, 0:1, :, :]  # [B, 1, H, W]
        grid_d = depth_grid.reshape(B, H_grid * W_grid, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H, W):
        """Generate 2d position decay mask."""
        index_h = jt.arange(H)
        index_w = jt.arange(W)
        grid_h, grid_w = jt.meshgrid(index_h, index_w)
        grid = jt.stack([grid_h, grid_w], dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """Generate 1d depth decay mask."""
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        return mask

    def generate_1d_decay(self, l):
        """Generate 1d decay mask."""
        index = jt.arange(l)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def execute(self, HW_tuple, depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:
            index = jt.arange(HW_tuple[0] * HW_tuple[1])
            sin = jt.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = jt.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = jt.arange(HW_tuple[0] * HW_tuple[1])
            sin = jt.sin(index[:, None] * self.angle[None, :])
            cos = jt.cos(index[:, None] * self.angle[None, :])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask_p = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask = self.weight[0] * mask_p.unsqueeze(0) + self.weight[1] * mask_d
            geo_prior = ((sin, cos), mask)

        return geo_prior


class Decomposed_GSA(nn.Module):
    """Decomposed Geometry Self-Attention module."""

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.value_factor = value_factor

        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim * value_factor)
        self.to_out = nn.Linear(embed_dim * value_factor, embed_dim)

    def execute(self, x, rel_pos, split_or_not=False):
        """
        x: B, H, W, C
        rel_pos: relative position encoding
        """
        B, H, W, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        if split_or_not:
            # Decomposed attention for efficiency
            ((sin_h, cos_h), (mask_h, mask_w)) = rel_pos
            
            # Split into height and width dimensions
            q_h = q.reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            k_h = k.reshape(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
            v_h = v.reshape(B, H, W, self.num_heads, self.head_dim * self.value_factor).permute(0, 3, 1, 2, 4)
            
            # Height-wise attention
            scale = self.head_dim ** -0.5
            attn_h = (q_h @ k_h.transpose(-2, -1)) * scale
            attn_h = attn_h + mask_h
            attn_h = F.softmax(attn_h, dim=-1)
            
            out_h = attn_h @ v_h
            out = out_h.permute(0, 2, 3, 1, 4).reshape(B, H, W, C * self.value_factor)
            out = self.to_out(out)
        else:
            (sin, cos), mask = rel_pos
            q = q.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(B, H * W, self.num_heads, self.head_dim * self.value_factor).permute(0, 2, 1, 3)

            # Apply rotary positional encoding
            q = q * cos + jt.stack([-q[:, :, :, 1::2], q[:, :, :, ::2]], dim=-1).flatten(-2) * sin
            k = k * cos + jt.stack([-k[:, :, :, 1::2], k[:, :, :, ::2]], dim=-1).flatten(-2) * sin

            # Compute attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn + mask
            attn = F.softmax(attn, dim=-1)

            out = attn @ v
            out = out.permute(0, 2, 1, 3).reshape(B, H, W, C * self.value_factor)
            out = self.to_out(out)

        return out

    def reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Full_GSA(nn.Module):
    """Full Geometry Self-Attention module."""

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.value_factor = value_factor

        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim * value_factor)
        self.to_out = nn.Linear(embed_dim * value_factor, embed_dim)

    def execute(self, x, rel_pos, split_or_not=False):
        """
        x: B, H, W, C
        rel_pos: relative position encoding
        """
        B, H, W, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        (sin, cos), mask = rel_pos
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, self.head_dim * self.value_factor).permute(0, 2, 1, 3)

        # Apply rotary positional encoding
        q = q * cos + jt.stack([-q[:, :, :, 1::2], q[:, :, :, ::2]], dim=-1).flatten(-2) * sin
        k = k * cos + jt.stack([-k[:, :, :, 1::2], k[:, :, :, ::2]], dim=-1).flatten(-2) * sin

        # Compute attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, H, W, C * self.value_factor)
        out = self.to_out(out)

        return out

    def reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network module."""

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.activation_fn = activation_fn
        self.subln = subln
        self.subconv = subconv

        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
        if subconv:
            self.dwconv = DWConv2d(ffn_dim, 3, 1, 1)

        # Implement dropout layers
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
            
        if activation_dropout > 0:
            self.activation_dropout = nn.Dropout(activation_dropout)
        else:
            self.activation_dropout = nn.Identity()

    def reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def execute(self, x):
        """
        x: B, H, W, C
        """
        residual = x
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        
        if self.subconv:
            x = self.dwconv(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class RGBD_Block(nn.Module):
    """RGBD Block for DFormerv2."""

    def __init__(
        self,
        split_or_not,
        embed_dim,
        num_heads,
        ffn_dim,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=2,
        heads_range=4,
    ):
        super().__init__()
        self.split_or_not = split_or_not
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.layerscale = layerscale

        self.geo_prior_gen = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)
        
        if split_or_not:
            self.attn = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.attn = Full_GSA(embed_dim, num_heads)

        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Implement DropPath
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        if layerscale:
            self.gamma1 = jt.full((embed_dim,), layer_init_values)
            self.gamma2 = jt.full((embed_dim,), layer_init_values)

    def execute(self, x, x_e, split_or_not=False):
        """
        x: RGB features [B, H, W, C]
        x_e: Depth features [B, H, W, C]
        """
        # Generate geometry prior
        B, H, W, C = x.shape
        depth_map = x_e.permute(0, 3, 1, 2)  # Convert to BCHW
        geo_prior = self.geo_prior_gen((H, W), depth_map, split_or_not)

        # Self-attention
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, geo_prior, split_or_not)
        if self.layerscale:
            x = self.gamma1 * x
        x = shortcut + self.drop_path(x)

        # Feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        if self.layerscale:
            x = self.gamma2 * x
        x = shortcut + self.drop_path(x)

        return x, x_e


class BasicLayer(nn.Module):
    """Basic layer for DFormerv2."""

    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value,
        heads_range,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        split_or_not=False,
        downsample=None,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            RGBD_Block(
                split_or_not=split_or_not,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=int(embed_dim * ffn_dim),
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                layerscale=layerscale,
                layer_init_values=layer_init_values,
                init_value=init_value,
                heads_range=heads_range,
            )
            for i in range(depth)
        ])

        # Downsample layer
        if downsample is not None:
            self.downsample = downsample(embed_dim, out_dim, norm_layer)
        else:
            self.downsample = None

    def execute(self, x, x_e):
        """
        x: RGB features
        x_e: Depth features
        """
        for block in self.blocks:
            x, x_e = block(x, x_e)

        if self.downsample is not None:
            x_down = self.downsample(x)
            x_e_down = self.downsample(x_e)
            return x, x_down, x_e, x_e_down
        else:
            return x, x, x_e, x_e


class dformerv2(nn.Module):
    """DFormerv2 model for RGBD semantic segmentation."""

    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        projection=1024,
        norm_cfg=None,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
        norm_eval=True,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.norm_eval = norm_eval

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0])
        self.patch_embed_e = PatchEmbed(in_chans=1, embed_dim=embed_dims[0])

        # Calculate drop path rates
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            split_or_not = i < 2  # Use decomposed attention for first two layers
            layer = BasicLayer(
                embed_dim=embed_dims[i],
                out_dim=embed_dims[i + 1] if i < len(depths) - 1 else embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                init_value=init_values[i],
                heads_range=heads_ranges[i],
                ffn_dim=mlp_ratios[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                split_or_not=split_or_not,
                downsample=PatchMerging if i < len(depths) - 1 else None,
                layerscale=layerscales[i],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        # Layer normalization
        self.norm = LayerNorm2d(embed_dims[-1])

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained is not None:
            try:
                checkpoint = load_checkpoint(self, pretrained, strict=False)
                print(f"Loaded checkpoint from {pretrained}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        else:
            self.apply(self._init_weights)

    def no_weight_decay(self):
        """Return parameters that should not use weight decay."""
        return {'pos_embed', 'cls_token'}

    def no_weight_decay_keywords(self):
        """Return parameter keywords that should not use weight decay."""
        return {'relative_position_bias_table', 'angle', 'decay'}

    def execute(self, x, x_e):
        """
        x: RGB input [B, 3, H, W]
        x_e: Depth input [B, 1, H, W]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, H/4, W/4, C]
        x_e = self.patch_embed_e(x_e)  # [B, H/4, W/4, C]

        outs = []
        for i, layer in enumerate(self.layers):
            if layer.downsample is not None:
                x_out, x, x_e_out, x_e = layer(x, x_e)
            else:
                x_out, x, x_e_out, x_e = layer(x, x_e)

            if i in self.out_indices:
                # Convert back to BCHW format
                x_out = x_out.permute(0, 3, 1, 2)
                outs.append(x_out)

        return outs

    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def DFormerv2_S(pretrained=False, **kwargs):
    """DFormerv2 Small model."""
    model = dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        **kwargs
    )
    if pretrained:
        model.init_weights(pretrained)
    return model


def DFormerv2_B(pretrained=False, **kwargs):
    """DFormerv2 Base model."""
    model = dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[2, 2, 18, 2],
        num_heads=[5, 5, 10, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        **kwargs
    )
    if pretrained:
        model.init_weights(pretrained)
    return model


def DFormerv2_L(pretrained=False, **kwargs):
    """DFormerv2 Large model."""
    model = dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[2, 2, 18, 2],
        num_heads=[7, 7, 14, 20],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        **kwargs
    )
    if pretrained:
        model.init_weights(pretrained)
    return model 