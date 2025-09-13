"""
Lightweight SwinIR-like model implementation in PyTorch.

This is a simplified, readable implementation focusing on restoration (same input/output resolution).
It uses Swin Transformer blocks with windowed self-attention and residual connections.

Key components:
- PatchEmbed: Conv layer to project input space to embedding dim.
- Swin Transformer Blocks: window attention + MLP
- Residual groups + reconstruction head

This is intentionally smaller than the full SwinIR paper to be CPU-friendly in inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import config

# ---------- Utilities ----------
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nH, nW, ws, ws, C
    windows = x.view(-1, window_size * window_size, C)  # (num_windows*B), ws*ws, C
    return windows

def window_reverse(windows, window_size, H, W):
    Bn, Ws2, C = windows.shape
    b = int(Bn / ((H // window_size) * (W // window_size)))
    x = windows.view(b, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(b, C, H, W)
    return x

# ---------- Basic layers ----------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  # window_size: int
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # Relative position bias table (small memory footprint)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # index pair for relative positions
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_flatten = coords.flatten(1)  # 2, ws*ws
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, ws*ws, ws*ws
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, ws*ws, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # ws*ws, ws*ws
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x: (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B_, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # N,N,num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0, drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        B, C, Hc, Wc = x.shape
        assert Hc == H and Wc == W
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        x = x.view(B * H * W // (self.window_size * self.window_size), self.window_size * self.window_size, C)
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = self.mlp(x)
        # reverse and reshape
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        x = x + shortcut
        return x

# ---------- The full model ----------
class SwinIRLite(nn.Module):
    """
    SwinIR-like architecture for image restoration (same resolution input->output).
    Structure:
        - shallow feature extraction (conv)
        - multiple residual Swin groups (each containing several SwinBlocks)
        - reconstruction conv to output image
    """

    def __init__(self,
                 in_chans=3,
                 embed_dim=60,
                 depths=(2,2,2),
                 num_heads=(3,6,12),
                 window_size=8,
                 mlp_ratio=2.0,
                 img_size=1024):
        super().__init__()
        self.img_size = img_size
        self.embed_conv = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.deep_features = nn.Sequential(*[
            self._make_residual_group(embed_dim, depths[i], num_heads[i], window_size, mlp_ratio)
            for i in range(len(depths))
        ])
        self.recon_conv = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

    def _make_residual_group(self, dim, depth, num_heads, window_size, mlp_ratio):
        layers = []
        for _ in range(depth):
            layers.append(SwinBlock(dim, num_heads, window_size=window_size, mlp_ratio=mlp_ratio))
        # A residual connection around the group
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: B x C x H x W (expected same H,W as img_size after padding)
        returns B x C x H x W
        """
        B, C, H, W = x.shape
        feat = self.embed_conv(x)  # B, dim, H, W
        out = feat
        # apply sequential groups (each SwinBlock implementation expects H,W)
        for g in self.deep_features:
            # manual residual for group
            res = out
            for block in g:
                out = block(out, H, W)
            out = out + res
        out = self.recon_conv(out)
        out = out + x  # global residual (image + residual)
        return out

# Convenience constructor
def create_model():
    m = SwinIRLite(in_chans=config.IN_CHANS,
                   embed_dim=config.EMBED_DIM,
                   depths=config.DEPTHS,
                   num_heads=config.NUM_HEADS,
                   window_size=config.WINDOW_SIZE,
                   mlp_ratio=config.MLP_RATIO,
                   img_size=config.TARGET_RESOLUTION)
    return m
