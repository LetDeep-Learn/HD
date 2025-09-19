"""
Lightweight SwinIR-like model implementation in PyTorch.

Patched:
- Robust window partition / reverse with explicit shape computation.
- WindowAttention.forward: compute head_dim from runtime C, assert divisibility, safe qkv reshape and checks.
- SwinBlock.forward: explicit partitioning, asserts for divisibility, clearer reshapes.
- SwinIRLite.forward: model-side padding to make H,W multiples of window_size and crop output back to original size.
- Minor safety/typing fixes (meshgrid indexing, long buffers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config

# ---------- Utilities ----------
def window_partition(x, window_size):
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"H ({H}) and W ({W}) must be divisible by window_size ({window_size})"
    num_h = H // window_size
    num_w = W // window_size
    x = x.view(B, C, num_h, window_size, num_w, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nH, nW, ws, ws, C
    windows = x.view(B * num_h * num_w, window_size * window_size, C)  # (num_windows*B), ws*ws, C
    return windows

def window_reverse(windows, window_size, H, W):
    Bn, Ws2, C = windows.shape
    assert Ws2 == window_size * window_size, "window size mismatch"
    num_h = H // window_size
    num_w = W // window_size
    b = Bn // (num_h * num_w)
    x = windows.view(b, num_h, num_w, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, nH, ws, nW, ws
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

        # qkv and proj
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # relative position index buffer
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, ws, ws
        coords_flatten = coords.flatten(1)  # 2, ws*ws
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, ws*ws, ws*ws
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, ws*ws, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1).long()  # ws*ws, ws*ws
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x: (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        # ensure heads divide channels
        head_dim = C // self.num_heads
        if self.num_heads * head_dim != C:
            raise RuntimeError(f"num_heads ({self.num_heads}) must divide C ({C})")

        # qkv
        qkv = self.qkv(x)  # (B_, N, 3*C)
        expected = B_ * N * 3 * C
        if qkv.numel() != expected:
            raise RuntimeError(f"qkv element count mismatch: got {qkv.numel()}, expected {expected}")
        qkv = qkv.view(B_, N, 3, self.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q = q * (head_dim ** -0.5)

        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        # relative position bias
        rp = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # N,N,num_heads
        rp = rp.permute(2, 0, 1).unsqueeze(0)  # 1, num_heads, N, N
        attn = attn + rp

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
        assert Hc == H and Wc == W, f"Input spatial dims ({Hc},{Wc}) != provided H,W ({H},{W})"

        # ensure divisibility
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise RuntimeError(f"H ({H}) and W ({W}) must be divisible by window_size ({self.window_size}). "
                               "Pad inputs to multiple of window_size before calling the model.")

        shortcut = x
        # partition into windows explicitly
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        num_h = H // self.window_size
        num_w = W // self.window_size
        # reshape to B, nH, ws, nW, ws, C
        x_windows = x_perm.view(B, num_h, self.window_size, num_w, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, nH, nW, ws, ws, C
        x_windows = x_windows.view(B * num_h * num_w, self.window_size * self.window_size, C)  # (B*nH*nW, ws*ws, C)

        x_windows = self.norm1(x_windows)
        x_windows = self.attn(x_windows)
        x_windows = self.norm2(x_windows)
        x_windows = self.mlp(x_windows)

        # reverse windows
        x_windows = x_windows.view(B, num_h, num_w, self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, nH, ws, nW, ws
        x_windows = x_windows.view(B, C, H, W)

        x = x_windows + shortcut
        return x

# ---------- The full model ----------
class SwinIRLite(nn.Module):
    """
    SwinIR-like architecture for image restoration (same resolution input->output).
    """

    def __init__(self,
             in_chans=3,
             embed_dim=768,
             depths=(6,6,6),
             num_heads=(12,12,12),
             window_size=8,
             mlp_ratio=2.0,
             img_size=1024):
        super().__init__()
        self.img_size = img_size
        self.window_size = window_size

        # simple validation for divisibility of embed_dim by num_heads
        for i, nh in enumerate(num_heads):
            if embed_dim % nh != 0:
                raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads[{i}] ({nh})")

        self.embed_conv = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.deep_features = nn.ModuleList([
            self._make_residual_group(embed_dim, depths[i], num_heads[i], window_size, mlp_ratio)
            for i in range(len(depths))
        ])
        self.recon_conv = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

    def _make_residual_group(self, dim, depth, num_heads, window_size, mlp_ratio):
        layers = []
        for _ in range(depth):
            layers.append(SwinBlock(dim, num_heads, window_size=window_size, mlp_ratio=mlp_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: B x C x H x W
        returns B x C x H x W
        """
        B, C, H, W = x.shape
        # model-side padding to make H,W divisible by window_size
        pad_h = (self.window_size - (H % self.window_size)) % self.window_size
        pad_w = (self.window_size - (W % self.window_size)) % self.window_size
        orig_h, orig_w = H, W
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H = H + pad_h
            W = W + pad_w

        feat = self.embed_conv(x)  # B, dim, H, W
        out = feat
        # apply sequential groups (each SwinBlock implementation expects H,W)
        for g in self.deep_features:
            res = out
            for block in g:
                out = block(out, H, W)
            out = out + res
        out = self.recon_conv(out)
        # match channels back to input; assume input range consistent with residual add
        # crop if we padded earlier
        if pad_h != 0 or pad_w != 0:
            out = out[..., :orig_h, :orig_w]
            x = x[..., :orig_h, :orig_w]
        out = out + x  # global residual
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
