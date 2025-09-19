import torch
import torch.nn as nn
import config

# Larger Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out

# Deep CNN for high-res image restoration
class CNN_SuperRes_Deep(nn.Module):
    def __init__(self, in_ch=config.IN_CHANS, out_ch=config.OUT_CHANS,
                 num_features=64, num_blocks=24):  # 24 blocks for depth
        super().__init__()
        self.entry = nn.Conv2d(in_ch, num_features, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.exit = nn.Conv2d(num_features, out_ch, 3, 1, 1)

    def forward(self, x):
        out = self.entry(x)
        res = out
        out = self.res_blocks(out)
        out = self.exit(out)
        return out + x  # global residual

def create_model():
    return CNN_SuperRes_Deep()
