
import torch
from einops.layers.torch import Rearrange
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()

        self.dc1 = nn.Conv2d(2, 1, 3, padding=1, dilation=1, bias=True)
        self.dc3 = nn.Conv2d(2, 1, 3, padding=3, dilation=3, bias=True)
        self.dc5 = nn.Conv2d(2, 1, 3, padding=5, dilation=5, bias=True)

        self.conv1x1 = nn.Conv2d(3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)


        dc1_out = self.dc1(x2)
        dc3_out = self.dc3(x2)
        dc5_out = self.dc5(x2)


        dc_concat = torch.cat([dc1_out, dc3_out, dc5_out], dim=1)


        sattn = self.conv1x1(dc_concat)
        sattn = self.sigmoid(sattn)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.gap_branch = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=True),
        )

        self.ds_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True),  # depthwise
            nn.Conv2d(dim, dim, 1, bias=True),  # pointwise
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        gap_out = self.gap(x)
        gap_attn = self.gap_branch(gap_out)


        ds_attn = self.ds_conv(x)
        ds_attn = torch.mean(ds_attn, dim=[2, 3], keepdim=True)  # 全局平均


        cattn = gap_attn + ds_attn
        cattn = self.sigmoid(cattn)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()

        if dim % 8 == 0:
            groups = 8
        elif dim % 16 == 0:
            groups = 16
        elif dim % 32 == 0:
            groups = 32
        else:

            import math
            groups = math.gcd(dim, 8)

        self.groups = groups
        print(f"PixelAttention: dim={dim}, using groups={groups}")


        self.pa1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid()
        )

        self.channel_shuffle = Rearrange('b (g c) h w -> b (c g) h w', g=groups)
        self.group_conv = nn.Conv2d(2 * dim, dim, 7, padding=3,
                                    groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape


        x_expanded = x.unsqueeze(dim=2)
        pattn1_expanded = pattn1.unsqueeze(dim=2)
        combined = torch.cat([x_expanded, pattn1_expanded], dim=2)
        combined = Rearrange('b c t h w -> b (c t) h w')(combined)


        shuffled = self.channel_shuffle(combined)


        pattn2 = self.group_conv(shuffled)
        pattn2 = self.sigmoid(pattn2)

        return pattn2