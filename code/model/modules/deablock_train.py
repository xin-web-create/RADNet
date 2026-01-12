import torch
from torch import nn

from .nfcconv  import NFConv
from .ca import SpatialAttention, ChannelAttention, PixelAttention


class CSPCA(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8):
        super(CSPCA, self).__init__()
        self.conv1 = NFConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)


        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

        self.sa = SpatialAttention(dim)
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.act1(res)
        res = res + x

        res = self.conv2(res)
        res = self.bn2(res)

        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)

        res = res * pattn2


        res = torch.clamp(res, -50, 50)

        res = res + x
        return res


class HorizontalCellTrain(nn.Module):
    def __init__(self, dim, groups=1, eps=1e-5):
        super(HorizontalCellTrain, self).__init__()
        self.groups = groups
        self.groups_dim = dim // groups
        self.eps = eps

        self.pre_norm = nn.GroupNorm(num_groups=min(4, dim // 4), num_channels=dim)
        self.nf_conv = NFConv(dim)
        self.residual_scale = nn.Parameter(torch.ones(1))  # 可学习的缩放因子

    def forward(self, x):
        identity = x
        x = self.pre_norm(x)
        x = self.nf_conv(x)
        return identity + self.residual_scale * x