import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ONConv(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ONConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=True)


        with torch.no_grad():
            center = kernel_size // 2
            kernel = torch.full((kernel_size, kernel_size), -1 / (kernel_size ** 2 - 1))
            kernel[center, center] = 2.0
            kernel = kernel.expand(out_channels, in_channels, -1, -1)
            self.conv.weight.copy_(kernel)
            self.conv.bias.data.zero_()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class OFFConv(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(OFFConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=True)


        with torch.no_grad():
            center = kernel_size // 2
            kernel = torch.full((kernel_size, kernel_size), 0.5 / (kernel_size ** 2 - 1))
            kernel[center, center] = -1.5
            kernel = kernel.expand(out_channels, in_channels, -1, -1)
            self.conv.weight.copy_(kernel)
            self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


class Conv2d_cd(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta
        self.kernel_size = kernel_size

    def forward(self, x):
        if math.fabs(self.theta - 0.0) < 1e-8:
            return self.conv(x)
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape


            kernel_cd = torch.zeros_like(conv_weight)
            kernel_cd[:, :, :, :] = conv_weight[:, :, :, :]
            center = self.kernel_size // 2
            kernel_cd[:, :, center, center] = conv_weight[:, :, center, center] - \
                                              torch.sum(conv_weight, dim=(2, 3))

            out = F.conv2d(x, kernel_cd, bias=self.conv.bias,
                           stride=self.conv.stride, padding=self.conv.padding,
                           dilation=self.conv.dilation, groups=self.conv.groups)
            return out




class NFConv(nn.Module):


    def __init__(self, dim):
        super(NFConv, self).__init__()
        self.dim = dim

        self.on_conv = ONConv(dim, dim)
        self.off_conv = OFFConv(dim, dim)
        self.cd_conv = Conv2d_cd(dim, dim)
        self.normal_conv = nn.Conv2d(dim, dim, 3, padding=1, bias=True)


        self.register_buffer('weights', torch.tensor([0.1, 0.1, 0.1, 0.7]))


        self.bn = nn.BatchNorm2d(dim)

        nn.init.kaiming_normal_(self.normal_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.normal_conv.bias is not None:
            nn.init.constant_(self.normal_conv.bias, 0)

    def forward(self, x):
        identity = x

        on_out = self.on_conv(x)
        off_out = self.off_conv(x)
        cd_out = self.cd_conv(x)
        normal_out = self.normal_conv(x)


        w = self.weights
        combined = (w[0] * on_out + w[1] * off_out +
                    w[2] * cd_out + w[3] * normal_out)


        combined = self.bn(combined)
        out = 0.1 * combined + identity

        return out