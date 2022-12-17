# Basic model parts for CNN

import torch
import torch.nn as nn
import torch.nn.functional as F


# Res block
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3D, self).__init__()
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch)
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_new = self.double_conv3d(x)
        x_shortcut = self.shortcut(x)
        x = self.out_relu(x_new + x_shortcut)
        return x


class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock3D, self).__init__()
        self.down_block3d = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.down_block3d(x)
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut_ch):
        super(UpBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch//2 + shortcut_ch, out_ch)

    def forward(self, x, x_down):
        x = self.up(x)
        # padding for misalignment
        dD = x_down.size()[2] - x.size()[2]
        dH = x_down.size()[3] - x.size()[3]
        dW = x_down.size()[4] - x.size()[4]
        if dD != 0 or dH != 0 or dW != 0:
            print('Warning: padding is used during upsampling!')
        x = F.pad(x, [dW//2, dW-dW//2, dH//2, dH-dH//2, dD//2, dD-dD//2])
        x = torch.cat([x_down, x], dim=1)
        x = self.conv(x)
        return x

