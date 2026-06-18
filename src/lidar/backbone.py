# Source: Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds," CVPR 2019.

import torch
import torch.nn as nn


def conv_block(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def deconv_block(in_c, out_c, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=stride, stride=stride, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class PillarBackbone(nn.Module):
    """
    Network1 downsamples the pseudo-image in three stages.
    Network2 upsamples each stage back to H/2, W/2 and concatenates.

    input -> 1a -> 2a -> 3a
              |     |     |
             1b    2b    3b
              |     |     |
              └─────┴─────┘
                  concat
              (6C, H/2, W/2)
    """
    def __init__(self, C: int = 64):
        super().__init__()

        self.conv_1a = conv_block(C, C, stride=2)
        self.conv_2a = conv_block(C, C*2, stride=2)
        self.conv_3a = conv_block(C*2, C*4, stride=2)

        self.deconv_1b = deconv_block(C, C*2, stride=1)
        self.deconv_2b = deconv_block(C*2, C*2, stride=2)
        self.deconv_3b = deconv_block(C*4, C*2, stride=4)

    def forward(self, x):
        out_1a = self.conv_1a(x)
        out_2a = self.conv_2a(out_1a)
        out_3a = self.conv_3a(out_2a)

        out_1b = self.deconv_1b(out_1a)
        out_2b = self.deconv_2b(out_2a)
        out_3b = self.deconv_3b(out_3a)

        return torch.cat([out_1b, out_2b, out_3b], dim=1)
