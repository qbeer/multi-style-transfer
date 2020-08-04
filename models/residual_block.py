import torch
import torch.nn.functional as F
from .categorical_conditional_instance_normalization import CategoricalConditionalInstanceNorm2d


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, n_styles):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels,
                                     channels,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=0)
        self.padding1 = (1, 1, 1, 1)
        self.instance_norm1 = CategoricalConditionalInstanceNorm2d(channels, n_styles)
        self.conv2 = torch.nn.Conv2d(channels,
                                     channels,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=0)
        self.padding2 = (1, 1, 1, 1)
        self.instance_norm2 = CategoricalConditionalInstanceNorm2d(channels, n_styles)

    def forward(self, x, styles):
        conv_relu = self.conv1(F.pad(x, self.padding1,
                                     mode='reflect')).clamp(min=0)
        conv_relu = self.instance_norm1(conv_relu, styles)
        conv_linear = self.conv2(
            F.pad(conv_relu, self.padding2, mode='reflect'))
        conv_linear = self.instance_norm2(conv_linear, styles)
        return conv_linear + x