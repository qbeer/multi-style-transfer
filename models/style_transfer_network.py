import torch
import torch.nn.functional as F
from .residual_block import ResidualBlock
from .upsampling_block import UpsamplingBlock
from .categorical_conditional_instance_normalization import CategoricalConditionalInstanceNorm2d


class FastStylizationNetwork(torch.nn.Module):
    def __init__(self, n_styles):
        super(FastStylizationNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (9, 9), stride=1, padding=0)
        self.padding1 = (4, 4, 4, 4)
        self.instance_norm1 = CategoricalConditionalInstanceNorm2d(
            32, n_styles)
        self.conv2 = torch.nn.Conv2d(32, 64, (3, 3), stride=2, padding=0)
        self.padding2 = (0, 1, 0, 1)
        self.instance_norm2 = CategoricalConditionalInstanceNorm2d(
            64, n_styles)
        self.conv3 = torch.nn.Conv2d(64, 128, (3, 3), stride=2, padding=0)
        self.padding3 = (0, 1, 0, 1)
        self.instance_norm3 = CategoricalConditionalInstanceNorm2d(
            128, n_styles)

        self.residual1 = ResidualBlock(128, n_styles)
        self.residual2 = ResidualBlock(128, n_styles)
        self.residual3 = ResidualBlock(128, n_styles)
        self.residual4 = ResidualBlock(128, n_styles)
        self.residual5 = ResidualBlock(128, n_styles)

        self.upsampling1 = UpsamplingBlock(128, 64)
        self.upsampling2 = UpsamplingBlock(64, 32)

        self.last_conv = torch.nn.Conv2d(32, 3, (9, 9), stride=1, padding=0)
        self.padding_last = (4, 4, 4, 4)
        self.instance_norm_last = CategoricalConditionalInstanceNorm2d(
            3, n_styles)

    def forward(self, x, styles):
        x = self.conv1(F.pad(x, self.padding1, mode='reflect')).clamp(min=0)
        x = self.instance_norm1(x, styles)
        x = self.conv2(F.pad(x, self.padding2, mode='reflect')).clamp(min=0)
        x = self.instance_norm2(x, styles)
        x = self.conv3(F.pad(x, self.padding3, mode='reflect')).clamp(min=0)
        x = self.instance_norm3(x, styles)

        x = self.residual1(x, styles)
        x = self.residual2(x, styles)
        x = self.residual3(x, styles)
        x = self.residual4(x, styles)
        x = self.residual5(x, styles)

        x = self.upsampling1(x)
        x = self.upsampling2(x)

        x = self.last_conv(F.pad(x, self.padding_last, mode='reflect'))
        x = self.instance_norm_last(x, styles)

        return torch.sigmoid(x)
