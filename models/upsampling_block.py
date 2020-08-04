import torch
import torch.nn.functional as F


class UpsamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=(3, 3),
                                    stride=1,
                                    padding=0)
        self.padding = (1, 1, 1, 1)
        self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        upsamp = F.interpolate(x, mode='nearest', scale_factor=2)
        conv_relu = self.conv(F.pad(upsamp, self.padding,
                                    mode='reflect')).clamp(min=0)
        outp = self.instance_norm(conv_relu)
        return outp