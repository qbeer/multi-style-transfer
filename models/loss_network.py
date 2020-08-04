import torch
from collections import namedtuple


class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.6.0',
                               'vgg16_bn',
                               pretrained=True)
        features = list(model.features)
        self.features = torch.nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ind, model in enumerate(self.features):
            x = model(x)
            if ind in {5, 12, 16, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs",
                                 ['relu1', 'relu2', 'relu3', 'relu4'])
        return vgg_outputs(*results)