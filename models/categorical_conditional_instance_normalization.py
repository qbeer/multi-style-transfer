import torch


class CategoricalConditionalInstanceNorm2d(torch.nn.Module):
    def __init__(self, num_features, n_styles):
        super(CategoricalConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.instance_norm = torch.nn.InstanceNorm2d(num_features,
                                                     affine=False)
        self.embed = torch.nn.Embedding(n_styles, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.instance_norm(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)
        return out
