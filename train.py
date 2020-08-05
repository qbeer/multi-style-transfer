import torch
import numpy as np

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from batch_transfroms.batch_transforms import Normalize

from models import LossNetwork, FastStylizationNetwork
from data_loaders import art_loader, image_loader

import matplotlib.pyplot as plt
import torchvision

import torchvision.transforms as T

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs", we'll be more specific here
writer = SummaryWriter('runs/example')


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def style_loss(vgg_art_outputs, vgg_styled_outputs):
    # Calculating for relu1, relu2, relu3
    loss = 0.0

    _, channels, H, W = vgg_art_outputs.relu1.size()
    art_gram_relu1 = gram_matrix(vgg_art_outputs.relu1)
    styled_gram_relu1 = gram_matrix(vgg_styled_outputs.relu1)
    loss += .5 * torch.norm(art_gram_relu1 - styled_gram_relu1, p='fro')**2

    _, channels, H, W = vgg_art_outputs.relu2.size()
    art_gram_relu2 = gram_matrix(vgg_art_outputs.relu2)
    styled_gram_relu2 = gram_matrix(vgg_styled_outputs.relu2)
    loss += .5 * torch.norm(art_gram_relu2 - styled_gram_relu2, p='fro')**2

    _, channels, H, W = vgg_art_outputs.relu3.size()
    art_gram_relu3 = gram_matrix(vgg_art_outputs.relu3)
    styled_gram_relu3 = gram_matrix(vgg_styled_outputs.relu3)
    loss += .5 * torch.norm(art_gram_relu3 - styled_gram_relu3, p='fro')**2

    _, channels, H, W = vgg_art_outputs.relu4.size()
    art_gram_relu4 = gram_matrix(vgg_art_outputs.relu4)
    styled_gram_relu4 = gram_matrix(vgg_styled_outputs.relu4)
    loss += .5 * torch.norm(art_gram_relu4 - styled_gram_relu4, p='fro')**2

    return loss


def content_loss(vgg_outputs, vgg_styled_outputs):
    # Calculating for relu3, relu4
    # Content is present closer to output, whilts
    # style is closer nearer to input
    loss = 0.0
    batch_size, channels, H, W = vgg_outputs.relu3.size()
    loss += F.mse_loss(vgg_outputs.relu3,
                       vgg_styled_outputs.relu3,
                       reduction='sum') / batch_size
    batch_size, channels, H, W = vgg_outputs.relu4.size()
    loss += F.mse_loss(vgg_outputs.relu4,
                       vgg_styled_outputs.relu4,
                       reduction='sum') / batch_size
    return loss


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.detach().cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


style_net = FastStylizationNetwork(n_styles=8).to(device='cuda')

optimizer = torch.optim.Adam(style_net.parameters(), lr=0.001)

loss_net = LossNetwork().to(device='cuda')
loss_net.eval()

normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                      device='cuda')

global_step = 0

for images, _ in image_loader:
    sample = next(iter(art_loader))
    art, styles = sample['image'].to('cuda'), sample['target'].to('cuda')

    images = images.to('cuda')

    styled_images = style_net(images, styles)

    prepared_styled_images = normalize(styled_images)
    prepared_images = normalize(images)

    vgg_art_outputs = loss_net(art)
    vgg_styled_outputs = loss_net(prepared_styled_images)
    vgg_outputs = loss_net(prepared_images)

    c_loss = content_loss(vgg_outputs, vgg_styled_outputs)
    s_loss = 10e5 * style_loss(vgg_art_outputs, vgg_styled_outputs)

    loss = c_loss + s_loss

    writer.add_scalars('losses', {
        'content': c_loss,
        'style': s_loss,
        'full': loss
    },
                       global_step=global_step)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if global_step % 25 == 0:
        img_grid = torchvision.utils.make_grid(styled_images, nrow=4)
        matplotlib_imshow(img_grid, one_channel=False)
        writer.add_image('styled_images', img_grid, global_step=global_step)

        torch.save(style_net.state_dict(), './weights/model')

    global_step += 1