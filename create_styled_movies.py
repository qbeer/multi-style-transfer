import os

import torch
from models import FastStylizationNetwork
import torchvision.transforms as T
from PIL import Image
import argparse
import cv2
import glob
from tqdm import tqdm


def sample(args):
    net = FastStylizationNetwork(n_styles=8)
    net.load_state_dict(torch.load('weights/model_best'))
    net = net.to('cuda')
    net.eval()

    img_paths = glob.glob(f'{args.img_path}/*.{args.img_ext}')

    for style_type in tqdm(range(8)):
        for img_path in tqdm(img_paths):
            frame = Image.open(img_path)
            frame = T.Resize(size=(args.height, args.width))(frame)
            frame = T.ToTensor()(frame)
            frame = torch.unsqueeze(frame, 0)  # expand dims basically, batch of 1

            style = torch.tensor([style_type], dtype=torch.int64)

            frame = frame.to('cuda')
            style = style.to('cuda')
            
            styled_frame = net(frame, style)
            styled_frame = styled_frame[0].cpu().detach().permute(1, 2, 0).numpy()
            styled_frame = (styled_frame * 255.).astype('uint8')

            cv2.imwrite(img_path.replace(f'.{args.img_ext}',
             f'_styled_{style_type}.png'),
              styled_frame)

            del frame, styled_frame


parser = argparse.ArgumentParser()

parser.add_argument('--img_ext', type=str, default='png', required=False)
parser.add_argument('--width', type=int, default=16 * 64, required=False)
parser.add_argument('--height', type=int, default=9 * 64, required=False)
parser.add_argument(
    '--img_path',
    default='/home/qbeer/pics_vesuvio',
    type=str,
    required=False,
)

args = parser.parse_args()

sample(args)
