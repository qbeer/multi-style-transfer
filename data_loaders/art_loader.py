import torch
import glob
from PIL import Image
import torchvision.transforms as T


class ArtDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, categories in subfolders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = [
            Image.open(im) for im in sorted(glob.glob(root_dir + "/*"))
        ]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = idx

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        sample = {'image': image, 'target': target}

        return sample


transforms = T.Compose([
    T.Resize(size=(312, 312)),
    T.RandomCrop(size=(256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

art_dataset = ArtDataset(root_dir='./data/art', transform=transforms)

art_loader = torch.utils.data.DataLoader(art_dataset,
                                         batch_size=8,
                                         shuffle=True)
