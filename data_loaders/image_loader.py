import torch
import torchvision

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(256, 256)),
    torchvision.transforms.ToTensor()
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(256, 256)),
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR100(root='./data/',
                                        transform=transforms,
                                        target_transform=None,
                                        download=True)

image_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           drop_last=True)
