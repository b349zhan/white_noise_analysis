import torch.utils.data.dataloader as dataloader
from torchvision.datasets import FashionMNIST, MNIST

def get_loader(dataset_name, image_transform, cuda:bool, download:bool):
    if dataset_name == "MNIST":
        train_dataset = MNIST('./data', train=True, download=download, transform=image_transform)

        test_dataset = MNIST('./data', train=False, download=download, transform=image_transform)
    else:
        train_dataset = FashionMNIST('./data', train=True, download=download, transform=image_transform)

        test_dataset = FashionMNIST('./data', train=False, download=download, transform=image_transform)

    dataloader_args = dict(shuffle=True, batch_size=256,num_workers=4, pin_memory=True) if cuda\
                 else dict(shuffle=True, batch_size=64)
    train_loader = dataloader.DataLoader(train_dataset, **dataloader_args)
    test_loader = dataloader.DataLoader(test_dataset, **dataloader_args)
    return train_dataset, test_dataset, train_loader, test_loader
