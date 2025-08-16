from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # mean/std di Fashion-MNIST
        ]
    )


def get_data_loaders(
    data_dir: str = "./data", batch_size: int = 64, num_workers: int = 2
):
    tfm = get_transforms()
    train_ds = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=tfm
    )
    test_ds = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=tfm
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, test_dl
