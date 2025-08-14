from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms


def test_fake_loader_shape():
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = FakeData(size=100, image_size=(1, 28, 28), num_classes=10, transform=tfm)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    x, y = next(iter(dl))
    assert x.shape == (16, 1, 28, 28)
    assert y.shape == (16,)
