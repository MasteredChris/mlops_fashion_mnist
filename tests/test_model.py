import torch
from mlops_fmnist.model import SmallCNN


def test_forward_shape():
    model = SmallCNN()
    x = torch.randn(8, 1, 28, 28)
    y = model(x)
    assert y.shape == (8, 10)
