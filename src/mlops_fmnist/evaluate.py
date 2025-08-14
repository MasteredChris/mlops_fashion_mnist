from __future__ import annotations
import argparse
import torch
from .data import get_dataloaders
from .model import SmallCNN
from .utils import get_device


def main(args):
    device = get_device(args.cpu)
    _, test_dl = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model = SmallCNN().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"Test accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="./outputs/model.pth")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
