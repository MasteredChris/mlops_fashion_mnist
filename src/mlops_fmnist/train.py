from __future__ import annotations
import argparse
import json
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from tqdm import tqdm
from .data import get_dataloaders
from .model import SmallCNN
from .utils import set_seed, get_device, ensure_dir


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train(args):
    set_seed(args.seed)
    device = get_device(args.cpu)
    train_dl, test_dl = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # quick eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Val acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            ensure_dir(args.output_dir)
            ckpt_path = f"{args.output_dir}/model.pth"
            torch.save(model.state_dict(), ckpt_path)

    # save metrics
    ensure_dir(args.output_dir)
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(
            {"best_val_acc": best_acc, "epochs": args.epochs, "ts": time.time()}, f
        )
    print(f"Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)
