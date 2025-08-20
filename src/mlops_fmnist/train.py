import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from mlops_fmnist.data import get_data_loaders
from mlops_fmnist.model import SmallCNN


def train_model(model, device, train_loader, criterion, optimizer, epoch, logger):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    return avg_loss


def evaluate_model(model, device, test_loader, criterion, logger):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(f"Validation Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    return avg_loss, accuracy


def main(args):
    #setup output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #setup logging
    log_path = output_dir / "train.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(batch_size=64)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    metrics = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_model(model, device, train_loader, criterion, optimizer, epoch, logger)
        val_loss, val_acc = evaluate_model(model, device, test_loader, criterion, logger)
        metrics[f"epoch_{epoch}"] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }

    #save final model
    model_path = output_dir / "final_tensor.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")

    #save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved at {metrics_path}")

    logger.info("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    args = parser.parse_args()
    main(args)
