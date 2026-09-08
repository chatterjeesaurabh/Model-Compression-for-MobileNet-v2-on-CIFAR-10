# src/utils.py
# Utility functions for training, evaluation, data loading, and mlflow setup.

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def configure_cudnn(deterministic=True, benchmark=False):
    """Configure cuDNN for deterministic or fast behavior."""
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                    epoch, best_accuracy, epoch_at_best_accuracy):
    """Save training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_accuracy": best_accuracy,
        "epoch_at_best_accuracy": epoch_at_best_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler,
                    epoch, best_accuracy, epoch_at_best_accuracy):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    best_accuracy = checkpoint["best_accuracy"]
    epoch_at_best_accuracy = checkpoint["epoch_at_best_accuracy"]
    return model, optimizer, scheduler, epoch, best_accuracy, epoch_at_best_accuracy


def generate_run_name(script, mode, epochs, batch_size, lr):
    """Generate a default MLflow run name from hyperparameters."""
    lr_str = str(lr).replace(".", "")
    rand_id = random.randint(1000, 9999)
    return f"{script}_{mode}_{epochs}_{batch_size}_{lr_str}_{rand_id}"


def configure_mlflow(experiment_name=None, run_name=None, config=None):
    """Initialize MLflow logging."""
    import mlflow

    mlflow.set_experiment(experiment_name or "default")
    mlflow.start_run(run_name=run_name)
    if config:
        mlflow.log_params(config)


def prepare_dataloaders(batch_size):
    """Prepare CIFAR-10 train and test dataloaders with data augmentation."""
    # Training: RandomCrop + RandomHorizontalFlip + normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root="./data", train=True, download=True,
                            transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)

    # Test: just normalization (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = CIFAR10(root="./data", train=False, download=True,
                           transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)

    return train_dataloader, test_dataloader


def prepare_calib_dataloader(batch_size):
    """Prepare calibration dataloader (train set without augmentation)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def train(model, optimizer, scheduler, criterion, device, train_dataloader):
    """Run one epoch of training. Returns average loss."""
    model.train()
    loss_epoch = 0.0
    num_correct = 0
    num_samples = 0
    for data in tqdm(train_dataloader, total=len(train_dataloader), desc="train"):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        num_correct += (predicted == labels).sum().item()
        num_samples += inputs.size(0)
    scheduler.step()
    return loss_epoch / num_samples, num_correct / num_samples


def evaluate(model, device, dataloader, criterion=None):
    model.eval()
    loss_epoch = 0.0
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader), desc="test"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                loss_epoch += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()
    return (loss_epoch / num_samples if criterion is not None else None), num_correct / num_samples


def test(model, device, test_dataloader):
    """Evaluate model on test set. Returns accuracy as a float in [0, 1]."""
    _, accuracy = evaluate(model, device, test_dataloader)
    return accuracy


def replace_relu(module):
    """Replace ReLU6 with ReLU (used for compatibility with some quant methods)."""
    reassign = {}
    for name, mod in module.named_children():
        replace_relu(mod)
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)
    for key, value in reassign.items():
        module._modules[key] = value
