"""selection-agent: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

class CNNModel(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, noniid: bool = False):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
        
    partitioner = IidPartitioner(num_partitions=num_partitions) if not noniid else DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=0.5,
        min_partition_size=10,
        self_balancing=True
    )
    fds = FederatedDataset(
        dataset='uoft-cs/cifar10',
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(model, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    squared_sum = 0
    num_samples = 0
    for epoch in range(epochs):
        total_loss = 0
        correct_pred = total_pred = 0

        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            output = model(images)
            
            loss = criterion(model(images), labels)
            
            if epoch == (epochs-1):
                squared_sum += float(sum(np.power(loss.cpu().detach().numpy(), 2)))
                num_samples += len(images)
            loss = loss.mean()

            predicted = output.argmax(1)
            total_pred += labels.size(0)
            correct_pred += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)            
            
    avg_acc = correct_pred / total_pred
    avg_loss = total_loss / len(trainloader)
    stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
    return avg_loss, avg_acc, stat_util


def test(model, testloader, device):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0.0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
