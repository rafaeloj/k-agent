import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CNNMNISTModel(nn.Module):
    """CNN for MNIST (1x28x28)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train_mnist_cnn(model, trainloader, epochs, lr, device):
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
            images = batch["image"].to(device)
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
        print(correct_pred / total_pred)  
    avg_acc = correct_pred / total_pred
    avg_loss = total_loss / len(trainloader)
    stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
    return avg_loss, avg_acc, stat_util

def test_mnist_cnn(model, testloader, device):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0.0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
