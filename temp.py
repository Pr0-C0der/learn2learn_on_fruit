import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root="data/tiny-imagenet-200/train", transform=transform)


import learn2learn as l2l

# Wrap the dataset with MetaDataset
meta_dataset = l2l.data.MetaDataset(dataset)


# Define tasksets for meta-learning
train_tasks = l2l.data.TaskDataset(meta_dataset, task_transforms=[
    l2l.data.transforms.NWays(meta_dataset, n=5),
    l2l.data.transforms.KShots(meta_dataset, k=5),
    l2l.data.transforms.LoadData(meta_dataset),
    l2l.data.transforms.RemapLabels(meta_dataset),
])

test_tasks = l2l.data.TaskDataset(meta_dataset, task_transforms=[
    l2l.data.transforms.NWays(meta_dataset, n=5),
    l2l.data.transforms.KShots(meta_dataset, k=5),
    l2l.data.transforms.LoadData(meta_dataset),
    l2l.data.transforms.RemapLabels(meta_dataset),
])


from torch.utils.data import DataLoader

# Create data loaders for the tasksets
train_loader = DataLoader(train_tasks, batch_size=1, shuffle=True)
test_loader = DataLoader(test_tasks, batch_size=1, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 11 * 11, 256)  # Adjusted for 100x100 input size
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the MAML algorithm
import learn2learn as l2l

model = SimpleCNN(num_classes=5)
maml = l2l.algorithms.MAML(model, lr=0.01)

# Define an optimizer
opt = torch.optim.Adam(maml.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
def train_maml(maml, train_loader, opt, loss_fn, adaptation_steps=1, num_epochs=5):
    maml.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.squeeze(0)  # Remove batch dimension added by DataLoader
            y = y.squeeze(0)

            # Create a clone of the MAML model
            learner = maml.clone()

            # Adaptation phase
            for step in range(adaptation_steps):
                pred = learner(x)
                loss = loss_fn(pred, y)
                learner.adapt(loss)

            # Meta-update phase
            pred = learner(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

# Testing loop
def test_maml(maml, test_loader, loss_fn, adaptation_steps=1):
    maml.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.squeeze(0)  # Remove batch dimension added by DataLoader
        y = y.squeeze(0)

        # Create a clone of the MAML model
        learner = maml.clone()

        # Adaptation phase
        for step in range(adaptation_steps):
            pred = learner(x)
            loss = loss_fn(pred, y)
            learner.adapt(loss)

        # Evaluation phase
        pred = learner(x)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100}%")

# Training and testing the MAML model
print("Model is training")
train_maml(maml, train_loader, opt, loss_fn, adaptation_steps=1, num_epochs=5)
test_maml(maml, test_loader, loss_fn, adaptation_steps=1)
