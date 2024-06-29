import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
import time
from collections import defaultdict
import random
def save_sampled_dataset(dataset, sampled_indices, target_dir):
    # Create the target directory if it does not exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each class (to mirror the original dataset structure)
    for class_idx in range(len(dataset.classes)):
        class_name = dataset.classes[class_idx]
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Copy sampled images to the target directory preserving original folder structure
    for idx in sampled_indices:
        img_path, label = dataset.samples[idx]
        class_name = dataset.classes[label]
        target_path = os.path.join(target_dir, class_name, os.path.basename(img_path))
        shutil.copy(img_path, target_path)

def sample_images_per_class(dataset, num_samples_per_class=10):
    class_indices = defaultdict(list)
    
    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    sampled_indices = []
    for indices in class_indices.values():
        sampled_indices.extend(random.sample(indices, num_samples_per_class))
    
    return sampled_indices

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset
data_dir = 'data/tiny-imagenet-200/train'  # Replace with your dataset path

# Create a dataset
full_dataset = ImageFolder(root=data_dir, transform=transform)

# Sample 10 images per class
sampled_indices = sample_images_per_class(full_dataset, num_samples_per_class=10)

# Create a subset of the dataset using the sampled indices
sampled_dataset = Subset(full_dataset, sampled_indices)


# Define the sizes for training and testing datasets
num_classes = len(full_dataset.classes)
train_size = num_classes * 5  # 55 classes * 5 images
test_size = num_classes * 5   # 55 classes * 5 images

# Split the sampled dataset into training and testing datasets
train_dataset, test_dataset = random_split(sampled_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# Load the dataset using MetaDataset

meta_dataset = l2l.data.MetaDataset(dataset_dir="/data/tiny-imagenet-200/train", transform=transform)

# Split the meta-dataset into training and testing datasets
train_dataset, test_dataset = meta_dataset.train_test_split(train_num=5, test_num=5)

# Create data loaders
train_loader = l2l.data.MetaLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = l2l.data.MetaLoader(test_dataset, batch_size=1, shuffle=True)



print("Loaded the dataset")
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=55):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
#         self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flattened_size = self._get_flattened_size()
#         self.fc1 = nn.Linear(self.flattened_size, 256)
#         self.fc2 = nn.Linear(256, num_classes)

#     def _get_flattened_size(self):
#         # Dummy forward pass to calculate the size after conv layers
#         dummy_input = torch.randn(1, 3, 100, 100)
#         x = self.pool(F.relu(self.conv1(dummy_input)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         return x.view(-1).size(0)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# # Define a model
# model = SimpleCNN(num_classes=200)
# maml = l2l.algorithms.MAML(model, lr=0.01)

# # Define an optimizer
# opt = torch.optim.Adam(maml.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss()


# print("Loaded the model")

# def train_maml(maml, train_loader, opt, loss_fn, adaptation_steps=1, num_epochs=1):
#     maml.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         print(f"The length of train loader is {len(train_loader)}")
#         for i, (x, y) in enumerate(train_loader):
#             print(i)
#             # Create a clone of the MAML model
#             learner = maml.clone()

#             # Adaptation phase
#             for step in range(adaptation_steps):
#                 pred = learner(x)
#                 # print(pred)
#                 loss = loss_fn(pred, y)
#                 learner.adapt(loss)

#             # Meta-update phase
#             pred = learner(x)
#             loss = loss_fn(pred, y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             epoch_loss += loss.item()
#         print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")



# def test_maml(maml, test_loader, adaptation_steps=1):
#     maml.eval()
#     correct = 0
#     total = 0
#     for i, (x, y) in enumerate(test_loader):
#         print(i)
#         # Create a clone of the MAML model
#         learner = maml.clone()

#         # Adaptation phase
#         for step in range(adaptation_steps):
#             pred = learner(x)
#             loss = loss_fn(pred, y)
#             learner.adapt(loss)

#         # Evaluation phase
#         pred = learner(x)
#         correct += (pred.argmax(dim=1) == y).sum().item()
#         total += y.size(0)

#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy * 100}%")



# print("Training the model")
# start_time = time.time()
# train_maml(maml, train_loader, opt, loss_fn, adaptation_steps=1, num_epochs=1)
# end_time = time.time()

# print(f"Execution Time in seconds: {end_time-start_time}")
# print("Testing the model")
# test_maml(maml, test_loader, adaptation_steps=1)