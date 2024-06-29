import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset
data_dir = 'path_to_your_dataset'  # Replace with your dataset path

# Create a dataset
full_dataset = ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and testing sets (55 classes, 5 images each for train and test)
train_size = 275  # 55 classes * 5 images
test_size = 275  # 55 classes * 5 images
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
