import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root="data/dataset", transform=transform)


import learn2learn as l2l

# Wrap the dataset with MetaDataset
meta_dataset = l2l.data.MetaDataset(dataset)

print("DONE")

# Define tasksets for meta-learning
train_tasks = l2l.data.TaskDataset(meta_dataset, task_transforms=[
    l2l.data.transforms.NWays(meta_dataset, n=5),
    l2l.data.transforms.KShots(meta_dataset, k=5),
    l2l.data.transforms.LoadData(meta_dataset),
    l2l.data.transforms.RemapLabels(meta_dataset),
])

print("DONE")
test_tasks = l2l.data.TaskDataset(meta_dataset, task_transforms=[
    l2l.data.transforms.NWays(meta_dataset, n=5),
    l2l.data.transforms.KShots(meta_dataset, k=5),
    l2l.data.transforms.LoadData(meta_dataset),
    l2l.data.transforms.RemapLabels(meta_dataset),
], num_tasks = 40)

print("DONE")
print(train_tasks)