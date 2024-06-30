#!/usr/bin/env python3

import os
import random

import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import learn2learn as l2l
from learn2learn.data import TaskDataset, MetaDataset
from learn2learn.data.transforms import LoadData, RemapLabels, ConsecutiveLabels

class SequentialNonRepeatingTaskSampler:
    def __init__(self, dataset, n_ways, k_shots):
        self.dataset = dataset
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.classes = len(dataset.labels_to_indices)
        self.tasks = self.generate_tasks()
        random.shuffle(self.tasks)
    
    def generate_tasks(self):
        tasks = []
        all_classes = list(range(self.classes))
        for i in range(0, self.classes, self.n_ways):
            if len(all_classes) < self.n_ways:
                break
            task_classes = all_classes[:self.n_ways]
            all_classes = all_classes[self.n_ways:]
            task_data = []
            for cls in task_classes:
                indices = self.dataset.labels_to_indices[cls]
                selected_indices = random.sample(indices, min(self.k_shots, len(indices)))
                task_data.extend([(self.dataset[idx][0], cls) for idx in selected_indices])
            tasks.append(task_data)
        return tasks

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root="data/dataset", transform=transform)

# Wrap the dataset with MetaDataset
meta_dataset = MetaDataset(dataset)

# Custom sampler to ensure sequential non-repeating classes
sampler = SequentialNonRepeatingTaskSampler(meta_dataset, n_ways=5, k_shots=5)

# Split tasks into training and evaluation sets
num_tasks = len(sampler.tasks)
num_train_tasks = int(0.8 * num_tasks)
train_tasks = sampler.tasks[:num_train_tasks]
eval_tasks = sampler.tasks[num_train_tasks:]

print(train_tasks)