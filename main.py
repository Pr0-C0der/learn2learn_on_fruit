#!/usr/bin/env python3
#learn2learn_on_fruit/data/dataset
#!/usr/bin/env python3

from tqdm import tqdm
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder

import learn2learn as l2l

class SequentialNonRepeatingTaskSampler:
    def __init__(self, dataset, n_ways, k_shots, k_query):
        self.dataset = dataset
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.k_query = k_query
        self.classes = list(dataset.class_to_idx.values())
        self.class_indices = {cls: np.where(np.array(self.dataset.targets) == cls)[0] for cls in self.classes}
        self.train_tasks, self.eval_tasks = self.generate_tasks()

    def generate_tasks(self):
        train_tasks = []
        eval_tasks = []

        num_classes = len(self.classes)
        num_tasks = num_classes // self.n_ways

        for task_id in range(num_tasks):
            task_classes = self.classes[task_id * self.n_ways : (task_id + 1) * self.n_ways]
            
            train_task = []
            eval_task = []

            for cls in task_classes:
                indices = self.class_indices[cls]
                random.shuffle(indices)

                train_indices = indices[:self.k_shots]
                eval_indices = indices[self.k_shots:self.k_shots + self.k_query]

                train_task.extend([(self.dataset[idx][0], cls) for idx in train_indices])
                eval_task.extend([(self.dataset[idx][0], cls) for idx in eval_indices])

            train_tasks.append(train_task)
            eval_tasks.append(eval_task)

        # # Print train and eval labels
        # print("Train Tasks Labels:")
        # for task in train_tasks:
        #     labels = [label for _, label in task]
        #     print(labels)
            
        # print("Eval Tasks Labels:")
        # for task in eval_tasks:
        #     labels = [label for _, label in task]
        #     print(labels)

        return train_tasks, eval_tasks

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root="learn2learn_on_fruit/data/dataset", transform=transform)

# Custom sampler to ensure sequential non-repeating classes with different samples for training and evaluation
sampler = SequentialNonRepeatingTaskSampler(dataset, n_ways=5, k_shots=5, k_query=5)

# The rest of your training and evaluation code
# Define the MAML model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # Output: (32, 98, 98)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # Output: (64, 47, 47)
        self.fc1 = nn.Linear(64 * 23 * 23, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)  # Output: (32, 49, 49)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)  # Output: (64, 23, 23)
        x = x.view(-1, 64 * 23 * 23)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=200).to(device)
maml = l2l.algorithms.MAML(model, lr=0.01, first_order=False).to(device)

# Define the optimizer
opt = optim.Adam(maml.parameters(), lr=0.001)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Meta-training loop
i = 0
for task_data in sampler.train_tasks:
    print(f"---------------------------------------TASK {i}---------------------------------------")
    i+=1
    for epoch in range(10):
        X, y = zip(*task_data)
        X = torch.stack(X).to(device)
        y = torch.tensor(y).to(device)

        # Adapt the model
        learner = maml.clone()
        adaptation_steps = 5
        for step in range(adaptation_steps):
            pred = learner(X)
            loss = loss_fn(pred, y)
            learner.adapt(loss)

        # Meta-optimization
        pred = learner(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Meta-evaluation
for task_data in sampler.eval_tasks:
    X, y = zip(*task_data)
    X = torch.stack(X).to(device)
    y = torch.tensor(y).to(device)
    
    # Adapt the model
    learner = maml.clone()
    adaptation_steps = 5
    for step in range(adaptation_steps):
        pred = learner(X)
        loss = loss_fn(pred, y)
        learner.adapt(loss)
    
    # Evaluate the adapted model
    pred = learner(X)
    loss = loss_fn(pred, y)
    acc = (pred.argmax(dim=1) == y).float().mean().item()
    print(f"Evaluation Loss: {loss.item()}, Accuracy: {acc}")