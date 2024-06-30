import itertools
import os
import random
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
import learn2learn as l2l
from tqdm import tqdm
import matplotlib.pyplot as plt

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

        return train_tasks, eval_tasks

# Define transforms for the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root="learn2learn_on_fruit/data/dataset", transform=transform)

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

# Parameters for grid search
n_ways_range = [5]
k_shots_range = [5]
k_query_range = [5]
adaptation_steps_range = [5]
maml_lr_range = [0.01, 0.001]
opt_lr_range = [0.001, 0.0001]
num_epochs_range = [10]

# Generate all combinations of the parameter ranges
param_grid = itertools.product(n_ways_range, k_shots_range, k_query_range, adaptation_steps_range, maml_lr_range, opt_lr_range, num_epochs_range)

# Path to save the results
save_path = "grid_search_results"

# Create the save path if it does not exist
os.makedirs(save_path, exist_ok=True)

# Iterate over each combination of parameters
for params in param_grid:
    n_ways, k_shots, k_query, adaptation_steps, maml_lr, opt_lr, num_epochs = params

    # Create a folder for the current grid search iteration
    iteration_path = os.path.join(save_path, f"nways_{n_ways}_kshots_{k_shots}_kquery_{k_query}_adaptsteps_{adaptation_steps}_mamllr_{maml_lr}_optlr_{opt_lr}_epochs_{num_epochs}")
    os.makedirs(iteration_path, exist_ok=True)

    # Sub-folder for task-wise loss
    task_loss_path = os.path.join(iteration_path, "Task Wise Loss")
    os.makedirs(task_loss_path, exist_ok=True)

    # Custom sampler to ensure sequential non-repeating classes with different samples for training and evaluation
    sampler = SequentialNonRepeatingTaskSampler(dataset, n_ways=n_ways, k_shots=k_shots, k_query=k_query)

    model = SimpleCNN(num_classes=200).to(device)
    maml = l2l.algorithms.MAML(model, lr=maml_lr, first_order=False).to(device)

    # Define the optimizer
    opt = optim.Adam(maml.parameters(), lr=opt_lr)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Lists to store losses and accuracies
    task_losses = []
    train_losses = []
    eval_accuracies = []

    # Meta-training loop
    for task_id, task_data in enumerate(tqdm(sampler.train_tasks, desc=f"Training with params {params}", unit="task")):
        task_epoch_losses = []
        for epoch in range(num_epochs):
            X, y = zip(*task_data)
            X = torch.stack(X).to(device)
            y = torch.tensor(y).to(device)
            # print(y)

            # Adapt the model
            learner = maml.clone()
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

            task_epoch_losses.append(loss.item())
            train_losses.append(loss.item())
        
        task_losses.append(task_epoch_losses)

        # Save task-wise loss graph
        plt.figure()
        plt.plot(range(1, num_epochs+1), task_epoch_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Task {task_id+1} Loss')
        plt.savefig(os.path.join(task_loss_path, f'task_{task_id+1}_loss.png'))
        plt.close()

    # Meta-evaluation
    eval_task_accuracies = []
    for task_data in sampler.eval_tasks:
        X, y = zip(*task_data)
        X = torch.stack(X).to(device)
        y = torch.tensor(y).to(device)
        
        # Adapt the model
        learner = maml.clone()
        for step in range(adaptation_steps):
            pred = learner(X)
            loss = loss_fn(pred, y)
            learner.adapt(loss)
        
        # Evaluate the adapted model
        pred = learner(X)
        loss = loss_fn(pred, y)
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        eval_task_accuracies.append(acc)

    eval_accuracies.append(np.mean(eval_task_accuracies))



    # Save training loss graph
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Across Tasks')
    plt.savefig(os.path.join(iteration_path, 'training_loss.png'))
    plt.close()

    # Save evaluation accuracy graph
    plt.figure()
    plt.plot(range(1, len(eval_task_accuracies)+1), eval_task_accuracies)
    plt.xlabel('Tasks')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Across Tasks')
    plt.savefig(os.path.join(iteration_path, 'test_accuracy.png'))
    plt.close()

    with open(os.path.join(iteration_path, 'loss_and_acc.txt'), 'w+') as f:
        f.write("Evaluation Task Accuracies:\n")
        f.write('\n'.join(map(str, eval_task_accuracies)) + '\n')
        f.write("Training Losses:\n")
        f.write('\n'.join(map(str, train_losses)) + '\n')

    print("DONE")
