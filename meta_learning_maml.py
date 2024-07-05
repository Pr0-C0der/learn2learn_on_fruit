import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import learn2learn as l2l
from learn2learn.data import TaskDataset, MetaDataset
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
from learn2learn.algorithms import MAML
from collections import Counter
import numpy as np
from itertools import product
import time

# Define a custom MLP model
class CustomMLP(nn.Module):
    def __init__(self, output_size):
        super(CustomMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100*100*3, 512)  # Correct input size for 100x100 RGB images
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def count_samples_per_class(labels, class_names):
    class_counts = Counter([label.item() for label in labels])
    for class_idx, count in class_counts.items():
        print(f"{class_names[class_idx]}: {count}")

def print_task_distribution(tasks, class_names, title, num_tasks=10):
    print(f"\n{title} Task Distribution:")
    for task_idx, task_indices in enumerate(itertools.islice(tasks, num_tasks)):
        print(f"Task {task_idx + 1}:")
        task_labels = [tasks.dataset.targets[i] for i in task_indices]
        count_samples_per_class(task_labels, class_names)

# Define the fast adaptation function
def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    adaptation_data, adaptation_labels = data[:shots*ways], labels[:shots*ways]
    evaluation_data, evaluation_labels = data[shots*ways:], labels[shots*ways:]
    
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)
        
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = (predictions.argmax(dim=1) == evaluation_labels).sum().item() / evaluation_labels.size(0)
    
    return valid_error, valid_accuracy

# Set the paths
dataset_path = 'learn2learn_on_fruit/data/dataset'  # Replace with the path to your dataset

# Define a simple transformation to normalize the images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Load the dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
class_names = full_dataset.classes

# Split the dataset such that there are only 5 examples per class for training and 5 for testing
def split_dataset(dataset, n_train, n_test):
    targets = np.array([s[1] for s in dataset.samples])
    train_indices = []
    test_indices = []

    for class_idx in range(len(dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:n_train + n_test])

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset

train_dataset, test_dataset = split_dataset(full_dataset, n_train=5, n_test=5)

# Print number of examples per class in train_dataset and test_dataset
def count_classes(dataset):
    targets = [s[1] for s in dataset]
    return Counter(targets)

train_class_counts = count_classes(train_dataset)
test_class_counts = count_classes(test_dataset)

print("Train Dataset Class Counts:")
for class_idx, count in train_class_counts.items():
    print(f"{class_names[class_idx]}: {count}")

print("\nTest Dataset Class Counts:")
for class_idx, count in test_class_counts.items():
    print(f"{class_names[class_idx]}: {count}")

# Create MetaDataset and TaskDataset
train_meta_dataset = MetaDataset(train_dataset)
test_meta_dataset = MetaDataset(test_dataset)
import itertools

class SequentialTaskSampler:
    def __init__(self, dataset, n, k):
        self.dataset = dataset
        self.n = n
        self.k = k
        self.current_class = 0
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_class >= len(self.dataset.targets):
            raise StopIteration
        class_indices = [i for i in self.indices if self.dataset.targets[i] == self.current_class]
        task_indices = class_indices[:self.k] + class_indices[self.k:self.k * 2]
        self.current_class += 1
        if len(task_indices) < self.n * self.k:
            raise StopIteration
        return task_indices

    def __len__(self):
        return len(self.dataset.targets) // (self.n * self.k)

# Define the custom sampler for sequential tasks
train_tasks = SequentialTaskSampler(train_meta_dataset, n=5, k=5)
test_tasks = SequentialTaskSampler(test_meta_dataset, n=5, k=5)


# Print the task distribution
print_task_distribution(train_tasks, class_names, "Train")
print_task_distribution(test_tasks, class_names, "Test")

# # Define the parameter grid
# maml_steps = [1, 5]
# train_mb_sizes = [32, 64]
# train_epochs_list = [10, 20]
# eval_mb_sizes = [32, 64]

# # Create the results directory
# results_dir = 'results'
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Grid search
# for maml_step, train_mb_size, train_epochs, eval_mb_size in product(maml_steps, train_mb_sizes, train_epochs_list, eval_mb_sizes):
#     # Define the model
#     start_time = time.time()
#     model = CustomMLP(output_size=len(class_names)).to(device)

#     # Define the MAML optimizer
#     maml = MAML(model, lr=0.01, first_order=False).to(device)

#     # Create a directory for this parameter set
#     param_dir = os.path.join(results_dir, f'maml_steps_{maml_step}_trainmb_{train_mb_size}_epochs_{train_epochs}_evalmb_{eval_mb_size}')
#     if not os.path.exists(param_dir):
#         os.makedirs(param_dir)

#     # File to save results
#     results_file = os.path.join(param_dir, 'results.txt')

#     total_accuracies = []
#     # Training loop
#     with open(results_file, 'w+') as f:
#         optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Create optimizer outside the loop
#         for epoch in range(train_epochs):
#             print(f"Start training on epoch {epoch}")
#             train_loader = DataLoader(train_tasks, batch_size=train_mb_size, shuffle=True)
            
#             for batch in train_loader:
#                 learner = maml.clone()
#                 optimizer.zero_grad()  # Reset gradients
                
#                 train_loss, _ = fast_adapt(batch, learner, torch.nn.CrossEntropyLoss(), maml_step, 5, 5, device)
#                 train_loss.backward()
#                 optimizer.step()

#             print(f"End training on epoch {epoch}")

#             print("Computing accuracy on the test set")
#             test_loader = DataLoader(test_tasks, batch_size=eval_mb_size, shuffle=False)
#             accuracies = []
            
#             for batch in test_loader:
#                 learner = maml.clone()
                
#                 _, test_accuracy = fast_adapt(batch, learner, torch.nn.CrossEntropyLoss(), maml_step, 5, 5, device)
#                 accuracies.append(test_accuracy)

#             avg_accuracy = sum(accuracies) / len(accuracies)
#             total_accuracies.append(avg_accuracy)
#             print(f"Accuracy: {avg_accuracy}")
#             f.write(f"Epoch {epoch} Accuracy: {avg_accuracy}\n")

#         f.write("Final Accuracies:\n")
#         f.write(' '.join(format(x, '.3f') for x in total_accuracies))

#     end_time = time.time()
#     print(f"Execution Time: {end_time-start_time} seconds")
