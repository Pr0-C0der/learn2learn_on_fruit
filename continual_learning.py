import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from avalanche.benchmarks import nc_benchmark
from avalanche.training.supervised import Naive
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger

# Define a custom MLP model
class CustomMLP(nn.Module):
    def __init__(self, num_classes):
        super(CustomMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100*100*3, 512)  # Correct input size for 100x100 RGB images
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

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

# Split the dataset into training and test sets
train_size = int(0.5 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create the continual learning scenario
scenario = nc_benchmark(train_dataset, test_dataset, n_experiences=40, task_labels=False, shuffle=False)

# Define the model
model = CustomMLP(num_classes=scenario.n_classes)

# Define the strategy
interactive_logger = InteractiveLogger()
evaluation_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

strategy = Naive(
    model=model,
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    criterion=torch.nn.CrossEntropyLoss(),
    train_mb_size=32,
    train_epochs=1,
    eval_mb_size=32,
    device='cpu',
    evaluator=evaluation_plugin
)

# Print class names for each experience
for experience in scenario.train_stream:
    exp_classes = [class_names[idx] for idx in experience.classes_in_this_experience]
    print(f"Classes in experience {experience.current_experience}: {exp_classes}")

# Training loop
for experience in scenario.train_stream:
    print(f"Start training on experience {experience.current_experience}")
    strategy.train(experience)
    print(f"End training on experience {experience.current_experience}")
    
    print("Computing accuracy on the test set")
    strategy.eval(scenario.test_stream)
