


N_EXPERIENCES = 40

import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from avalanche.benchmarks import nc_benchmark
from avalanche.training.supervised import LwF
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from itertools import product

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
scenario = nc_benchmark(train_dataset, test_dataset, n_experiences=N_EXPERIENCES, shuffle=False, task_labels=True) #Change n_experiences

# Define the parameter grid
alphas = [0.5, 1.0]
temperatures = [1, 2]
train_mb_sizes = [32, 64]
train_epochs_list = [1, 2]
eval_mb_sizes = [32, 64]

# Create the results directory
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Grid search
for alpha, temperature, train_mb_size, train_epochs, eval_mb_size in product(alphas, temperatures, train_mb_sizes, train_epochs_list, eval_mb_sizes):
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

    strategy = LwF(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        criterion=torch.nn.CrossEntropyLoss(),
        alpha=alpha,
        temperature=temperature,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device='cpu',
        evaluator=evaluation_plugin
    )

    # Create a directory for this parameter set
    param_dir = os.path.join(results_dir, f'alpha_{alpha}_temp_{temperature}_trainmb_{train_mb_size}_epochs_{train_epochs}_evalmb_{eval_mb_size}')
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    # File to save results
    results_file = os.path.join(param_dir, 'results.txt')

    # Training loop
    with open(results_file, 'w+') as f:
        for experience in scenario.train_stream:
            print(f"Start training on experience {experience.current_experience}")
            # f.write(f"Start training on experience {experience.current_experience}\n")
            strategy.train(experience)
            print(f"End training on experience {experience.current_experience}")
            # f.write(f"End training on experience {experience.current_experience}\n")
            
            print("Computing accuracy on the test set")
            eval_results = strategy.eval(scenario.test_stream)
            
            # Log the accuracy on all test experiences
            all_accuracies = []
            for exp in scenario.test_stream:
                curr = exp.current_experience
                if curr < 10:
                    accuracy = eval_results[f'Top1_Acc_Exp/eval_phase/test_stream/Task00{curr}/Exp00{curr}']
                else:
                    accuracy = eval_results[f'Top1_Acc_Exp/eval_phase/test_stream/Task0{curr}/Exp0{curr}']
                    
                all_accuracies.append(accuracy)
                # f.write(f"Accuracy on test experience {exp.current_experience}: {accuracy}\n")

            print(f"#########################ALL ACCURACIES: {all_accuracies}############################")
            #Printing accuracies in the file
            f.write(' '.join(format(x, '.3f') for x in all_accuracies))
            f.write('\n')

