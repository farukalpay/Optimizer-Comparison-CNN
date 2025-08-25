import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import argparse
from psd_optimizer import PSDOptimizer

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

# Configuration
class CONFIG:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    DATASETS = ["MNIST", "CIFAR10"]
    OPTIMIZERS = ["SGD", "Adam", "PSD"]
    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE_SGD = 0.01
    LEARNING_RATE_ADAM = 0.001
    LEARNING_RATE_PSD = 0.01
    MOMENTUM_SGD = 0.9
    PSD_GRAD_THRESHOLD = 1e-3  # Threshold for gradient norm to trigger perturbation
    PSD_PERTURBATION_RADIUS = 1e-4  # Radius of the perturbation noise
    PSD_T = 10  # Number of steps after perturbation
    DATA_DIR = "./data"

# Data Loading Function
def get_dataloaders(dataset_name: str, batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepare and load datasets with appropriate transforms and splits.
    
    Args:
        dataset_name: Name of the dataset ('MNIST' or 'CIFAR10')
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, validation_loader)
    """
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full training set
        full_train_set = torchvision.datasets.MNIST(
            root=CONFIG.DATA_DIR, train=True, download=True, transform=transform
        )
        
        # Split into train and validation (50,000/10,000)
        train_size = 50000
        val_size = 10000
        train_set, val_set = torch.utils.data.random_split(
            full_train_set, [train_size, val_size]
        )
        
        test_set = torchvision.datasets.MNIST(
            root=CONFIG.DATA_DIR, train=False, download=True, transform=transform
        )
        
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load full training set
        full_train_set = torchvision.datasets.CIFAR10(
            root=CONFIG.DATA_DIR, train=True, download=True, transform=transform
        )
        
        # Split into train and validation (45,000/5,000)
        train_size = 45000
        val_size = 5000
        train_set, val_set = torch.utils.data.random_split(
            full_train_set, [train_size, val_size]
        )
        
        test_set = torchvision.datasets.CIFAR10(
            root=CONFIG.DATA_DIR, train=False, download=True, transform=transform
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, validation_loader, test_loader

# CNN Model Definition
class SimpleCNN(nn.Module):
    """
    A simple CNN architecture that works for both MNIST and CIFAR-10.
    
    Architecture:
        conv1 (3x3, 32 filters) -> ReLU -> MaxPool (2x2)
        conv2 (3x3, 64 filters) -> ReLU -> MaxPool (2x2)
        fc1 (128 units) -> ReLU -> fc2 (num_classes)
    """
    
    def __init__(self, num_classes: int, input_channels: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # For MNIST: 28x28 -> 14x14 -> 7x7
        # For CIFAR-10: 32x32 -> 16x16 -> 8x8
        self.fc1_input_features = 64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8
        
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_input_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom PSD Optimizer
# Training and Evaluation Functions
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Device to run training on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
    
    avg_loss = running_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    
    return avg_loss, accuracy

# Visualization Function
def plot_results(results, datasets, optimizers):
    """
    Plot training loss, validation loss, and validation accuracy for all experiments.
    
    Args:
        results: Dictionary containing results for all experiments
        datasets: List of dataset names
        optimizers: List of optimizer names
    """
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 5 * len(datasets)))
    
    # If only one dataset, make axes 2D
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    colors = {'SGD': 'blue', 'Adam': 'red', 'PSD': 'green'}
    linestyles = {'SGD': '-', 'Adam': '--', 'PSD': '-.'}
    
    for i, dataset in enumerate(datasets):
        # Training Loss
        ax = axes[i, 0]
        for optimizer in optimizers:
            key = f"{dataset}_{optimizer}"
            if key in results:
                epochs = range(1, len(results[key]['train_loss']) + 1)
                ax.plot(epochs, results[key]['train_loss'], 
                       label=optimizer, color=colors[optimizer], linestyle=linestyles[optimizer])
        ax.set_title(f'{dataset} - Training Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Validation Loss
        ax = axes[i, 1]
        for optimizer in optimizers:
            key = f"{dataset}_{optimizer}"
            if key in results:
                epochs = range(1, len(results[key]['val_loss']) + 1)
                ax.plot(epochs, results[key]['val_loss'], 
                       label=optimizer, color=colors[optimizer], linestyle=linestyles[optimizer])
        ax.set_title(f'{dataset} - Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Validation Accuracy
        ax = axes[i, 2]
        for optimizer in optimizers:
            key = f"{dataset}_{optimizer}"
            if key in results:
                epochs = range(1, len(results[key]['val_acc']) + 1)
                ax.plot(epochs, results[key]['val_acc'], 
                       label=optimizer, color=colors[optimizer], linestyle=linestyles[optimizer])
        ax.set_title(f'{dataset} - Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main Experiment
if __name__ == "__main__":
    # Initialize results dictionary
    results = {}
    
    # Run experiments for each dataset and optimizer
    for dataset_name in CONFIG.DATASETS:
        # Get dataset properties
        if dataset_name == "MNIST":
            num_classes = 10
            input_channels = 1
        else:  # CIFAR10
            num_classes = 10
            input_channels = 3
        
        # Get data loaders
        train_loader, val_loader, test_loader = get_dataloaders(dataset_name, CONFIG.BATCH_SIZE)
        
        for optimizer_name in CONFIG.OPTIMIZERS:
            print(f"\n===== Training {optimizer_name} on {dataset_name} =====")
            
            # Initialize model, optimizer, and criterion
            model = SimpleCNN(num_classes, input_channels).to(CONFIG.DEVICE)
            criterion = nn.CrossEntropyLoss()
            
            if optimizer_name == "SGD":
                optimizer = optim.SGD(
                    model.parameters(), 
                    lr=CONFIG.LEARNING_RATE_SGD, 
                    momentum=CONFIG.MOMENTUM_SGD
                )
            elif optimizer_name == "Adam":
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=CONFIG.LEARNING_RATE_ADAM
                )
            elif optimizer_name == "PSD":
                optimizer = PSDOptimizer(
                    model.parameters(),
                    lr=CONFIG.LEARNING_RATE_PSD,
                    epsilon=CONFIG.PSD_GRAD_THRESHOLD,
                    r=CONFIG.PSD_PERTURBATION_RADIUS,
                    T=CONFIG.PSD_T,
                )
            
            # Initialize lists to track metrics
            train_losses = []
            val_losses = []
            val_accs = []
            
            # Training loop
            start_time = time.time()
            
            for epoch in range(CONFIG.NUM_EPOCHS):
                # Train for one epoch
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CONFIG.DEVICE)
                
                # Evaluate on validation set
                val_loss, val_acc = evaluate(model, val_loader, criterion, CONFIG.DEVICE)
                
                # Record metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Print progress
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{CONFIG.NUM_EPOCHS}], "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate on test set
            test_loss, test_acc = evaluate(model, test_loader, criterion, CONFIG.DEVICE)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Store results
            key = f"{dataset_name}_{optimizer_name}"
            results[key] = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_acc': val_accs,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'training_time': training_time
            }
    
    # Plot results
    plot_results(results, CONFIG.DATASETS, CONFIG.OPTIMIZERS)
    
    # Print summary table
    print("\n===== SUMMARY =====")
    print(f"{'Dataset':<10} {'Optimizer':<10} {'Test Acc':<10} {'Training Time':<15}")
    print("-" * 45)
    for dataset in CONFIG.DATASETS:
        for optimizer in CONFIG.OPTIMIZERS:
            key = f"{dataset}_{optimizer}"
            if key in results:
                test_acc = results[key]['test_acc']
                training_time = results[key]['training_time']
                print(f"{dataset:<10} {optimizer:<10} {test_acc:<10.2f} {training_time:<15.2f}")
