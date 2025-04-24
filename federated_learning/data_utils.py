import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_data(client_id, num_clients=2, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    
    # Split dataset across clients (i.i.d.)
    np.random.seed(42)  # For reproducibility
    indices = np.arange(len(trainset))
    np.random.shuffle(indices)
    split_size = len(trainset) // num_clients
    client_indices = indices[client_id * split_size:(client_id + 1) * split_size]
    
    client_dataset = Subset(trainset, client_indices)
    client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
    
    return client_loader