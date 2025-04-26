import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from model import Net
from data_utils import get_mnist_data
from compression import apply_random_mask, apply_subsampling, apply_quantization, apply_random_rotation
import os
import sys
import logging
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

# Set up logging to log the output to a file
logging.basicConfig(filename='/home/tomsy/Desktop/paper/federated_learning/client_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def evaluate(model, test_loader):
    # Evaluate the model on the test set
    model.eval()
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            total_loss += criterion(output, target).item()
            _, preds = torch.max(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            total_samples += target.size(0)
    
    accuracy = accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss

def calculate_tensor_size(tensor):
    # Calculate tensor size in bytes and convert to MB
    return tensor.element_size() * tensor.numel() / (1024 * 1024)

def client(rank, world_size):
    os.environ['GLOO_SOCKET_IFNAME'] = "eno1"
    os.environ['MASTER_ADDR'] = "172.16.66.131"
    os.environ['MASTER_PORT'] = "6006"
    # Initialize distributed environment
    dist.init_process_group(backend='gloo', init_method='tcp://172.16.66.131:6006', world_size=world_size, rank=rank)
    
    # Load local data
    client_id = rank - 1  # Clients are 1-indexed, server is rank 0
    train_loader = get_mnist_data(client_id)
    
    # Load the MNIST test set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST(root='/home/tomsy/Desktop/paper/federated_learning/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    global_model = Net()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Federated Learning loop
    num_rounds = 10
    for round in range(num_rounds):
        logging.info(f"Client {rank}, Round {round + 1}/{num_rounds}")
        print(f"Client {rank}, Round {round + 1}/{num_rounds}")
        
        # Receive global model
        model.eval()
        comm_cost_mb = 0.0
        for param in global_model.parameters():
            dist.broadcast(param.data, src=0)
            comm_cost_mb += calculate_tensor_size(param.data)
        
        # Local training
        model.train()
        for i in range(5):  # 5 local epochs
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate model on test set
        accuracy, loss = evaluate(model, test_loader)
        
        # Compute and compress update
        updates = []
        for param, global_param in zip(model.parameters(), global_model.parameters()):
            update = param.data - global_param.data
            # Apply structured update (random mask)
            update, mask = apply_random_mask(update, sparsity=0.25, seed=round + rank)
            # Apply sketched update (subsampling + quantization + rotation)
            update = apply_random_rotation(update, seed=round + rank)
            update, indices = apply_subsampling(update, fraction=0.25, seed=round + rank)
            update, quant_info = apply_quantization(update, bits=2)
            updates.append(update)
            # Send compressed update to server
            dist.send(tensor=update, dst=0)
            comm_cost_mb += calculate_tensor_size(update)
        
        # Log metrics for the round
        logging.info(f"Client {rank}, Round {round + 1}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, Comm Cost: {comm_cost_mb:.2f} MB")
        print(f"Client {rank}, Round {round + 1}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, Comm Cost: {comm_cost_mb:.2f} MB")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 3  # 1 server + 2 clients
    rank = int(sys.argv[1])  # Pass rank as command-line argument
    client(rank, world_size)