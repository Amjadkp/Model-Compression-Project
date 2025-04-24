import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from model import Net
from data_utils import get_mnist_data
from compression import apply_random_mask, apply_subsampling, apply_quantization, apply_random_rotation
import os
import sys

def client(rank, world_size):
    os.environ['GLOO_SOCKET_IFNAME'] = "eno1"
    os.environ['MASTER_ADDR'] = "172.16.66.131"
    os.environ['MASTER_PORT'] = "6006"
    # Initialize distributed environment
    dist.init_process_group(backend='gloo', init_method='tcp://172.16.66.131:6006', world_size=world_size, rank=rank)
    
    # Load local data
    client_id = rank - 1  
    train_loader = get_mnist_data(client_id)
    
    # Initialize model
    global_model = Net()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Federated Learning loop
    num_rounds = 10
    for round in range(num_rounds):
        print(f"Client {rank}, Round {round + 1}/{num_rounds}")
        
        # Receive global model
        model.eval()
        for param in global_model.parameters():
            dist.broadcast(param.data, src=0)
        
        # Local training
        model.train()
        for i in range(5):  # 5 local epochs
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                print(f"one local epoch completed {i}")
        
        # Compute and compress update
        updates = []
        for param, global_param in zip(model.parameters(), global_model.parameters()):
            print("started")
            # dist.broadcast(global_param.data, src=0)  # Get global model again for update
            update = param.data - global_param.data 
            print("ended")
            # Apply structured update (random mask)
            update, mask = apply_random_mask(update, sparsity=0.25, seed=round + rank)
            # Apply sketched update (subsampling + quantization + rotation)
            update = apply_random_rotation(update, seed=round + rank)
            update, indices = apply_subsampling(update, fraction=0.25, seed=round + rank)
            update, quant_info = apply_quantization(update, bits=2)
            updates.append(update)
        # Send compressed updates to server
        # for update in updates:
            dist.send(tensor=update, dst=0)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    
    world_size = 3  # 1 server + 2 clients
    rank = int(sys.argv[1])  # Pass rank as command-line argument
    client(rank, world_size)