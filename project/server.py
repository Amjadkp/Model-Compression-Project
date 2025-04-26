
import torch
import torch.distributed as dist
import torch.nn as nn
from model import Net
from compression import apply_random_mask, apply_subsampling, apply_quantization, apply_random_rotation
import os

def server(rank, world_size):
    os.environ['GLOO_SOCKET_IFNAME'] = "eno1"
    os.environ['MASTER_ADDR'] = "172.16.66.131"
    os.environ['MASTER_PORT'] = "6006"
    
    # Initialize distributed environment
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)
    
    # Initialize model
    model = Net()
    model.eval()

    # Federated Learning loop
    num_rounds = 25
    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")

        # Broadcast current model to clients
        for param in model.parameters():
            dist.broadcast(param.data, src=0)  # Server is rank 0
        
        # Receive updates from clients
        updates = [torch.zeros_like(param) for param in model.parameters()]
        for client_rank in range(1, world_size):
            for i, param in enumerate(model.parameters()):
                # Receive compressed update
                compressed_update = torch.zeros_like(param)
                dist.recv(tensor=compressed_update, src=client_rank)
                
                # Decompress update (e.g., apply inverse rotation, subsampling, etc.)
                update = compressed_update  # Placeholder; add decompression logic
                updates[i] += update
        
        # Aggregate updates (average)
        for i, param in enumerate(model.parameters()):
            updates[i] /= (world_size - 1)  # Average over clients
            param.data += updates[i]  # Update global model

    # Save final model (optional)
    # torch.save(model.state_dict(), 'global_model.pth')

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 3  # 1 server + 2 clients
    server(0, world_size)