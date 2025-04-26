import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv
import os
import subprocess
import time
import signal
from model import Net
from compression import apply_random_mask, apply_subsampling, apply_quantization, apply_random_rotation

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader.dataset)
    return accuracy, loss if not torch.isnan(torch.tensor(loss)) else float('inf')

def calculate_tensor_size(tensor):
    return tensor.element_size() * tensor.numel() / (1024 * 1024)  # Size in MB

def server_with_metrics(rank, world_size, results_file='results.csv'):
    os.environ['GLOO_SOCKET_IFNAME'] = "eno1"
    os.environ['MASTER_ADDR'] = "172.16.66.131"
    os.environ['MASTER_PORT'] = "6006"
    
    print(f"Server: Initializing process group (rank={rank}, world_size={world_size})")
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank, timeout=torch.distributed.timedelta(seconds=180))
    print("Server: Process group initialized")
    
    # Initialize test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = torchvision.datasets.MNIST(root='/home/tomsy/Desktop/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=1000, shuffle=False)
    
    # Initialize CSV file
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Test_Accuracy', 'Test_Loss', 'Comm_Cost_MB'])
    
    # Initialize model
    model = Net()
    model.eval()
    
    total_comm_cost = 0.0
    num_rounds = 25
    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        
        # Broadcast current model to clients
        comm_cost_round = 0
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
            comm_cost_round += calculate_tensor_size(param.data)
        
        # Receive updates from clients
        updates = [torch.zeros_like(param) for param in model.parameters()]
        num_valid_clients = 0
        for client_rank in range(1, world_size):
            client_success = True
            for i, param in enumerate(model.parameters()):
                compressed_update = torch.zeros_like(param)
                try:
                    dist.recv(tensor=compressed_update, src=client_rank)
                    comm_cost_round += calculate_tensor_size(compressed_update)
                    update = compressed_update  # No decompression (matches server.py)
                    updates[i] += update
                except Exception as e:
                    print(f"Server: Error receiving update from client {client_rank}: {e}")
                    client_success = False
                    break
            if client_success:
                num_valid_clients += 1
        
        # Aggregate updates
        if num_valid_clients > 0:
            for i, param in enumerate(model.parameters()):
                updates[i] /= num_valid_clients
                param.data += updates[i]
        
        # Evaluate model
        accuracy, loss = evaluate_model(model, test_loader)
        total_comm_cost += comm_cost_round
        
        # Save metrics
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round + 1, accuracy, loss, total_comm_cost])
        
        print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}, Total Comm Cost: {total_comm_cost:.2f} MB")
    
    # Save final model
    torch.save(model.state_dict(), 'global_model.pth')
    print("Server: Destroying process group")
    dist.destroy_process_group()

def run_server():
    world_size = 3
    client1_ip = "172.16.66.212"
    client2_ip = "172.16.66.250"
    client1_script = "/home/tomsy/Desktop/paper/federated_learning/client.py 1"
    client2_script = "/home/tomsy/Desktop/paper/federated_learning/client.py 2"
    ssh_cmd_1 = (
        f"ssh -o StrictHostKeyChecking=no tomsy@{client1_ip} "
        f"'source /home/tomsy/Desktop/paper/venv/bin/activate && "
        f"python3 {client1_script} > /home/tomsy/Desktop/paper/federated_learning/client1_log.txt 2>&1'"
    )
    ssh_cmd_2 = (
        f"ssh -o StrictHostKeyChecking=no tomsy@{client2_ip} "
        f"'source /home/tomsy/Desktop/paper/venv/bin/activate && "
        f"python3 {client2_script} > /home/tomsy/Desktop/paper/federated_learning/client2_log.txt 2>&1'"
    )
    
    print("Starting client 1...")
    client1_process = subprocess.Popen(ssh_cmd_1, shell=True, preexec_fn=os.setsid)
    print("Starting client 2...")
    client2_process = subprocess.Popen(ssh_cmd_2, shell=True, preexec_fn=os.setsid)
    
    time.sleep(15)
    
    try:
        print("Starting server with metrics...")
        server_with_metrics(0, world_size)
    except KeyboardInterrupt:
        print("Terminating processes...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        try:
            os.killpg(os.getpgid(client1_process.pid), signal.SIGTERM)
            os.killpg(os.getpgid(client2_process.pid), signal.SIGTERM)
            client1_process.wait(timeout=5)
            client2_process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            print("Warning: Some client processes did not terminate cleanly")

if __name__ == '__main__':
    run_server()