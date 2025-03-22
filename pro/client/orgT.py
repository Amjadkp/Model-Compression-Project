import torch
import torch.distributed as dist
import os
import time
import logging
from datetime import timedelta
import torch.nn as nn
import torch.nn.functional as F
import io 


logging.basicConfig(filename="/app/client.log",format='%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d_%H-%M-%S')


class Initialize:
    def __init__(self, rank, world_size, master_addr="client", master_port="6000", backend="gloo", iface="eth0"):
        os.environ['GLOO_SOCKET_IFNAME'] = iface
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        self.backend = backend
        self.rank = rank
        self.world_size = world_size

    def init_connection(self):
        try:
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
                timeout=timedelta(seconds=60)
            )
            print(f"Rank {self.rank}: Initialized and ready to communicate.")
            return dist.is_initialized()
        except Exception as e:
            print(f"Rank {self.rank}: Initialization failed - {e}")
            return False

    def terminate_connection(self):
        try:
            dist.destroy_process_group()
            print(f"Rank {self.rank}: Connection terminated successfully.")
        except Exception as e:
            print(f"Rank {self.rank}: Error in termination - {e}")

    def send(self, arr, dst):
        try:
            dist.send(tensor=arr, dst=dst)
            print(f"Rank {self.rank}: Tensor sent successfully to rank {dst}.")
        except Exception as e:
            print(f"Rank {self.rank}: Error sending tensor to rank {dst} - {e}")
    
    def send_model(self,model,dst):
        for key,param in model.state_dict().items():
            dist.send(param,dst=dst)
            print(f"{key} sended")
    
    def recv_model(self,model,src):
        for key,param in model.state_dict().items():
            dist.recv(param.data,src=src)
            print(f"{key} recieved")
        
class Net(nn.Module):
    """ Network architecture. """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

rank = 0
world_size = 2
process = Initialize(rank=rank, world_size=world_size)
process.init_connection()

model = Net()

start_time = time.time()
process.send_model(model,dst=1)
end_time = time.time()

tot_time = (end_time - start_time)

logging.info(f"model not zipped send time = {tot_time}")