import pandas as pd
import matplotlib.pyplot as plt

# Read logs from both clients
log_files = ['client_log.txt', 'client_log.txt']
client_ids = ['client1', 'client2']
data = []

for log_file, client_id in zip(log_files, client_ids):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if ', Accuracy:' in line:
            parts = line.split(', ')
            try:
                round_num = int(parts[1].split('Round ')[1].split('/')[0])
                accuracy = float(parts[2].split('Accuracy: ')[1])
                loss = float(parts[3].split('Loss: ')[1])
                comm_cost = float(parts[4].split('Comm Cost: ')[1].split(' MB')[0])
                data.append({
                    'Client': client_id,
                    'Round': round_num,
                    'Accuracy': accuracy,
                    'Loss': loss,
                    'Comm Cost': comm_cost
                })
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed line in {log_file}: {line.strip()} (Error: {e})")

# Create DataFrame
df = pd.DataFrame(data)

# Average metrics across clients
df_avg = df.groupby('Round').agg({
    'Accuracy': 'mean',
    'Loss': 'mean',
    'Comm Cost': 'mean'
}).reset_index()

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df_avg['Round'], df_avg['Accuracy'], marker='o', label='Combined Compression')
plt.title('Average Client Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/home/tomsy/Desktop/paper/federated_learning/accuracy.png')
plt.close()

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(df_avg['Round'], df_avg['Loss'], marker='o', label='Combined Compression')
plt.title('Average Client Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/home/tomsy/Desktop/paper/federated_learning/loss.png')
plt.close()