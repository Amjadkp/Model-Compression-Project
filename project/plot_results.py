import pandas as pd
import matplotlib.pyplot as plt

# Read logs from both clients
log_files = ['client1_log.txt', 'client2_log.txt']
data = []

for log_file in log_files:
    client_id = log_file.split('_')[0]  # 'client1' or 'client2'
    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_technique = ''
    for line in lines:
        # Update current technique
        if 'Running federated learning with' in line:
            current_technique = line.split('with ')[1].split(' compression')[0]
            if current_technique == 'no': current_technique = 'none'
        # Process metric lines
        if ', Technique:' in line:
            parts = line.split(', ')
            try:
                round_num = int(parts[1].split('Round ')[1].split('/')[0])
                technique = parts[2].split('Technique: ')[1]
                accuracy = float(parts[3].split('Accuracy: ')[1])
                loss = float(parts[4].split('Loss: ')[1])
                comm_cost = float(parts[5].split('Comm Cost: ')[1].split(' MB')[0])
                data.append({
                    'Client': client_id,
                    'Technique': technique,
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
df_avg = df.groupby(['Technique', 'Round']).agg({
    'Accuracy': 'mean',
    'Loss': 'mean',
    'Comm Cost': 'mean'
}).reset_index()

# Plot Accuracy
plt.figure(figsize=(10, 6))
for technique in df_avg['Technique'].unique():
    subset = df_avg[df_avg['Technique'] == technique]
    plt.plot(subset['Round'], subset['Accuracy'], marker='o', label=technique)
plt.title('Average Client Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/home/tomsy/Desktop/paper/project/avg_accuracy.png')
plt.close()

# Plot Loss
plt.figure(figsize=(10, 6))
for technique in df_avg['Technique'].unique():
    subset = df_avg[df_avg['Technique'] == technique]
    plt.plot(subset['Round'], subset['Loss'], marker='o', label=technique)
plt.title('Average Client Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/home/tomsy/Desktop/paper/project/avg_loss.png')
plt.close()