import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Read JSON data
with open('./experiment_results_v2.json', 'r') as f:
    data = json.load(f)

# Aggregate data by (n, delta)
aggregated = defaultdict(list)
for exp in data['experiments']:
    key = (exp['n'], exp['delta'])
    aggregated[key].append(exp['runtime'])

# Average runtimes for each (n, delta) pair
averaged_data = {}
for key, runtimes in aggregated.items():
    averaged_data[key] = sum(runtimes) / len(runtimes)

# Separate data for two delta values
delta_101_data = []
delta_001_data = []

for (n, delta), runtime in averaged_data.items():
    if delta == 0.01:  # Substitute 0.01 for 0.101 if needed
        delta_101_data.append((n, runtime))
    elif delta == 0.001:
        delta_001_data.append((n, runtime))

# Sort by n
delta_101_data.sort()
delta_001_data.sort()

# Create plot
plt.figure(figsize=(10, 6))
if delta_101_data:
    n_vals, runtime_vals = zip(*delta_101_data)
    plt.plot(n_vals, runtime_vals, 'o-', label='delta=0.01', marker='o')
if delta_001_data:
    n_vals, runtime_vals = zip(*delta_001_data)
    plt.plot(n_vals, runtime_vals, 's-', label='delta=0.001', marker='s')

plt.xlabel('N (elements)')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Create plots directory and save
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/runtime_vs_n.png')
plt.savefig('plots/runtime_vs_n.svg')
plt.show()