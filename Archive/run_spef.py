#!/usr/bin/env python3
import torch
import numpy as np
import time
from spef_matching import spef_matching_2

def measure_memory_baseline(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(device)
    return 0

def measure_memory_current(device, baseline):
    if device.type == "cuda":
        return (torch.cuda.memory_allocated(device) - baseline) / 1024**2
    return 0

# Initialize parameters
n = 10000
d = 2
k = 1000
delta = 0.01
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"SPACE-EFFICIENT ALGORITHM")
print(f"Parameters: n={n}, d={d}, k={k}, delta={delta}, seed={seed}")
print(f"Device: {device}")
print("=" * 50)

# Generate test data
torch.manual_seed(seed)
np.random.seed(seed)
xa = torch.rand(n, d, device=device)
xb = torch.rand(n, d, device=device)
print(f"Generated data: xa.shape={xa.shape}, xb.shape={xb.shape}")

# Calculate C using random B point
random_b_idx = torch.randint(0, n, (1,), device=device)
random_b_point = xb[random_b_idx]  # [1, d]
distances = torch.sum((xa - random_b_point)**2, dim=1)  # [n]
C_spef = torch.max(distances) * 2
print(f"C calculated from random B point: C={float(C_spef):.4f}")

# Run space-efficient algorithm
baseline_memory = measure_memory_baseline(device)
t0 = time.perf_counter()
Mb, yA, yB, cost, iterations = spef_matching_2(xa, xb, C_spef, k, delta, device)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

runtime = t1 - t0
mem_mb = measure_memory_current(device, baseline_memory)

print(f"\nRESULTS:")
print(f"Runtime: {runtime:.4f} s")
print(f"GPU memory: {mem_mb:.2f} MB")
print(f"Iterations: {int(iterations)}")
sqrt_cost = float(cost) ** 0.5
print(f"Matching cost: {float(cost):.4f}")
print(f"Sqrt matching cost: {sqrt_cost:.4f}")
print(f"Sqrt matching cost / n: {sqrt_cost / n:.4f}")