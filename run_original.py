#!/usr/bin/env python3
import torch
import numpy as np
import time
from scipy.spatial.distance import cdist
from matching import matching_torch_v1

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
delta = 0.01
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"ORIGINAL ALGORITHM")
print(f"Parameters: n={n}, d={d}, delta={delta}, seed={seed}")
print(f"Device: {device}")
print("=" * 50)

# Generate test data
torch.manual_seed(seed)
np.random.seed(seed)
xa = torch.rand(n, d, device=device)
xb = torch.rand(n, d, device=device)
print(f"Generated data: xa.shape={xa.shape}, xb.shape={xb.shape}")

# Calculate cost matrix
xa_cpu = xa.detach().cpu().numpy()
xb_cpu = xb.detach().cpu().numpy()
W = cdist(xb_cpu, xa_cpu, "sqeuclidean")
W_tensor = torch.tensor(W, device=device, dtype=torch.float32)
C = torch.tensor(W.max(), device=device, dtype=torch.float32)
print(f"Cost matrix computed: W.shape={W.shape}, C={float(C):.4f}")

# Run original algorithm
baseline_memory = measure_memory_baseline(device)
t0 = time.perf_counter()
Mb, yA, yB, cost, iterations = matching_torch_v1(W_tensor, C, delta, device)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

runtime = t1 - t0
mem_mb = measure_memory_current(device, baseline_memory)

print(f"\nRESULTS:")
print(f"Runtime: {runtime:.4f} s")
print(f"GPU memory: {mem_mb:.2f} MB")
print(f"Iterations: {int(iterations)}")
print(f"Matching cost: {float(cost):.4f}")