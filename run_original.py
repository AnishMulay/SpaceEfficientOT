#!/usr/bin/env python3
import torch
import numpy as np
from scipy.spatial.distance import cdist
from matching import matching_torch_v1

# Initialize parameters
n = 10000
d = 2
delta = 0.01
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate test data
torch.manual_seed(seed)
np.random.seed(seed)
xa = torch.rand(n, d, device=device)
xb = torch.rand(n, d, device=device)

# Calculate cost matrix
xa_cpu = xa.detach().cpu().numpy()
xb_cpu = xb.detach().cpu().numpy()
W = cdist(xb_cpu, xa_cpu, "sqeuclidean")
W_tensor = torch.tensor(W, device=device, dtype=torch.float32)
C = torch.tensor(W.max(), device=device, dtype=torch.float32)

# Run original algorithm
Mb, yA, yB, cost, iterations = matching_torch_v1(W_tensor, C, delta, device)

print(f"Original algorithm cost: {float(cost):.4f}")