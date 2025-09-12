#!/usr/bin/env python3
import torch
import numpy as np
from spef_matching import spef_matching_2

# Initialize parameters
n = 10000
d = 2
k = 1000
delta = 0.01
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate test data
torch.manual_seed(seed)
np.random.seed(seed)
xa = torch.rand(n, d, device=device)
xb = torch.rand(n, d, device=device)

# Calculate C using random B point
random_b_idx = torch.randint(0, n, (1,), device=device)
random_b_point = xb[random_b_idx]  # [1, d]
distances = torch.sum((xa - random_b_point)**2, dim=1)  # [n]
C_spef = torch.max(distances) * 2

# Run space-efficient algorithm
Mb, yA, yB, cost, iterations = spef_matching_2(xa, xb, C_spef, k, delta, device)

print(f"Space-efficient algorithm cost: {float(cost):.4f}")