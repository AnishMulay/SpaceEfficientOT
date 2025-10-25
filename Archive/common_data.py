#!/usr/bin/env python3
import torch
import os

SEED = 42

def get_points(n, d, device, seed=SEED, cache=True):
    """Generate or load cached random points for experiments.
    
    Returns:
        tuple: (xa, xb) where both are torch tensors of shape (n, d)
    """
    if cache:
        os.makedirs("datasets", exist_ok=True)
        cache_path = f"datasets/points_n{n}_d{d}_seed{seed}.pt"
        
        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location=device)
            return data["xa"], data["xb"]
    
    # Generate new data
    torch.manual_seed(seed)
    xa = torch.rand(n, d, device=device)
    xb = torch.rand(n, d, device=device)
    
    if cache:
        torch.save({"xa": xa.cpu(), "xb": xb.cpu()}, cache_path)
        xa = xa.to(device)
        xb = xb.to(device)
    
    return xa, xb