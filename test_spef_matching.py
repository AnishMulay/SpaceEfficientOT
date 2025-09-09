import torch
import numpy as np
import time
from scipy.spatial.distance import cdist
from matching import matching_torch_v1
from spef_matching import spef_matching_2

def test_matching_algorithms():
    # Set parameters
    n = 100
    d = 10
    k = 10  # tile size
    delta = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Testing with n={n}, d={d}, k={k}, delta={delta}")
    
    # Generate random points
    torch.manual_seed(42)
    np.random.seed(42)
    
    xa = torch.rand(n, d, device=device)
    xb = torch.rand(n, d, device=device)
    
    # Calculate cost matrix W using squared Euclidean distance
    xa_cpu = xa.cpu().numpy()
    xb_cpu = xb.cpu().numpy()
    W = cdist(xb_cpu, xa_cpu, 'sqeuclidean')
    W_tensor = torch.tensor(W, device=device, dtype=torch.float32)
    C = torch.tensor(W.max(), device=device, dtype=torch.float32)
    
    print(f"Cost matrix shape: {W_tensor.shape}")
    print(f"Max cost C: {C.item():.4f}")
    
    # Run original matching algorithm with timing and memory tracking
    print("\nRunning matching_torch_v1...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize()
    
    start_time = time.time()
    Mb, yA, yB, matching_cost, iteration = matching_torch_v1(W_tensor, C, delta, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    original_time = end_time - start_time
    original_memory = (torch.cuda.memory_allocated(device) - baseline_memory) / 1024**2 if device.type == 'cuda' else 0
    
    print(f"Original algorithm completed in {iteration} iterations")
    print(f"Original algorithm time: {original_time:.4f} seconds")
    print(f"Original algorithm GPU memory: {original_memory:.2f} MB")
    print(f"Matching cost: {matching_cost.item():.4f}")
    
    # Run SPEF matching with timing and memory tracking
    print("\nRunning spef_matching_2...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize()
    
    start_time = time.time()
    spef_Mb, spef_yA, spef_yB, spef_cost, spef_iteration = spef_matching_2(xa, xb, C, k, delta, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    spef_time = end_time - start_time
    spef_memory = (torch.cuda.memory_allocated(device) - baseline_memory) / 1024**2 if device.type == 'cuda' else 0
    
    print(f"SPEF matching completed in {spef_iteration} iterations")
    print(f"SPEF matching time: {spef_time:.4f} seconds")
    print(f"SPEF matching GPU memory: {spef_memory:.2f} MB")
    print(f"SPEF matching cost: {spef_cost.item():.4f}")
    
    # Summary comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Original vs SPEF - Time: {original_time:.4f}s vs {spef_time:.4f}s (speedup: {original_time/spef_time:.2f}x)")
    print(f"Original vs SPEF - Memory: {original_memory:.2f}MB vs {spef_memory:.2f}MB")
    print(f"Original vs SPEF - Iterations: {iteration} vs {spef_iteration}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_matching_algorithms()