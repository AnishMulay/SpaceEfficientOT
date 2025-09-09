import torch
import numpy as np
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
    
    # Run original matching algorithm
    print("\nRunning matching_torch_v1...")
    Mb, yA, yB, matching_cost, iteration = matching_torch_v1(W_tensor, C, delta, device)
    print(f"Original algorithm completed in {iteration} iterations")
    print(f"Matching cost: {matching_cost.item():.4f}")
    
    # Run SPEF matching
    print("\nRunning spef_matching_torch...")
    spef_Mb, spef_yA, spef_yB, spef_cost, spef_iteration = spef_matching_2(xa, xb, C, k, delta, device)
    print(f"SPEF matching completed in {spef_iteration} iterations")
    print(f"SPEF matching cost: {spef_cost.item():.4f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_matching_algorithms()