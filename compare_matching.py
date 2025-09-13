#!/usr/bin/env python3
import argparse
import time
import torch
import numpy as np
from scipy.spatial.distance import cdist
from matching import matching_torch_v1
from spef_matching import spef_matching_2

def measure_peak_reset(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

def measure_peak_read(device):
    if device.type == "cuda":
        # bytes; convert to MB in print
        return torch.cuda.max_memory_allocated(device)
    return 0

def run_comparison(d, k, n=10000, delta=0.01, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comparison with n={n}, d={d}, k={k}, delta={delta}")
    print(f"Using device: {device}")
    print("=" * 60)

    # Generate test data
    torch.manual_seed(seed)
    np.random.seed(seed)
    xa = torch.rand(n, d, device=device)
    xb = torch.rand(n, d, device=device)

    # --------------------------
    # Calculate shared C parameter
    # --------------------------
    try:
        # Calculate C using space-efficient method (random B point)
        random_b_idx = torch.randint(0, n, (1,), device=device)
        random_b_point = xb[random_b_idx]
        distances = torch.sum((xa - random_b_point)**2, dim=1)
        C = torch.max(distances) * 2
        print(f"Shared C parameter: {float(C):.4f}")
    except Exception as e:
        print(f"Failed to calculate C parameter: {str(e)}")
        return

    # --------------------------
    # Original algorithm (A)
    # --------------------------
    print("\nORIGINAL ALGORITHM RESULTS:")
    print("-" * 40)
    try:
        # Calculate cost matrix for original algorithm (baseline uses dense W)
        xa_cpu = xa.detach().cpu().numpy()
        xb_cpu = xb.detach().cpu().numpy()
        W = cdist(xb_cpu, xa_cpu, "sqeuclidean")
        W_normalized = W / float(C)  # Normalize cost matrix using shared C
        W_tensor = torch.tensor(W_normalized, device=device, dtype=torch.float32)

        measure_peak_reset(device)
        t0 = time.perf_counter()
        Mb1, yA1, yB1, cost1, iter1 = matching_torch_v1(W_tensor, C, delta, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        peak_bytes = measure_peak_read(device)

        runtime = t1 - t0
        mem_mb = peak_bytes / (1024**2)
        thr = n / runtime if runtime > 0 else float("inf")
        sqrt_cost = float(cost1) ** 0.5

        print(f"Runtime: {runtime:.4f} s")
        print(f"Peak GPU memory: {mem_mb:.2f} MB")
        print(f"Iterations: {int(iter1)}")
        print(f"Throughput: {thr:.0f} points/s")
        print(f"Matching cost: {float(cost1):.4f}")
        print(f"Sqrt matching cost: {sqrt_cost:.4f}")
        print(f"Sqrt matching cost / n: {sqrt_cost / n:.4f}")
    except Exception as e:
        print(f"Original algorithm failed: {str(e)}")

    # --------------------------
    # Space-efficient algorithm (B)
    # --------------------------
    print("\nSPACE-EFFICIENT ALGORITHM RESULTS:")
    print("-" * 40)
    try:

        measure_peak_reset(device)
        t0 = time.perf_counter()
        Mb2, yA2, yB2, cost2, iter2 = spef_matching_2(xa, xb, C, k, delta, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        peak_bytes = measure_peak_read(device)

        runtime = t1 - t0
        mem_mb = peak_bytes / (1024**2)
        thr = n / runtime if runtime > 0 else float("inf")
        sqrt_cost = float(cost2) ** 0.5

        print(f"Runtime: {runtime:.4f} s")
        print(f"Peak GPU memory: {mem_mb:.2f} MB")
        print(f"Iterations: {int(iter2)}")
        print(f"Throughput: {thr:.0f} points/s")
        print(f"Matching cost: {float(cost2):.4f}")
        print(f"Sqrt matching cost: {sqrt_cost:.4f}")
        print(f"Sqrt matching cost / n: {sqrt_cost / n:.4f}")
    except Exception as e:
        print(f"Space-efficient algorithm failed: {str(e)}")

    print("\nTest completed!")

def main():
    parser = argparse.ArgumentParser(description="Compare original vs space-efficient matching algorithms")
    parser.add_argument("--d", type=int, required=True, help="Dimension of points")
    parser.add_argument("--k", type=int, required=True, help="Tile size for space-efficient algorithm")
    parser.add_argument("--n", type=int, required=True, help="Number of points")
    parser.add_argument("--delta", type=float, required=True, help="Scaling factor delta")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    run_comparison(d=args.d, k=args.k, n=args.n, delta=args.delta, seed=args.seed)

if __name__ == "__main__":
    main()