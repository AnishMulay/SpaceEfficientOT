#!/usr/bin/env python3
import torch
import time
import json
import os
from spef_matching_v2 import spef_matching_2
from common_data import get_points, SEED

# Static parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "experiment_results_v2.json"

# Experiment parameters
N_VALUES = [10000, 25000, 50000, 100000, 200000, 300000, 400000, 500000]
DELTA_VALUES = [0.001]
D_VALUES = [2, 5, 10]
K_VALUES = [1000]

def load_results():
    """Load existing results from JSON file."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": []}

def save_results(results):
    """Save results to JSON file."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def run_experiment(n, delta, d, k):
    """Run a single experiment and return results."""
    print(f"Running experiment: n={n}, delta={delta}, d={d}, k={k}")
    
    # Generate test data
    xa, xb = get_points(n, d, DEVICE, seed=SEED, cache=True)
    
    # Calculate C using space-efficient method
    random_b_idx = torch.randint(0, n, (1,), device=DEVICE)
    random_b_point = xb[random_b_idx]  # [1, d]
    dists = torch.cdist(random_b_point, xa)  # [1, n], Euclidean
    C = dists.max() ** 2  # scalar tensor on DEVICE
    
    # Run space-efficient algorithm
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    Mb, yA, yB, cost, iterations = spef_matching_2(xa, xb, C, k, delta, DEVICE)
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    t1 = time.perf_counter()
    
    runtime = t1 - t0
    matching_cost = float(cost)
    cost_per_n = matching_cost / n
    
    result = {
        "algorithm": "spef_matching",
        "n": n,
        "delta": delta,
        "d": d,
        "k": k,
        "runtime": runtime,
        "matching_cost": matching_cost,
        "cost_per_n": cost_per_n,
        "iterations": int(iterations),
        "C": float(C)
    }
    
    print(f"  Runtime: {runtime:.4f}s, Cost: {matching_cost:.4f}, Cost/n: {cost_per_n:.8f}")
    return result

def main():
    print(f"Starting experiments on device: {DEVICE}")
    print(f"Parameters: SEED={SEED}")
    print("=" * 60)
    
    # Load existing results
    results = load_results()
    
    # Run experiments
    for n in N_VALUES:
        for delta in DELTA_VALUES:
            for d in D_VALUES:
                for k in K_VALUES:
                    try:
                        spef_result = run_experiment(n, delta, d, k)
                        results["experiments"].append(spef_result)
                        save_results(results)
                        print(f" Results saved to {RESULTS_FILE}")
                    except Exception as e:
                        print(f" ERROR in SPEF n={n}, delta={delta}, d={d}, k={k}: {str(e)}")
                    print("-" * 40)
    
    print("All experiments completed!")
    print(f"Total experiments: {len(results['experiments'])}")

if __name__ == "__main__":
    main()