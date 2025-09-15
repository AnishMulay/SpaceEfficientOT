#!/usr/bin/env python3
import torch
import numpy as np
import time
import json
import os
from spef_matching import spef_matching_2

# Static parameters
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "experiment_results.json"

# Experiment parameters
N_VALUES = [10000, 25000, 50000, 100000, 200000, 300000, 400000, 500000]
DELTA_VALUES = [0.01, 0.005, 0.001]
D_VALUES = [2, 5, 10]
K_VALUES = [500, 1000, 2000]

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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    xa = torch.rand(n, d, device=DEVICE)
    xb = torch.rand(n, d, device=DEVICE)
    
    # Calculate C using space-efficient method
    random_b_idx = torch.randint(0, n, (1,), device=DEVICE)
    random_b_point = xb[random_b_idx]
    distances = torch.sum((xa - random_b_point)**2, dim=1)
    C = torch.max(distances) * 2
    
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
                        result = run_experiment(n, delta, d, k)
                        results["experiments"].append(result)
                        
                        # Save after each experiment
                        save_results(results)
                        print(f"  Results saved to {RESULTS_FILE}")
                        
                    except Exception as e:
                        print(f"  ERROR in experiment n={n}, delta={delta}, d={d}, k={k}: {str(e)}")
                        continue
                    
                    print("-" * 40)
    
    print("All experiments completed!")
    print(f"Total experiments: {len(results['experiments'])}")

if __name__ == "__main__":
    main()