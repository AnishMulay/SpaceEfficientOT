#!/usr/bin/env python3
import torch
import time
import json
import os
import math
import numpy as np
from geomloss import SamplesLoss
from common_data import get_points, SEED

# Static parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "results_geomloss.json"

# Experiment parameters
N_VALUES = [10000, 25000, 50000, 100000, 200000, 300000, 400000, 500000]
D_VALUES = [2, 5, 10]

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

def run_geomloss_experiment(n, d):
    """Run GeomLoss experiment and return results."""
    print(f"Running GeomLoss: n={n}, d={d}")
    
    # Load identical test data as SPEF experiments
    xa, xb = get_points(n, d, DEVICE, seed=SEED, cache=True)
    
    Loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        cost="SqDist(X,Y)",
        debias=False,
        blur=0.05,
        scaling=0.9,
        diameter=math.sqrt(d),
        backend="online"
    )
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    L = Loss(xa, xb)
    
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    t1 = time.perf_counter()
    
    runtime = t1 - t0
    cost_per_n = float(L)
    matching_cost = cost_per_n * n
    
    result = {
        "algorithm": "geomloss_sinkhorn",
        "n": n,
        "delta": None,
        "d": d,
        "k": None,
        "runtime": runtime,
        "matching_cost": matching_cost,
        "cost_per_n": cost_per_n,
        "iterations": 0,
        "C": None,
        "timing_metrics": {},
        "geomloss_config": {
            "loss": "sinkhorn",
            "p": 2,
            "cost": "SqDist(X,Y)",
            "debias": False,
            "blur": 0.05,
            "scaling": 0.9,
            "diameter": math.sqrt(d),
            "backend": "online",
            "epsilon": 0.05**2
        }
    }
    
    print(f"  Runtime: {runtime:.4f}s, Cost: {matching_cost:.4f}, Cost/n: {cost_per_n:.8f}")
    return result

def main():
    print(f"Starting GeomLoss experiments on device: {DEVICE}")
    print(f"Parameters: SEED={SEED}")
    print("=" * 60)
    
    # Load existing results
    results = load_results()
    
    # Run experiments
    for n in N_VALUES:
        for d in D_VALUES:
            try:
                geom_result = run_geomloss_experiment(n, d)
                results["experiments"].append(geom_result)
                save_results(results)
                print(f"  Results saved to {RESULTS_FILE}")
            except Exception as e:
                print(f"  ERROR in GeomLoss n={n}, d={d}: {str(e)}")
            print("-" * 40)
    
    print("All GeomLoss experiments completed!")
    print(f"Total experiments: {len(results['experiments'])}")

if __name__ == "__main__":
    main()