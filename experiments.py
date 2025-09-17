#!/usr/bin/env python3
import torch
import numpy as np
import time
import json
import os
from spef_matching import spef_matching_2
from geomloss import SamplesLoss
import math

# Static parameters
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "experiment_results.json"

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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    xa = torch.rand(n, d, device=DEVICE)
    xb = torch.rand(n, d, device=DEVICE)
    
    # Calculate C using space-efficient method
    random_b_idx = torch.randint(0, n, (1,), device=DEVICE)
    random_b_point = xb[random_b_idx]  # [1, d]
    dists = torch.cdist(random_b_point, xa)  # [1, n], Euclidean
    C = dists.max() ** 2  # scalar tensor on DEVICE
    
    # Run space-efficient algorithm
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    Mb, yA, yB, cost, iterations, timing_metrics = spef_matching_2(xa, xb, C, k, delta, DEVICE)
    
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
        "C": float(C),
        "timing_metrics": timing_metrics
    }
    
    print(f"  Runtime: {runtime:.4f}s, Cost: {matching_cost:.4f}, Cost/n: {cost_per_n:.8f}")
    return result

def run_geomloss_experiment(n, delta, d, k):
    """
    GeomLoss baseline: Sinkhorn divergence with squared L2 ground cost.
    Returns a dict with identical keys as run_experiment, plus geomloss_config, and algorithm tag "geomloss_sinkhorn".
    The per-n cost (L) is returned as cost_per_n; matching_cost is n * L to compare with the sum from spef_matching_2.
    """
    print(f"Running GeomLoss: n={n}, d={d}, delta={delta}, k={k}")  # delta,k included for bookkeeping parity.

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    xa = torch.rand(n, d, device=DEVICE)
    xb = torch.rand(n, d, device=DEVICE)

    # Backend and parameters chosen per GeomLoss docs for unit-cube inputs and scalability:
    backend = "multiscale" if d <= 3 else "online"
    blur = 0.05          # sigma; epsilon = blur**p for p=2
    scaling = 0.9        # accuracy-oriented epsilon-scaling
    diameter = math.sqrt(d)

    Loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        cost="SqDist(X,Y)",   # matches pure squared L2 scale (default would be SqDist/2)
        debias=True,
        blur=blur,
        scaling=scaling,
        diameter=diameter,
        backend=backend,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    L = Loss(xa, xb)          # uniform weights by default -> average cost per unit mass
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    runtime = t1 - t0
    cost_per_n = float(L)     # average cost
    matching_cost = cost_per_n * n  # sum-style to match spef output

    result = {
        "algorithm": "geomloss_sinkhorn",
        "n": n,
        "delta": delta,
        "d": d,
        "k": k,
        "runtime": runtime,
        "matching_cost": matching_cost,
        "cost_per_n": cost_per_n,
        # Fields kept for parity with spef run; C and timing_metrics are not used here.
        "iterations": 0,
        "C": None,
        "timing_metrics": {},
        "geomloss_config": {
            "loss": "sinkhorn",
            "p": 2,
            "cost": "SqDist(X,Y)",
            "debias": True,
            "blur": blur,
            "scaling": scaling,
            "diameter": diameter,
            "backend": backend
        }
    }
    print(f" GeomLoss Runtime: {runtime:.4f}s, Cost: {matching_cost:.4f}, Cost/n: {cost_per_n:.8f}")
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

                    try:
                        geom_result = run_geomloss_experiment(n, delta, d, k)
                        results["experiments"].append(geom_result)
                        save_results(results)
                        print(f" Results saved to {RESULTS_FILE}")
                    except Exception as e:
                        import traceback
                        print(f" ERROR in GeomLoss n={n}, d={d}: {str(e)}")
                        print("Full traceback:")
                        traceback.print_exc()
                    print("-" * 40)
    
    print("All experiments completed!")
    print(f"Total experiments: {len(results['experiments'])}")

if __name__ == "__main__":
    main()