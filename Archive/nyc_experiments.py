#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime
from nyc_day_experiment import run_nyc_experiment

# Experiment parameters
INPUT_PATH = "./nyc_data/yellow_tripdata_2014-01.parquet"
ZONES_PATH = "./nyc_data/taxi_zones/taxi_zones.shp"
DATE = "2014-01-10"

# Create timestamped results folder and file
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = "nyc_experiment_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")

# Parameter ranges
N_VALUES = [10000, 20000, 30000]
STOPPING_CONDITION_VALUES = [500, 1000, 2000]
CMAX_VALUES = [1, 5, 10]

def load_results():
    """Load existing results from JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {
        "metadata": {
            "experiment_type": "NYC Taxi Bipartite Matching",
            "date": DATE,
            "timestamp_utc": timestamp,
            "n_values": N_VALUES,
            "stopping_condition_values": STOPPING_CONDITION_VALUES,
            "cmax_values": CMAX_VALUES,
            "description": "Varying n (requests), stopping_condition (free B threshold), and cmax (cost threshold)"
        },
        "experiments": []
    }

def save_results(results):
    """Save results to JSON file."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def run_single_experiment(n, stopping_condition, cmax):
    """Run a single NYC experiment."""
    print(f"Running: n={n}, stopping_condition={stopping_condition}, cmax={cmax}")
    
    try:
        result = run_nyc_experiment(
            input_path=INPUT_PATH,
            zones_path=ZONES_PATH,
            date=DATE,
            n=n,
            cmax=cmax,
            stopping_condition=stopping_condition,
            random_sample=True
        )
        
        # Add experiment parameters
        result.update({
            "stopping_condition": stopping_condition,
            "cmax": cmax,
            "date": DATE
        })
        
        print(f"  Wall time: {result['wall_time']:.3f}s")
        print(f"  Feasible matches: {result['feasible_matches']}")
        print(f"  Unmatched requests: {result['free_B']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return None

def main():
    print("Starting NYC taxi experiments")
    print(f"Data: {DATE}")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print("=" * 60)
    
    results = load_results()
    
    for n in N_VALUES:
        for stopping_condition in STOPPING_CONDITION_VALUES:
            for cmax in CMAX_VALUES:
                result = run_single_experiment(n, stopping_condition, cmax)
                if result:
                    results["experiments"].append(result)
                    save_results(results)
                    print(f"  Saved to {RESULTS_FILE}")
                print("-" * 40)
    
    print("All experiments completed!")
    print(f"Total experiments: {len(results['experiments'])}")

if __name__ == "__main__":
    main()