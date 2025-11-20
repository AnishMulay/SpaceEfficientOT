#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Tuple

# JAX configuration
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
import ott
from ott.geometry import costs, pointcloud
from ott.solvers.linear import sinkhorn

# Ensure local package imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
NYC_TAXI_DIR = EXPERIMENT_DIR.parent / "nyc_taxi"
if str(NYC_TAXI_DIR) not in sys.path:
    sys.path.insert(0, str(NYC_TAXI_DIR))

from loader import load_day
from prepare import prepare_tensors

EARTH_RADIUS_METERS = 6_371_000.0

def _project_lonlat_to_meters(
    xA_deg: np.ndarray,
    xB_deg: np.ndarray,
    *,
    origin_lon: float | None = None,
    origin_lat: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # xA_deg, xB_deg are numpy arrays [N, 2]
    if origin_lon is None or origin_lat is None:
        lonA, latA = xA_deg[:, 0], xA_deg[:, 1]
        lonB, latB = xB_deg[:, 0], xB_deg[:, 1]
        lon0 = float(np.median(np.concatenate((lonA, lonB))))
        lat0 = float(np.median(np.concatenate((latA, latB))))
    else:
        lon0, lat0 = float(origin_lon), float(origin_lat)

    deg2rad = np.pi / 180.0
    lat0_rad = lat0 * deg2rad
    scale_x = EARTH_RADIUS_METERS * np.cos(lat0_rad)
    scale_y = EARTH_RADIUS_METERS

    def _proj(coords: np.ndarray) -> np.ndarray:
        lon = coords[:, 0] - lon0
        lat = coords[:, 1] - lat0
        x = lon * deg2rad * scale_x
        y = lat * deg2rad * scale_y
        return np.stack((x, y), axis=1)

    return _proj(xA_deg), _proj(xB_deg), lon0, lat0

@dataclass
class ExperimentConfig:
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20251014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 100000
    random_sample: bool = True
    seed: int = 1
    epsilon: float = 1e-3 # Regularization parameter for Sinkhorn
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 100000.0
    future_only: bool = True
    out: str | None = None
    no_warmup: bool = False
    origin_lon: float | None = None
    origin_lat: float | None = None

@jax.tree_util.register_pytree_node_class
class TaxiCost(costs.CostFn):
    """Custom cost function for NYC Taxi experiment."""
    def __init__(self, speed_mps: float, y_max_meters: float, future_only: bool):
        super().__init__()
        self.speed_mps = speed_mps
        self.y_max_meters = y_max_meters
        self.future_only = future_only

    def __call__(self, x, y):
        # x, y are [3] (x_m, y_m, t_s)
        dist = jnp.sqrt(jnp.sum((x[:2] - y[:2]) ** 2))
        
        cost = dist
        
        # Apply y_max clamping/validity
        if self.y_max_meters > 0:
             cost = jnp.minimum(cost, self.y_max_meters)

        # Constraints
        # We use a large value for infinity. 
        # Note: Sinkhorn in log-space handles inf, but standard Sinkhorn might be unstable with actual inf.
        # OTT usually handles infs correctly if using log-domain.
        # However, hard constraints in Sinkhorn are tricky. 
        # We will return a very large cost for invalid pairs.
        
        is_valid = jnp.array(True)
        
        t_diff = y[2] - x[2] # tB - tA
        
        if self.future_only:
            is_valid = is_valid & (t_diff >= 0)
            
        if self.speed_mps > 0:
            # t_diff >= dist / speed  =>  t_diff * speed >= dist
            # Avoid division by zero if t_diff is small? 
            # If t_diff < 0, it's already invalid by future_only (if enabled).
            # If future_only is False, t_diff could be negative.
            # Speed constraint usually implies forward in time? 
            # The reference implementation: mask_speed = dt < time_needed
            # time_needed = dist / speed.
            # So if dt < dist/speed, it's invalid.
            time_needed = dist / self.speed_mps
            is_valid = is_valid & (t_diff >= time_needed)

        # If invalid, return infinity
        # We use a large finite number to avoid NaNs in gradients if that matters, 
        # but for pure solver, inf is fine.
        # OTT uses `jnp.inf` support.
        
        return jnp.where(is_valid, cost, jnp.inf)

    def tree_flatten(self):
        return ((), (self.speed_mps, self.y_max_meters, self.future_only))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi OTT-JAX Experiment")
    parser.add_argument("--input", type=str, default="./data/2014_Yellow_Taxi_Trip_Data_20251014-3.csv")
    parser.add_argument("--date", type=str, default="2014-10-14")
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--random-sample", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-1, help="Entropic regularization")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for online computation")
    parser.add_argument("--speed-mps", type=float, default=8.0)
    parser.add_argument("--y-max-meters", type=float, default=100000.0)
    parser.add_argument("--future-only", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load Data
    print(f"Loading data from {args.input}...")
    df, mapping = load_day(
        args.input,
        date=args.date,
        n=args.n,
        random_sample=args.random_sample,
        seed=args.seed,
        logger=print
    )
    
    # Prepare coordinates (using numpy for projection)
    pickup_coords = df[[mapping.pickup_lon, mapping.pickup_lat]].to_numpy(dtype=np.float32)
    dropoff_coords = df[[mapping.dropoff_lon, mapping.dropoff_lat]].to_numpy(dtype=np.float32)
    
    # Project
    xA_m, xB_m, _, _ = _project_lonlat_to_meters(dropoff_coords, pickup_coords)
    
    # Times
    def _to_unix(series):
        if series.dt.tz is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return (series.view("int64") // 10**9).to_numpy(dtype=np.float64)

    tA = _to_unix(df[mapping.dropoff_time])
    tB = _to_unix(df[mapping.pickup_time])
    
    # Stack into [N, 3] arrays: x, y, t
    # A is dropoff (supply), B is pickup (demand)
    # We match A -> B
    # xA: [N, 3], xB: [M, 3]
    # Note: In OTT, geom(x, y) usually means cost between x[i] and y[j].
    # Here we want cost between dropoff A[i] and pickup B[j].
    
    points_A = np.column_stack((xA_m, tA))
    points_B = np.column_stack((xB_m, tB))
    
    # Convert to JAX
    points_A_jax = jnp.array(points_A)
    points_B_jax = jnp.array(points_B)
    
    # Define Geometry
    cost_fn = TaxiCost(
        speed_mps=args.speed_mps if args.speed_mps else 0.0,
        y_max_meters=args.y_max_meters if args.y_max_meters else 0.0,
        future_only=args.future_only
    )
    
    geom = pointcloud.PointCloud(
        points_A_jax, 
        points_B_jax, 
        cost_fn=cost_fn, 
        epsilon=args.epsilon,
        batch_size=args.batch_size
    )
    
    # Solver
    solver = sinkhorn.Sinkhorn()
    
    print("Starting solver...")
    start_time = time.perf_counter()
    out = solver(geom)
    # Block until ready
    out.converged.block_until_ready()
    end_time = time.perf_counter()
    
    print(f"Solver finished in {end_time - start_time:.4f}s")
    print(f"Converged: {out.converged}")
    print(f"Regulated OT Cost: {out.reg_ot_cost}")
    print(f"Primal Cost: {out.primal_cost}")
    
    # Metrics
    # We want to compare with the existing experiment.
    # Existing experiment reports: feasible matches, free B, removed by constraints, total cost.
    # OTT Sinkhorn is a soft match (entropic). It produces a coupling matrix P.
    # We can't easily get "removed by future" counts from the soft coupling without materializing it or doing extra work.
    # But we can report the total transport cost.
    
    # To get comparable metrics, we might need to round the solution or interpret the marginals.
    # For now, let's report the primal cost and runtime.
    
    results = {
        "runtime_sec": end_time - start_time,
        "converged": bool(out.converged),
        "reg_ot_cost": float(out.reg_ot_cost),
        "primal_cost": float(out.primal_cost),
        "n": args.n,
        "epsilon": args.epsilon
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
