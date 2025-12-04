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
from ott.problems.linear import linear_problem
from ott import utils 

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
        
        # 1. Base cost is the distance
        cost = dist
        
        # 2. Clamp valid costs to y_max if configured
        if self.y_max_meters > 0:
             cost = jnp.minimum(cost, self.y_max_meters)

        # 3. Check Constraints
        is_valid = jnp.array(True)
        t_diff = y[2] - x[2] # tB - tA
        
        # Constraint: Future Only (pickups must happen before dropoffs)
        if self.future_only:
            is_valid = is_valid & (t_diff >= 0)
            
        # Constraint: Speed Limit (must be physically possible to reach B from A)
        if self.speed_mps > 0:
            time_needed = dist / self.speed_mps
            is_valid = is_valid & (t_diff >= time_needed)

        # 4. Handle Infeasible Edges
        # Instead of returning infinity (which causes NaNs in Sinkhorn when rows are empty),
        # we return y_max_meters. This acts as a "penalty" for unserved requests.
        penalty = self.y_max_meters if self.y_max_meters > 0 else 1e9
        
        return jnp.where(is_valid, cost, penalty)

    def tree_flatten(self):
        return ((), (self.speed_mps, self.y_max_meters, self.future_only))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi OTT-JAX Experiment")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--date", type=str, default="2014-10-14")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--random-sample", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-1, help="Entropic regularization")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for online computation")
    parser.add_argument("--speed-mps", type=float, default=8.0)
    
    # Updated default to 10km (10,000m) as per your request
    parser.add_argument("--y-max-meters", type=float, default=10000.0) 
    
    parser.add_argument("--future-only", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--inner-iterations", type=int, default=10, help="Print progress every X iterations")
    parser.add_argument("--max-iterations", type=int, default=5000, help="Max sinkhorn iterations")

    args = parser.parse_args()

    # --- Path Resolution ---
    input_path_str = args.input or "./data/2014_Yellow_Taxi_Trip_Data_20251014-3.csv"
    input_path = Path(input_path_str)
    if not input_path.is_absolute():
        input_path = (NYC_TAXI_DIR / input_path).resolve()

    # Load Data
    print(f"Loading data from {input_path}...")
    df, mapping = load_day(
        input_path,
        date=args.date,
        n=args.n,
        random_sample=args.random_sample,
        seed=args.seed,
        logger=print
    )
    
    # Prepare coordinates
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
    
    points_A = np.column_stack((xA_m, tA))
    points_B = np.column_stack((xB_m, tB))
    
    # Convert to JAX
    points_A_jax = jnp.array(points_A)
    points_B_jax = jnp.array(points_B)
    
    print(f"Configuring TaxiCost with y_max={args.y_max_meters}m (used as penalty for infeasible edges)")
    
    cost_fn = TaxiCost(
        speed_mps=args.speed_mps if args.speed_mps else 0.0,
        y_max_meters=args.y_max_meters if args.y_max_meters else 0.0,
        future_only=args.future_only
    )
    
    # Create Geometry with Online Streaming (batch_size)
    geom = pointcloud.PointCloud(
        points_A_jax, 
        points_B_jax, 
        cost_fn=cost_fn, 
        epsilon=args.epsilon,
        batch_size=args.batch_size 
    )
    
    # Create Linear Problem (wraps geometry)
    prob = linear_problem.LinearProblem(geom)
    
    # Solver
    solver = sinkhorn.Sinkhorn(
        progress_fn=utils.default_progress_fn(),
        inner_iterations=args.inner_iterations,
        max_iterations=args.max_iterations
    )
    
    print("Starting solver...")
    start_time = time.perf_counter()
    out = solver(prob)
    out.converged.block_until_ready()
    end_time = time.perf_counter()
    
    print(f"Solver finished in {end_time - start_time:.4f}s")
    print(f"Converged: {out.converged}")
    print(f"Regulated OT Cost: {out.reg_ot_cost}")
    
    results = {
        "runtime_sec": end_time - start_time,
        "converged": bool(out.converged),
        "reg_ot_cost": float(out.reg_ot_cost),
        "primal_cost": float(out.primal_cost) if out.primal_cost is not None else None,
        "n": args.n,
        "epsilon": args.epsilon,
        "y_max": args.y_max_meters
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.out}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()