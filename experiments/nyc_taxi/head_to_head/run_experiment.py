#!/usr/bin/env python3
"""
run_experiment.py
=================
Worker script.  Each SLURM job calls this with one (n, method, trial) cell.

Usage (called automatically by submit_experiments.py):
    python run_experiment.py \\
        --n 5000 --method scaled --trial 0 --seed 42 \\
        --results-csv /path/to/raw_results.csv \\
        [other solver flags]

Writes exactly ONE row to --results-csv (thread-safe via fcntl file lock).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any local imports
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent          # head_to_head/
NYC_TAXI_DIR = HERE.parent                              # nyc_taxi/
REPO_ROOT    = NYC_TAXI_DIR.parents[1]                  # repo root
SRC_PATH     = REPO_ROOT / "src"

for p in (str(SRC_PATH), str(NYC_TAXI_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Local imports (order matters: spef_ot before loader)
from spef_ot import match, MatchResult, scaling_match, ScalingMatchResult  # noqa: E402
from loader import load_day                                                  # noqa: E402
from prepare import prepare_tensors                                          # noqa: E402

import spef_ot.kernels.euclidean_speed
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6_371_000.0

# Default data path — override with --data-path if needed
DEFAULT_DATA = NYC_TAXI_DIR / "data" / "2014_Yellow_Taxi_Trip_Data_20251014-3.csv"

# CSV columns written by every job
CSV_FIELDS = [
    "timestamp", "n", "method", "trial", "seed", "status",
    "runtime_sec",
    "total_cost_m", "total_cost_km",
    "avg_cost_m",   "avg_cost_km",
    "feasible_matches", "free_b", "match_rate",
    "phases", "iterations",
    "hostname", "slurm_job_id",
]

# ---------------------------------------------------------------------------
# Coordinate projection (identical to run.py in the JAX experiment)
# ---------------------------------------------------------------------------

def _project_to_meters(
    xA_deg: np.ndarray,
    xB_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local tangent-plane projection from (lon, lat) degrees to meters."""
    lon0 = float(np.median(np.concatenate((xA_deg[:, 0], xB_deg[:, 0]))))
    lat0 = float(np.median(np.concatenate((xA_deg[:, 1], xB_deg[:, 1]))))
    r2d  = np.pi / 180.0
    sx   = EARTH_RADIUS_M * np.cos(lat0 * r2d)
    sy   = EARTH_RADIUS_M

    def _proj(coords: np.ndarray) -> np.ndarray:
        x = (coords[:, 0] - lon0) * r2d * sx
        y = (coords[:, 1] - lat0) * r2d * sy
        return np.stack((x, y), axis=1).astype(np.float32)

    return _proj(xA_deg), _proj(xB_deg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(args: argparse.Namespace):
    """Load and project taxi data.  Returns (xA_m, xB_m, tA, tB) as numpy arrays."""
    data_path = Path(args.data_path) if args.data_path else DEFAULT_DATA
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Set --data-path or place the file at {DEFAULT_DATA}"
        )

    df, mapping = load_day(
        data_path,
        date=args.date,
        n=args.n,
        random_sample=True,
        seed=args.seed,
        logger=print,
    )

    pickup_deg  = df[[mapping.pickup_lon,  mapping.pickup_lat]].to_numpy(np.float32)
    dropoff_deg = df[[mapping.dropoff_lon, mapping.dropoff_lat]].to_numpy(np.float32)

    # dropoffs → A (supply),  pickups → B (demand)
    xA_m, xB_m = _project_to_meters(dropoff_deg, pickup_deg)

    def _to_unix(series):
        if series.dt.tz is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return (series.astype("int64") // 10 ** 9).to_numpy(np.int64)

    tA = _to_unix(df[mapping.dropoff_time])
    tB = _to_unix(df[mapping.pickup_time])

    return xA_m, xB_m, tA, tB, len(df)


# ---------------------------------------------------------------------------
# Solver runners — each returns a flat metrics dict
# ---------------------------------------------------------------------------

def _kernel_kwargs(args: argparse.Namespace, tA: np.ndarray, tB: np.ndarray) -> dict:
    """Build EuclideanSpeedKernel keyword args shared across both GPU methods."""
    return {
        "times_A":     tA,
        "times_B":     tB,
        "speed_mps":   args.speed_mps if args.future_only else None,
        "y_max_meters": args.y_max_meters,
        "future_only": args.future_only,
    }


def _metrics_from_result(
    cost_tensor,           # torch scalar (meters)
    feasible_matches: int,
    n: int,
    iterations: int,
    phases: int | None,
    runtime_sec: float,
) -> dict:
    total_m  = float(cost_tensor)
    total_km = total_m / 1000.0
    avg_m    = total_m  / feasible_matches if feasible_matches > 0 else 0.0
    avg_km   = total_km / feasible_matches if feasible_matches > 0 else 0.0
    return {
        "runtime_sec":      runtime_sec,
        "total_cost_m":     total_m,
        "total_cost_km":    total_km,
        "avg_cost_m":       avg_m,
        "avg_cost_km":      avg_km,
        "feasible_matches": feasible_matches,
        "free_b":           n - feasible_matches,
        "match_rate":       feasible_matches / n if n > 0 else 0.0,
        "phases":           phases,
        "iterations":       iterations,
    }


def run_unscaled(xA_m, xB_m, tA, tB, args) -> dict:
    """Tiled push-relabel solver without ε-scaling."""
    import torch
    device = args.device

    xA_t = torch.from_numpy(xA_m).to(device)
    xB_t = torch.from_numpy(xB_m).to(device)

    t0 = time.perf_counter()
    result: MatchResult = match(
        xA_t, xB_t,
        kernel  = "euclidean_speed",
        C       = args.C,
        k       = args.k,
        delta   = args.delta,
        device  = device,
        seed    = args.seed,
        fill_policy = args.fill_policy,
        **_kernel_kwargs(args, tA, tB),
    )
    runtime = time.perf_counter() - t0

    feasible = int(result.metrics.get("feasible_matches",
                                      int((result.Mb != -1).sum().item())))
    return _metrics_from_result(
        result.matching_cost, feasible, len(xB_m),
        result.iterations, phases=None, runtime_sec=runtime,
    )


def run_scaled(xA_m, xB_m, tA, tB, args) -> dict:
    """Tiled push-relabel solver with ε-scaling (SpefOT Scaled)."""
    import torch
    device = args.device

    xA_t = torch.from_numpy(xA_m).to(device)
    xB_t = torch.from_numpy(xB_m).to(device)

    t0 = time.perf_counter()
    result: ScalingMatchResult = scaling_match(
        xA_t, xB_t,
        kernel        = "euclidean_speed",
        C             = args.C,
        k             = args.k,
        target_delta  = args.delta,
        initial_delta = None,          # uses 16× heuristic from solver_scaling.py
        device        = device,
        seed          = args.seed,
        fill_policy   = args.fill_policy,
        verbose       = True,
        **_kernel_kwargs(args, tA, tB),
    )
    runtime = time.perf_counter() - t0

    feasible = int(result.metrics.get("feasible_matches",
                                      int((result.Mb != -1).sum().item())))
    return _metrics_from_result(
        result.matching_cost, feasible, len(xB_m),
        result.iterations, phases=result.phases, runtime_sec=runtime,
    )


def run_exact(xA_m, xB_m, tA, tB, args) -> dict:
    """Exact minimum-cost assignment via scipy.optimize.linear_sum_assignment.

    Builds the full n×n cost matrix on CPU.  Infeasible edges are assigned a
    high penalty (2 × y_max_meters) so the algorithm avoids them while still
    producing a valid complete assignment.
    """
    from scipy.optimize import linear_sum_assignment

    n = len(xA_m)
    mem_bytes = n * n * 4   # float32
    print(f"[exact] building {n}×{n} cost matrix "
          f"({mem_bytes / 1e6:.0f} MB float32) …")

    # Pairwise Euclidean distances [n, n]
    # Use float64 intermediates to avoid catastrophic cancellation
    diff = (xB_m[:, None, :].astype(np.float64)
            - xA_m[None, :, :].astype(np.float64))      # [n, n, 2]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)   # [n, n]

    # Temporal feasibility masks
    dt_mat = tB[:, None] - tA[None, :]   # [n, n] int64

    penalty = float(args.y_max_meters) * 2.0
    cost    = dist.copy()

    if args.future_only:
        cost[dt_mat < 0] = penalty

    if args.speed_mps > 0:
        time_needed = dist / float(args.speed_mps)   # seconds
        cost[dt_mat < time_needed] = penalty

    if args.y_max_meters > 0:
        feasible_mask = cost < penalty
        cost[feasible_mask] = np.minimum(cost[feasible_mask],
                                         float(args.y_max_meters))

    print(f"[exact] running scipy linear_sum_assignment (n={n}) …")
    t0 = time.perf_counter()
    row_ind, col_ind = linear_sum_assignment(cost)
    runtime = time.perf_counter() - t0

    matched_costs = cost[row_ind, col_ind]
    # Only count matches that are truly feasible (cost < penalty threshold)
    feasible_mask_1d = matched_costs < penalty
    feasible_matches  = int(feasible_mask_1d.sum())
    total_cost_m      = float(dist[row_ind[feasible_mask_1d],
                                   col_ind[feasible_mask_1d]].sum())

    return _metrics_from_result(
        np.float64(total_cost_m), feasible_matches, n,
        iterations=1, phases=None, runtime_sec=runtime,
    )


# ---------------------------------------------------------------------------
# Thread-safe CSV append (identical pattern to run_one.py)
# ---------------------------------------------------------------------------

def _append_csv(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        needs_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one (n, method, trial) experiment cell")

    # Identity
    p.add_argument("--n",           type=int,   required=True)
    p.add_argument("--method",      choices=["unscaled", "scaled", "exact"], required=True)
    p.add_argument("--trial",       type=int,   required=True)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--results-csv", type=str,   required=True)

    # Data
    p.add_argument("--data-path",   type=str,   default=None,
                   help="Override default taxi CSV path")
    p.add_argument("--date",        type=str,   default="2014-10-14")

    # Solver shared settings
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--k",             type=int,   default=512)
    p.add_argument("--delta",         type=float, default=0.001)
    p.add_argument("--C",             type=float, default=100000.0)
    p.add_argument("--speed-mps",     type=float, default=8.0)
    p.add_argument("--y-max-meters",  type=float, default=10000.0)
    p.add_argument("--fill-policy",   type=str,   default="none")
    p.add_argument("--future-only",   action="store_true",  default=True)
    p.add_argument("--no-future-only",dest="future_only", action="store_false")
    p.add_argument("--max-exact-n",   type=int,   default=5000,
                   help="Skip exact solver if n exceeds this (prevents CPU OOM)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    csv_path = Path(args.results_csv)

    base_row: dict[str, Any] = {
        "timestamp":   dt.datetime.now().isoformat("T", "seconds"),
        "n":           args.n,
        "method":      args.method,
        "trial":       args.trial,
        "seed":        args.seed,
        "hostname":    socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }

    # Guard: skip exact if n is too large
    if args.method == "exact" and args.n > args.max_exact_n:
        print(f"[skip] n={args.n} > max_exact_n={args.max_exact_n}; "
              f"writing status=skipped")
        _append_csv({**base_row, "status": "skipped"}, csv_path)
        return

    # Load data
    try:
        xA_m, xB_m, tA, tB, n_actual = _load_data(args)
        base_row["n"] = n_actual    # update with actual loaded count
    except Exception:
        traceback.print_exc()
        _append_csv({**base_row, "status": "data_error"}, csv_path)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  n={n_actual}  method={args.method}  trial={args.trial}  seed={args.seed}")
    print(f"  device={args.device}  delta={args.delta}  C={args.C}")
    print(f"{'='*60}\n")

    # Run solver
    try:
        runners = {
            "unscaled": run_unscaled,
            "scaled":   run_scaled,
            "exact":    run_exact,
        }
        metrics = runners[args.method](xA_m, xB_m, tA, tB, args)
        row     = {**base_row, "status": "success", **metrics}

    except MemoryError:
        print("[OOM] MemoryError — likely cost-matrix allocation for exact solver")
        traceback.print_exc()
        row = {**base_row, "status": "oom"}

    except Exception as exc:
        # Catch CUDA OOM (subclass of RuntimeError) and any other failures
        msg = str(exc)
        status = "cuda_oom" if "out of memory" in msg.lower() else "failed"
        print(f"[{status}] {type(exc).__name__}: {msg}")
        traceback.print_exc()
        row = {**base_row, "status": status}

    _append_csv(row, csv_path)
    status = row.get("status", "unknown")
    print(f"\n→ Wrote row to {csv_path}  [status={status}]")

    if status not in ("success",):
        sys.exit(1)


if __name__ == "__main__":
    main()