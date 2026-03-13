#!/usr/bin/env python3
"""Partial Wasserstein baseline via POT — NYC taxi POT comparison."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ot

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
NYC_TAXI_DIR = EXPERIMENT_DIR.parent / "nyc_taxi"
if str(NYC_TAXI_DIR) not in sys.path:
    sys.path.insert(0, str(NYC_TAXI_DIR))

from loader import load_day        # noqa: E402
from prepare import prepare_tensors  # noqa: E402

EARTH_RADIUS_METERS = 6_371_000.0

# ---------------------------------------------------------------------------
# Coordinate projection (identical to run_spef.py)
# ---------------------------------------------------------------------------

def _project_lonlat_to_meters(xA_deg, xB_deg, *, origin_lon=None, origin_lat=None):
    if origin_lon is None or origin_lat is None:
        all_lon = torch.cat((xA_deg[:, 0], xB_deg[:, 0]))
        all_lat = torch.cat((xA_deg[:, 1], xB_deg[:, 1]))
        lon0 = float(torch.median(all_lon).item())
        lat0 = float(torch.median(all_lat).item())
    else:
        lon0, lat0 = float(origin_lon), float(origin_lat)

    deg2rad = torch.tensor(torch.pi / 180.0, device=xA_deg.device, dtype=torch.float32)
    lat0_rad = torch.tensor(lat0 * (torch.pi / 180.0), device=xA_deg.device, dtype=torch.float32)
    scale_x = float(EARTH_RADIUS_METERS) * torch.cos(lat0_rad)
    scale_y = float(EARTH_RADIUS_METERS)

    def _proj(coords):
        lon = coords[:, 0] - lon0
        lat = coords[:, 1] - lat0
        x = lon.to(dtype=torch.float32) * deg2rad * scale_x
        y = lat.to(dtype=torch.float32) * deg2rad * scale_y
        return torch.stack((x, y), dim=1)

    return _proj(xA_deg), _proj(xB_deg), lon0, lat0


# ---------------------------------------------------------------------------
# Cost matrix build with the same feasibility logic as run_ortools.py
# ---------------------------------------------------------------------------

def _solve(
    xA_np: np.ndarray,
    xB_np: np.ndarray,
    tA_np: np.ndarray,
    tB_np: np.ndarray,
    speed_mps: float | None,
    y_max_meters: float | None,
    future_only: bool,
    top_k: int,
    log,
) -> dict[str, Any]:
    n = len(xA_np)
    if y_max_meters is None:
        raise ValueError("run_pot.py requires --y-max-meters so the POT penalty is well-defined.")
    if not (0 <= top_k <= n):
        raise ValueError(f"--top-k must satisfy 0 <= top_k <= n_used ({n}), got {top_k}.")

    penalty = float(y_max_meters) * 2.0
    log(f"Building dense cost matrix: n={n}, top_k={top_k}")
    t_build_start = time.perf_counter()

    diff = xB_np[:, None, :] - xA_np[None, :, :]
    M_raw = np.sqrt((diff * diff).sum(axis=2)).astype(np.float64, copy=False)
    dt = tB_np[:, None].astype(np.float64) - tA_np[None, :].astype(np.float64)

    feasible = np.ones((n, n), dtype=bool)
    if future_only:
        feasible &= dt >= 0
    feasible &= M_raw < float(y_max_meters)
    if speed_mps is not None:
        pos_dt = dt > 0
        feasible &= pos_dt | (M_raw < 1e-3)
        max_dist = np.where(pos_dt, float(speed_mps) * dt, 0.0)
        feasible &= M_raw <= max_dist + 1e-3

    M = np.full((n, n), penalty, dtype=np.float64)
    M[feasible] = np.minimum(M_raw[feasible], float(y_max_meters))
    num_feasible = int(feasible.sum())
    build_time = time.perf_counter() - t_build_start
    log(f"Cost matrix built: {num_feasible} feasible entries  [{build_time:.2f}s]")

    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(n, dtype=np.float64) / n
    m = float(top_k) / float(n) if n > 0 else 0.0

    log("Solving POT partial Wasserstein...")
    t_solve_start = time.perf_counter()
    gamma = ot.partial_wasserstein(a, b, M, m=m)
    solve_time = time.perf_counter() - t_solve_start
    log(f"Solve finished in {solve_time:.3f}s")

    opt_cost_m = float(np.sum(gamma * M_raw) * n)
    opt_cost_km = opt_cost_m / 1000.0

    return {
        "build_time_sec": build_time,
        "solve_time_sec": solve_time,
        "num_feasible_edges": num_feasible,
        "top_k": top_k,
        "opt_cost_m": opt_cost_m,
        "opt_cost_km": opt_cost_km,
        "opt_avg_cost_m": (opt_cost_m / top_k) if top_k > 0 else None,
        "opt_avg_cost_km": (opt_cost_km / top_k) if top_k > 0 else None,
    }


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20141014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 1000
    random_sample: bool = True
    seed: int = 1
    top_k: int | None = None
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 10000.0
    future_only: bool = True
    out: str | None = None
    origin_lon: float | None = None
    origin_lat: float | None = None


DEFAULT_CONFIG = ExperimentConfig()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NYC taxi partial Wasserstein solver via POT")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    p.add_argument("--no-random-sample", dest="random_sample", action="store_false", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--top-k", dest="top_k", type=int, default=None)
    p.add_argument("--speed-mps", dest="speed_mps", type=float, default=None)
    p.add_argument("--y-max-meters", dest="y_max_meters", type=float, default=None)
    p.add_argument("--future-only", dest="future_only", action="store_true", default=None)
    p.add_argument("--no-future-only", dest="future_only", action="store_false", default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--origin-lon", dest="origin_lon", type=float, default=None)
    p.add_argument("--origin-lat", dest="origin_lat", type=float, default=None)
    return p


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    config = DEFAULT_CONFIG
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged = asdict(config)
        for key, value in data.items():
            if key in merged:
                merged[key] = value
        config = ExperimentConfig(**merged)
    for field in fields(ExperimentConfig):
        name = field.name
        val = getattr(args, name, None)
        if val is not None:
            config = replace(config, **{name: val})
    return config


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _resolve_config(args)
    if config.top_k is None:
        raise ValueError("--top-k is required.")

    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = (NYC_TAXI_DIR / input_path).resolve()
    config = replace(config, input=str(input_path))

    random.seed(config.seed)
    np.random.seed(config.seed)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=== NYC Taxi POT Comparison: pot_partial ===")
    log(f"n={config.n}  seed={config.seed}  top_k={config.top_k}  speed_mps={config.speed_mps}")

    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
        logger=log,
    )
    log(f"Loaded {len(df)} trips.")

    # Load on CPU for the POT baseline.
    device = torch.device("cpu")
    xA_deg, xB_deg, tA, tB = prepare_tensors(df, mapping, device=device)
    xA_m, xB_m, lon0, lat0 = _project_lonlat_to_meters(
        xA_deg, xB_deg,
        origin_lon=config.origin_lon,
        origin_lat=config.origin_lat,
    )
    log(f"Projection origin: lon={lon0:.6f}  lat={lat0:.6f}")

    # Convert to numpy
    xA_np = xA_m.numpy().astype(np.float64)
    xB_np = xB_m.numpy().astype(np.float64)
    tA_np = tA.numpy()
    tB_np = tB.numpy()

    t_total_start = time.perf_counter()
    solve_info = _solve(
        xA_np, xB_np, tA_np, tB_np,
        speed_mps=config.speed_mps,
        y_max_meters=config.y_max_meters,
        future_only=bool(config.future_only),
        top_k=int(config.top_k),
        log=log,
    )
    total_runtime = time.perf_counter() - t_total_start

    output: dict[str, Any] = {
        "solver": "pot_partial",
        "params": {
            "n_requested": config.n,
            "n_used": len(df),
            "seed": config.seed,
            "top_k": int(config.top_k),
            "speed_mps": config.speed_mps,
            "y_max_meters": config.y_max_meters,
            "future_only": bool(config.future_only),
            "origin_lon": lon0,
            "origin_lat": lat0,
        },
        "performance": {
            "runtime_sec": total_runtime,
            "build_time_sec": solve_info["build_time_sec"],
            "solve_time_sec": solve_info["solve_time_sec"],
            "num_feasible_edges": solve_info["num_feasible_edges"],
        },
        "metrics": {
            "opt_cost_m": solve_info["opt_cost_m"],
            "opt_cost_km": solve_info["opt_cost_km"],
            "opt_avg_cost_m": solve_info["opt_avg_cost_m"],
            "opt_avg_cost_km": solve_info["opt_avg_cost_km"],
        },
    }

    log("\n=== Match Summary ===")
    log(f"top_k            : {solve_info['top_k']}")
    log(f"Opt cost (m)     : {solve_info['opt_cost_m']:.4f}")
    log(f"Opt cost (km)    : {solve_info['opt_cost_km']:.6f}")
    log(f"Build time       : {solve_info['build_time_sec']:.3f}s")
    log(f"Solve time       : {solve_info['solve_time_sec']:.3f}s")
    log(f"Total runtime    : {total_runtime:.3f}s")

    print(json.dumps(output, indent=2), flush=True)

    if config.out:
        out_path = Path(config.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        log(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
