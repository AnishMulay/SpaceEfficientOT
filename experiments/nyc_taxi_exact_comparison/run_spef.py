#!/usr/bin/env python3
"""Runner for SPEF solvers (unscaled and scaled) — NYC taxi exact comparison experiment.

Solver variants:
  spef_unscaled  — calls spef_ot.match()          (single fixed delta, no cost scaling)
  spef_scaled    — calls spef_ot.scaling_match()  (epsilon/cost scaling with phase warmup)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
NYC_TAXI_DIR = EXPERIMENT_DIR.parent / "nyc_taxi"
if str(NYC_TAXI_DIR) not in sys.path:
    sys.path.insert(0, str(NYC_TAXI_DIR))

torch.set_float32_matmul_precision("highest")
try:
    torch.backends.cuda.matmul.allow_tf32 = False   # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = False          # type: ignore[attr-defined]
except Exception:
    pass

import spef_ot.kernels.euclidean_speed  # noqa: F401 — ensure kernel registration
from spef_ot import match, scaling_match
from loader import load_day       # noqa: E402
from prepare import prepare_tensors  # noqa: E402

EARTH_RADIUS_METERS = 6_371_000.0
VALID_SOLVERS = ("spef_unscaled", "spef_scaled")


# ---------------------------------------------------------------------------
# Coordinate projection (identical to existing run.py)
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
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    solver: str = "spef_unscaled"
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20141014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 1000
    random_sample: bool = True
    seed: int = 1
    device: str | None = "cuda"
    k: int = 512
    delta: float = 0.001
    stopping_condition: int | None = 50
    C: float | None = 100000.0
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 10000.0
    future_only: bool = True
    fill_policy: str = "none"
    out: str | None = None
    origin_lon: float | None = None
    origin_lat: float | None = None


DEFAULT_CONFIG = ExperimentConfig()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NYC taxi SPEF solver (unscaled or scaled)")
    p.add_argument("--solver", choices=VALID_SOLVERS, default=None)
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--delta", type=float, default=None)
    p.add_argument("--stopping-condition", dest="stopping_condition", type=int, default=None)
    p.add_argument("--C", dest="C", type=float, default=None)
    p.add_argument("--speed-mps", dest="speed_mps", type=float, default=None)
    p.add_argument("--y-max-meters", dest="y_max_meters", type=float, default=None)
    p.add_argument("--future-only", dest="future_only", action="store_true", default=None)
    p.add_argument("--no-future-only", dest="future_only", action="store_false", default=None)
    p.add_argument("--fill-policy", dest="fill_policy", choices=("greedy", "none"), default=None)
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


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _resolve_config(args)

    if config.solver not in VALID_SOLVERS:
        raise ValueError(f"--solver must be one of {VALID_SOLVERS}, got {config.solver!r}")

    # Resolve input path relative to nyc_taxi data dir if not absolute
    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = (NYC_TAXI_DIR / input_path).resolve()
    config = replace(config, input=str(input_path))

    _seed_all(int(config.seed))

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"=== NYC Taxi Exact Comparison: {config.solver} ===")
    log(f"n={config.n}  seed={config.seed}  speed_mps={config.speed_mps}  device={device}")

    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
        logger=log,
    )
    log(f"Loaded {len(df)} trips.")

    xA_deg, xB_deg, tA, tB = prepare_tensors(df, mapping, device=device)
    xA_m, xB_m, lon0, lat0 = _project_lonlat_to_meters(
        xA_deg, xB_deg,
        origin_lon=config.origin_lon,
        origin_lat=config.origin_lat,
    )
    log(f"Projection origin: lon={lon0:.6f}  lat={lat0:.6f}")

    C = float(config.C) if config.C is not None else 100000.0

    kernel_kwargs: dict[str, Any] = dict(
        times_A=tA,
        times_B=tB,
        speed_mps=config.speed_mps,
        y_max_meters=config.y_max_meters,
        future_only=bool(config.future_only),
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if config.solver == "spef_unscaled":
        result = match(
            xA_m, xB_m,
            kernel="euclidean_speed",
            C=C,
            k=config.k,
            delta=config.delta,
            device=device,
            seed=config.seed,
            stopping_condition=config.stopping_condition,
            fill_policy=config.fill_policy,
            **kernel_kwargs,
        )
        phases = 1
        total_inner_loops = int(result.iterations)

    else:  # spef_scaled
        result = scaling_match(
            xA_m, xB_m,
            kernel="euclidean_speed",
            C=C,
            k=config.k,
            target_delta=config.delta,
            device=device,
            seed=config.seed,
            stopping_condition=config.stopping_condition,
            fill_policy=config.fill_policy,
            verbose=False,
            **kernel_kwargs,
        )
        phases = int(result.phases)
        total_inner_loops = int(
            result.metrics.get("total_inner_loops",
            result.metrics.get("total_iterations", result.iterations))
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0

    total_cost_m = float(result.matching_cost)
    total_cost_km = total_cost_m / 1000.0
    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))

    output: dict[str, Any] = {
        "solver": config.solver,
        "params": {
            "n_requested": config.n,
            "n_used": len(df),
            "seed": config.seed,
            "speed_mps": config.speed_mps,
            "y_max_meters": config.y_max_meters,
            "future_only": bool(config.future_only),
            "delta": config.delta,
            "C": C,
            "k": config.k,
            "stopping_condition": config.stopping_condition,
            "fill_policy": config.fill_policy,
            "device": str(device),
            "origin_lon": lon0,
            "origin_lat": lat0,
        },
        "performance": {
            "runtime_sec": runtime,
            "phases": phases,
            "total_inner_loops": total_inner_loops,
        },
        "metrics": {
            "matching_cost_m": total_cost_m,
            "matching_cost_km": total_cost_km,
            "avg_cost_m": total_cost_m / feasible_matches if feasible_matches > 0 else None,
            "avg_cost_km": (total_cost_km / feasible_matches) if feasible_matches > 0 else None,
            "feasible_matches": feasible_matches,
            "free_B": free_B,
            "removed_by_future": float(result.metrics.get("removed_by_future", 0.0)),
            "removed_by_speed": float(result.metrics.get("removed_by_speed", 0.0)),
            "removed_by_ymax": float(result.metrics.get("removed_by_ymax", 0.0)),
        },
    }

    print(json.dumps(output, indent=2), flush=True)

    if config.out:
        out_path = Path(config.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        log(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
