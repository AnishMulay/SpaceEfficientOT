#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Tuple

# Determinism and precision controls BEFORE importing torch
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import torch

# Ensure local package imports work without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

NYC_TAXI_DIR = EXPERIMENT_DIR.parent / "nyc_taxi"
if str(NYC_TAXI_DIR) not in sys.path:
    sys.path.insert(0, str(NYC_TAXI_DIR))

# Highest precision matmul; disable TF32 paths
torch.set_float32_matmul_precision("highest")
try:
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
except Exception:
    pass

import spef_ot.kernels.euclidean_speed  # noqa: F401 - ensure kernel registration
from spef_ot import MatchResult, match  # noqa: E402

from loader import load_day  # noqa: E402
from prepare import prepare_tensors  # noqa: E402


EARTH_RADIUS_METERS = 6_371_000.0


def _project_lonlat_to_meters(
    xA_deg: torch.Tensor,
    xB_deg: torch.Tensor,
    *,
    origin_lon: float | None = None,
    origin_lat: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    # Choose origin as medians if not provided
    if origin_lon is None or origin_lat is None:
        lonA, latA = xA_deg[:, 0], xA_deg[:, 1]
        lonB, latB = xB_deg[:, 0], xB_deg[:, 1]
        lon0 = torch.median(torch.cat((lonA, lonB))).item()
        lat0 = torch.median(torch.cat((latA, latB))).item()
    else:
        lon0, lat0 = float(origin_lon), float(origin_lat)

    # Convert degrees to meters in local tangent plane
    deg2rad = torch.tensor(torch.pi / 180.0, device=xA_deg.device, dtype=torch.float32)
    lat0_rad = torch.tensor(lat0 * (torch.pi / 180.0), device=xA_deg.device, dtype=torch.float32)
    scale_x = float(EARTH_RADIUS_METERS) * torch.cos(lat0_rad)
    scale_y = float(EARTH_RADIUS_METERS)

    def _proj(coords: torch.Tensor) -> torch.Tensor:
        lon = coords[:, 0] - float(lon0)
        lat = coords[:, 1] - float(lat0)
        x = lon.to(dtype=torch.float32) * deg2rad * scale_x
        y = lat.to(dtype=torch.float32) * deg2rad * scale_y
        return torch.stack((x, y), dim=1)

    return _proj(xA_deg), _proj(xB_deg), lon0, lat0


@dataclass
class ExperimentConfig:
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20251014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 100000
    random_sample: bool = True
    seed: int = 1
    device: str | None = "cuda"
    k: int = 512
    delta: float = 0.001
    stopping_condition: int | None = 1000
    c_sample: int = 1
    C: float | None = 100000.0
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 100000.0
    future_only: bool = True
    fill_policy: str = "none"
    preview_count: int = 5
    verbose: bool = True
    out: str | None = None
    no_warmup: bool = False
    origin_lon: float | None = None
    origin_lat: float | None = None


DEFAULT_CONFIG = ExperimentConfig()


def _load_config_from_json(path: Path, base: ExperimentConfig) -> ExperimentConfig:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must contain an object at the top level")
    merged = asdict(base)
    for key, value in data.items():
        if key in merged:
            merged[key] = value
    return ExperimentConfig(**merged)


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates: dict[str, Any] = {}
    for field in fields(ExperimentConfig):
        name = field.name
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        updates[name] = value
    if not updates:
        return config
    return replace(config, **updates)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NYC taxi experiment using the Euclidean speed slack kernel",
        epilog=(
            "Example: python experiments/nyc_taxi_speed_euclidean/run.py "
            "--input ./data/nyc.csv --date 2014-10-14 --device cuda "
            "--k 512 --delta 0.001 --speed-mps 8.0 --stopping-condition 1000 --fill-policy none"
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--input", type=str, default=None, help="Path to NYC taxi CSV/Parquet")
    parser.add_argument("--date", type=str, default=None, help="Filter date (YYYY-MM-DD)")
    parser.add_argument("--n", type=int, default=None, help="Number of trips to keep")
    parser.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    parser.add_argument("--no-random-sample", dest="random_sample", action="store_false", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--k", type=int, default=None, help="Tile size (K)")
    parser.add_argument("--delta", type=float, default=None, help="Scaling delta")
    parser.add_argument("--stopping-condition", dest="stopping_condition", type=int, default=None)
    parser.add_argument("--c-sample", dest="c_sample", type=int, default=None)
    parser.add_argument("--C", dest="C", type=float, default=None, help="Scaling constant C (meters)")
    parser.add_argument("--speed-mps", dest="speed_mps", type=float, default=None)
    parser.add_argument("--y-max-meters", dest="y_max_meters", type=float, default=None)
    parser.add_argument("--future-only", dest="future_only", action="store_true", default=None)
    parser.add_argument("--no-future-only", dest="future_only", action="store_false", default=None)
    parser.add_argument("--fill-policy", dest="fill_policy", choices=("greedy", "none"), default=None)
    parser.add_argument("--preview-count", dest="preview_count", type=int, default=None)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--no-warmup", dest="no_warmup", action="store_true", default=None)
    parser.add_argument("--origin-lon", dest="origin_lon", type=float, default=None)
    parser.add_argument("--origin-lat", dest="origin_lat", type=float, default=None)
    return parser


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    config = DEFAULT_CONFIG
    if args.config:
        config_path = Path(args.config)
        config = _load_config_from_json(config_path, config)
    config = _apply_overrides(config, args)
    return config


def _resolve_paths(config: ExperimentConfig) -> ExperimentConfig:
    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = (NYC_TAXI_DIR / input_path).resolve()

    out_path = config.out
    if out_path is not None:
        out_obj = Path(out_path)
        if not out_obj.is_absolute():
            out_obj = (EXPERIMENT_DIR / out_obj).resolve()
        out_path = str(out_obj)

    return replace(config, input=str(input_path), out=out_path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _estimate_c_euclidean(
    xA_m: torch.Tensor,
    xB_m: torch.Tensor,
    *,
    sample_size: int = 64,
    seed: int = 1,
) -> float:
    device = xA_m.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    m = xB_m.shape[0]
    sample_size = min(max(1, sample_size), m)
    idx = torch.randperm(m, device=device, generator=generator)[:sample_size]
    xb = xB_m.index_select(0, idx)  # [S,2]
    xa_T = xA_m.transpose(0, 1)
    xa2 = (xA_m.square()).sum(dim=1)
    # Distances for the sample vs all A
    xb2 = (xb.square()).sum(dim=1)
    prod = xb @ xa_T
    d2 = xb2.unsqueeze(1) + xa2.unsqueeze(0) - 2.0 * prod
    d2 = torch.clamp(d2, min=0.0)
    dist = torch.sqrt(d2)
    max_dist = float(dist.max().item())
    return 4.0 * max_dist


def _run_solver(
    *,
    xA: torch.Tensor,
    xB: torch.Tensor,
    C: float,
    k: int,
    delta: float,
    device: torch.device,
    seed: int,
    times_A: torch.Tensor,
    times_B: torch.Tensor,
    stopping_condition: int | None,
    speed_mps: float | None,
    y_max_meters: float | None,
    future_only: bool,
    fill_policy: str,
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
) -> tuple[MatchResult, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = match(
        xA,
        xB,
        kernel="euclidean_speed",
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
        times_A=times_A,
        times_B=times_B,
        stopping_condition=stopping_condition,
        speed_mps=speed_mps,
        y_max_meters=y_max_meters,
        future_only=future_only,
        fill_policy=fill_policy,
        progress_callback=progress_callback,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])
    config = _resolve_paths(_resolve_config(args))

    # Seeds for determinism
    _seed_all(int(config.seed))

    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def log(msg: str) -> None:
        print(msg)

    log("=== NYC Taxi Euclidean Speed Experiment ===")
    log(f"Input file      : {config.input}")
    log(f"Date            : {config.date}")
    log(f"Requested trips : {config.n} ({'random' if config.random_sample else 'first'})")
    log(f"Speed constraint: {config.speed_mps} m/s")
    log(f"Y max clamp     : {config.y_max_meters} m")
    log(f"Future-only     : {config.future_only}")
    log(f"Fill policy     : {config.fill_policy}")
    log(f"Device          : {device}")

    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
        logger=log,
    )
    log(f"Loader returned {len(df)} trips after filtering.")

    xA_deg, xB_deg, tA, tB = prepare_tensors(df, mapping, device=device)

    # Project lon/lat to planar meters (local tangent plane)
    xA_m, xB_m, lon0, lat0 = _project_lonlat_to_meters(
        xA_deg, xB_deg, origin_lon=config.origin_lon, origin_lat=config.origin_lat
    )
    log(
        f"Projection origin (lon,lat): ({lon0:.6f}, {lat0:.6f}); "
        f"xA{tuple(xA_m.shape)} xB{tuple(xB_m.shape)}"
    )

    # C handling: prefer provided; else estimate in planar metric
    if config.C is not None:
        if config.C <= 0:
            raise ValueError("C must be positive when provided")
        C = float(config.C)
        log(f"Using provided C value: C={C:.4f}")
    else:
        C = _estimate_c_euclidean(xA_m, xB_m, sample_size=config.c_sample, seed=config.seed)
        log(f"Estimated C value (Euclidean): C={C:.4f} (sample_size={config.c_sample})")

    log(
        "Solver parameters: "
        f"k={config.k}, delta={config.delta}, stopping_condition={config.stopping_condition}"
    )

    warmup_time = 0.0
    if not config.no_warmup:
        _, warmup_time = _run_solver(
            xA=xA_m,
            xB=xB_m,
            C=C,
            k=config.k,
            delta=config.delta,
            device=device,
            seed=config.seed,
            times_A=tA,
            times_B=tB,
            stopping_condition=config.stopping_condition,
            speed_mps=config.speed_mps,
            y_max_meters=config.y_max_meters,
            future_only=bool(config.future_only),
            fill_policy=config.fill_policy,
            progress_callback=None,
        )
        log(f"Warmup run completed in {warmup_time:.4f} s")
    else:
        log("Skipping warm-up run (--no-warmup)")

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        if event == "iteration":
            print(
                f"[Iter {payload['iteration']}] "
                f"free_B={payload['free_b']} matched_B={payload['matched_b']} "
                f"f={payload['objective_gap']:.3f} threshold={payload['threshold']:.3f}"
            )

    result, runtime = _run_solver(
        xA=xA_m,
        xB=xB_m,
        C=C,
        k=config.k,
        delta=config.delta,
        device=device,
        seed=config.seed,
        times_A=tA,
        times_B=tB,
        stopping_condition=config.stopping_condition,
        speed_mps=config.speed_mps,
        y_max_meters=config.y_max_meters,
        future_only=bool(config.future_only),
        fill_policy=config.fill_policy,
        progress_callback=progress_callback if config.verbose else None,
    )
    print(f"Measured run completed in {runtime:.4f} s over {int(result.iterations)} iterations")

    total_cost_m = float(result.matching_cost)
    total_cost_km = total_cost_m / 1000.0
    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))
    removed_by_future = float(result.metrics.get("removed_by_future", 0.0))
    removed_by_speed = float(result.metrics.get("removed_by_speed", 0.0))
    removed_by_ymax = float(result.metrics.get("removed_by_ymax", 0.0))

    avg_cost_m = total_cost_m / feasible_matches if feasible_matches > 0 else None
    avg_cost_km = avg_cost_m / 1000.0 if avg_cost_m is not None else None
    print("\n=== Match Summary ===")
    print(f"Feasible matches : {feasible_matches}")
    print(f"Free B nodes     : {free_B}")
    print(f"Removed by future: {removed_by_future}")
    print(f"Removed by speed : {removed_by_speed}")
    print(f"Removed by y_max : {removed_by_ymax}")
    print(f"Total cost (m)   : {total_cost_m:.4f}")
    print(f"Total cost (km)  : {total_cost_km:.6f}")
    if avg_cost_m is not None:
        print(f"Average cost (m) : {avg_cost_m:.4f}")
        print(f"Average cost (km): {avg_cost_km:.6f}")
    else:
        print("Average cost     : undefined (no feasible matches)")

    output = {
        "params": {
            "input": str(Path(config.input).resolve()),
            "date": config.date,
            "n_requested": config.n,
            "n_used": len(df),
            "random_sample": bool(config.random_sample),
            "seed": config.seed,
            "device": str(device),
            "k": config.k,
            "delta": config.delta,
            "stopping_condition": config.stopping_condition,
            "C_estimate": C,
            "c_sample": config.c_sample,
            "speed_mps": config.speed_mps,
            "y_max_meters": config.y_max_meters,
            "future_only": bool(config.future_only),
            "fill_policy": config.fill_policy,
            "origin_lon": lon0,
            "origin_lat": lat0,
        },
        "performance": {
            "warmup_runtime_sec": warmup_time,
            "runtime_sec": runtime,
            "iterations": int(result.iterations),
            "timing_metrics": result.metrics,
        },
        "metrics": {
            "matching_cost_m": total_cost_m,
            "matching_cost_km": total_cost_km,
            "avg_cost_m": avg_cost_m,
            "avg_cost_km": avg_cost_km,
            "feasible_matches": feasible_matches,
            "free_B": free_B,
            "removed_by_future": removed_by_future,
            "removed_by_speed": removed_by_speed,
            "removed_by_ymax": removed_by_ymax,
        },
    }

    print(json.dumps(output, indent=2))

    if config.out:
        out_path = Path(config.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote results to {out_path}")
    else:
        print("\nNo output path provided; results printed to stdout.")


if __name__ == "__main__":
    main()

