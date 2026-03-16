#!/usr/bin/env python3
"""Run unscaled SPEF, scaled SPEF, then POT in one combined NYC taxi job."""
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
import ot

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
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
except Exception:
    pass

import spef_ot.kernels.euclidean_speed  # noqa: F401
from spef_ot import match, scaling_match
from loader import load_day  # noqa: E402
from prepare import prepare_tensors  # noqa: E402

EARTH_RADIUS_METERS = 6_371_000.0
POT_BATCH_ROWS = 512


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


@dataclass
class ExperimentConfig:
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20141014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 1000
    random_sample: bool = True
    seed: int = 1
    device: str | None = "cuda"
    k: int = 512
    delta: float = 0.0001
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
    p = argparse.ArgumentParser(description="Run unscaled SPEF, scaled SPEF, and POT in one job")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    p.add_argument("--no-random-sample", dest="random_sample", action="store_false", default=None)
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
        value = getattr(args, name, None)
        if value is not None:
            config = replace(config, **{name: value})
    return config


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _common_params(
    *,
    config: ExperimentConfig,
    n_used: int,
    origin_lon: float,
    origin_lat: float,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "n_requested": config.n,
        "n_used": n_used,
        "seed": config.seed,
        "speed_mps": config.speed_mps,
        "y_max_meters": config.y_max_meters,
        "future_only": bool(config.future_only),
        "delta": config.delta,
        "C": float(config.C) if config.C is not None else 100000.0,
        "k": config.k,
        "stopping_condition": config.stopping_condition,
        "fill_policy": config.fill_policy,
        "device": str(device),
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
    }


def _run_spef_variant(
    *,
    solver: str,
    config: ExperimentConfig,
    xA_m: torch.Tensor,
    xB_m: torch.Tensor,
    tA: torch.Tensor,
    tB: torch.Tensor,
    device: torch.device,
    common_params: dict[str, Any],
) -> dict[str, Any]:
    C = float(config.C) if config.C is not None else 100000.0
    kernel_kwargs: dict[str, Any] = {
        "times_A": tA,
        "times_B": tB,
        "speed_mps": config.speed_mps,
        "y_max_meters": config.y_max_meters,
        "future_only": bool(config.future_only),
    }

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if solver == "spef_unscaled":
        result = match(
            xA_m,
            xB_m,
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
    elif solver == "spef_scaled":
        result = scaling_match(
            xA_m,
            xB_m,
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
            result.metrics.get(
                "total_inner_loops",
                result.metrics.get("total_iterations", result.iterations),
            )
        )
    else:
        raise ValueError(f"Unexpected SPEF solver {solver!r}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0

    matching_cost_m = float(result.matching_cost)
    matching_cost_km = matching_cost_m / 1000.0
    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))

    return {
        "solver": solver,
        "params": dict(common_params),
        "performance": {
            "runtime_sec": runtime,
            "phases": phases,
            "total_inner_loops": total_inner_loops,
        },
        "metrics": {
            "matching_cost_m": matching_cost_m,
            "matching_cost_km": matching_cost_km,
            "avg_cost_m": (matching_cost_m / feasible_matches) if feasible_matches > 0 else None,
            "avg_cost_km": (matching_cost_km / feasible_matches) if feasible_matches > 0 else None,
            "feasible_matches": feasible_matches,
            "free_B": free_B,
            "removed_by_future": float(result.metrics.get("removed_by_future", 0.0)),
            "removed_by_speed": float(result.metrics.get("removed_by_speed", 0.0)),
            "removed_by_ymax": float(result.metrics.get("removed_by_ymax", 0.0)),
        },
    }


def _extract_scaled_top_k(scaled_result: dict[str, Any], n_used: int) -> int:
    feasible_matches = scaled_result["metrics"].get("feasible_matches")
    if feasible_matches is None:
        raise ValueError("Scaled SPEF result is missing feasible_matches.")
    top_k = int(round(float(feasible_matches)))
    if abs(float(feasible_matches) - top_k) > 1e-6:
        raise ValueError(f"Scaled feasible_matches must be integral, got {feasible_matches}.")
    if not (0 <= top_k <= n_used):
        raise ValueError(f"Scaled feasible_matches must satisfy 0 <= K <= n_used ({n_used}), got {top_k}.")
    return top_k


def _run_pot(
    *,
    xA_np: np.ndarray,
    xB_np: np.ndarray,
    tA_np: np.ndarray,
    tB_np: np.ndarray,
    config: ExperimentConfig,
    top_k: int,
    common_params: dict[str, Any],
    log,
) -> dict[str, Any]:
    n = len(xA_np)
    if config.y_max_meters is None:
        raise ValueError("--y-max-meters is required so the POT penalty is well-defined.")
    if not (0 <= top_k <= n):
        raise ValueError(f"top_k must satisfy 0 <= top_k <= n_used ({n}), got {top_k}.")

    penalty = float(config.y_max_meters) * 2.0
    t_build_start = time.perf_counter()
    M_raw = np.empty((n, n), dtype=np.float64)
    M = np.empty((n, n), dtype=np.float64)
    num_feasible = 0

    tA64 = tA_np.astype(np.float64, copy=False)
    tB64 = tB_np.astype(np.float64, copy=False)

    log(f"Building dense POT cost matrix in batches of {POT_BATCH_ROWS} rows: n={n}, top_k={top_k}")
    for row_start in range(0, n, POT_BATCH_ROWS):
        row_end = min(row_start + POT_BATCH_ROWS, n)
        block_diff = xB_np[row_start:row_end, None, :] - xA_np[None, :, :]
        block_sq = np.sum(block_diff * block_diff, axis=2)
        block_raw = np.sqrt(block_sq, out=np.empty_like(block_sq))
        block_dt = tB64[row_start:row_end, None] - tA64[None, :]

        block_feasible = np.ones((row_end - row_start, n), dtype=bool)
        if config.future_only:
            block_feasible &= block_dt >= 0
        block_feasible &= block_raw < float(config.y_max_meters)
        if config.speed_mps is not None:
            pos_dt = block_dt > 0
            block_feasible &= pos_dt | (block_raw < 1e-3)
            max_dist = np.where(pos_dt, float(config.speed_mps) * block_dt, 0.0)
            block_feasible &= block_raw <= max_dist + 1e-3

        block_cost = np.full((row_end - row_start, n), penalty, dtype=np.float64)
        block_cost[block_feasible] = np.minimum(block_raw[block_feasible], float(config.y_max_meters))

        M_raw[row_start:row_end, :] = block_raw
        M[row_start:row_end, :] = block_cost
        num_feasible += int(block_feasible.sum())

    build_time = time.perf_counter() - t_build_start
    log(f"POT cost matrix built: {num_feasible} feasible entries  [{build_time:.2f}s]")

    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(n, dtype=np.float64) / n
    m = float(top_k) / float(n) if n > 0 else 0.0

    log("Solving POT partial Wasserstein...")
    print(f"[POT] top_k={top_k}, m={m:.6f}, delta={config.delta}, nb_dummies=10, numItermax=10000000")
    t_solve_start = time.perf_counter()
    gamma = ot.partial.partial_wasserstein(a, b, M, m=m, nb_dummies=10, numItermax=10000000)
    solve_time = time.perf_counter() - t_solve_start
    total_runtime = build_time + solve_time
    log(f"POT solve finished in {solve_time:.3f}s")

    opt_cost_m = float(np.sum(gamma * M_raw) * n)
    opt_cost_km = opt_cost_m / 1000.0

    pot_params = dict(common_params)
    pot_params["top_k"] = top_k
    pot_params["top_k_source"] = "spef_scaled"

    return {
        "solver": "pot_partial",
        "params": pot_params,
        "performance": {
            "runtime_sec": total_runtime,
            "build_time_sec": build_time,
            "solve_time_sec": solve_time,
            "num_feasible_edges": num_feasible,
        },
        "metrics": {
            "opt_cost_m": opt_cost_m,
            "opt_cost_km": opt_cost_km,
            "opt_avg_cost_m": (opt_cost_m / top_k) if top_k > 0 else None,
            "opt_avg_cost_km": (opt_cost_km / top_k) if top_k > 0 else None,
        },
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _resolve_config(args)

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

    log("=== NYC Taxi POT Comparison: combined ===")
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
        xA_deg,
        xB_deg,
        origin_lon=config.origin_lon,
        origin_lat=config.origin_lat,
    )
    log(f"Projection origin: lon={lon0:.6f}  lat={lat0:.6f}")

    common_params = _common_params(
        config=config,
        n_used=len(df),
        origin_lon=lon0,
        origin_lat=lat0,
        device=device,
    )

    log("Running SPEF unscaled phase...")
    spef_unscaled = _run_spef_variant(
        solver="spef_unscaled",
        config=config,
        xA_m=xA_m,
        xB_m=xB_m,
        tA=tA,
        tB=tB,
        device=device,
        common_params=common_params,
    )

    log("Running SPEF scaled phase...")
    spef_scaled = _run_spef_variant(
        solver="spef_scaled",
        config=config,
        xA_m=xA_m,
        xB_m=xB_m,
        tA=tA,
        tB=tB,
        device=device,
        common_params=common_params,
    )

    top_k = _extract_scaled_top_k(spef_scaled, len(df))
    log(f"Running POT phase with top_k={top_k} from spef_scaled...")

    xA_np = xA_m.detach().cpu().numpy().astype(np.float64, copy=False)
    xB_np = xB_m.detach().cpu().numpy().astype(np.float64, copy=False)
    tA_np = tA.detach().cpu().numpy()
    tB_np = tB.detach().cpu().numpy()
    pot_result = _run_pot(
        xA_np=xA_np,
        xB_np=xB_np,
        tA_np=tA_np,
        tB_np=tB_np,
        config=config,
        top_k=top_k,
        common_params=common_params,
        log=log,
    )

    total_runtime_sec = (
        float(spef_unscaled["performance"]["runtime_sec"])
        + float(spef_scaled["performance"]["runtime_sec"])
        + float(pot_result["performance"]["runtime_sec"])
    )

    output: dict[str, Any] = {
        "solver": "combined",
        "params": dict(common_params),
        "performance": {
            "runtime_sec": total_runtime_sec,
        },
        "runs": {
            "spef_unscaled": spef_unscaled,
            "spef_scaled": spef_scaled,
            "pot_partial": pot_result,
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
