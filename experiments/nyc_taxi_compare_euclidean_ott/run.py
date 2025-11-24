#!/usr/bin/env python3
from __future__ import annotations

"""
Compare the space-efficient Euclidean-speed solver with OTT-JAX Sinkhorn on the
same NYC taxi sample using Euclidean distance and a y_max cutoff.

Outputs a pretty-printed summary and writes result/comparison.json.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Tuple

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

# Ensure kernel registration for the Euclidean-speed solver
import spef_ot.kernels.euclidean_speed  # noqa: F401
from spef_ot import MatchResult, match

from loader import load_day
from prepare import prepare_tensors


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
    # Data / sampling
    input: str = "./data/2014_Yellow_Taxi_Trip_Data_20141014-3.csv"
    date: str = "2014-10-14"
    n: int | None = 100000
    random_sample: bool = True
    seed: int = 1

    # Solver (space-efficient)
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

    # Projection
    origin_lon: float | None = None
    origin_lat: float | None = None

    # OTT-JAX parameters
    ott_epsilon: float = 1e-2
    ott_batch_size: int = 4096
    threshold: float = 1e-3

    # Output
    out: str = "result/comparison.json"
    no_warmup: bool = False


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
    p = argparse.ArgumentParser(
        description="Compare space-efficient Euclidean-speed solver vs OTT-JAX Sinkhorn (Euclidean).",
    )
    # Config I/O
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    p.add_argument("--input", type=str, default=None, help="Path to NYC taxi CSV/Parquet")
    p.add_argument("--date", type=str, default=None, help="Filter date (YYYY-MM-DD)")
    p.add_argument("--out", type=str, default=None, help="Output JSON (default: result/comparison.json)")

    # Sampling
    p.add_argument("--n", type=int, default=None, help="Number of trips to keep")
    p.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    p.add_argument("--no-random-sample", dest="random_sample", action="store_false", default=None)
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    # Solver controls
    p.add_argument("--device", type=str, default=None, help="Device for space-efficient solver (cuda/cpu)")
    p.add_argument("--k", type=int, default=None, help="Tile size (K)")
    p.add_argument("--delta", type=float, default=None, help="Scaling delta")
    p.add_argument("--stopping-condition", dest="stopping_condition", type=int, default=None)
    p.add_argument("--c-sample", dest="c_sample", type=int, default=None)
    p.add_argument("--C", dest="C", type=float, default=None, help="Scaling constant C (meters)")
    p.add_argument("--speed-mps", dest="speed_mps", type=float, default=None)
    p.add_argument("--y-max-meters", dest="y_max_meters", type=float, default=None)
    p.add_argument("--future-only", dest="future_only", action="store_true", default=None)
    p.add_argument("--no-future-only", dest="future_only", action="store_false", default=None)
    p.add_argument("--fill-policy", dest="fill_policy", choices=("greedy", "none"), default=None)

    # Projection
    p.add_argument("--origin-lon", dest="origin_lon", type=float, default=None)
    p.add_argument("--origin-lat", dest="origin_lat", type=float, default=None)

    # OTT-JAX
    p.add_argument("--ott-epsilon", dest="ott_epsilon", type=float, default=None)
    p.add_argument("--ott-batch-size", dest="ott_batch_size", type=int, default=None)
    p.add_argument("--threshold", dest="threshold", type=float, default=None)
    p.add_argument("--no-warmup", dest="no_warmup", action="store_true", default=None)
    return p


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
    out_path = Path(config.out)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    return replace(config, input=str(input_path), out=str(out_path))


def _seed_all(seed: int) -> None:
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
    xb2 = (xb.square()).sum(dim=1)
    prod = xb @ xa_T
    d2 = xb2.unsqueeze(1) + xa2.unsqueeze(0) - 2.0 * prod
    d2 = torch.clamp(d2, min=0.0)
    dist = torch.sqrt(d2)
    max_dist = float(dist.max().item())
    return 4.0 * max_dist


def _run_space_efficient(
    *,
    xA_m: torch.Tensor,
    xB_m: torch.Tensor,
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
) -> tuple[MatchResult, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = match(
        xA_m,
        xB_m,
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
        progress_callback=None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _run_ott_sinkhorn(
    *,
    xA_m: torch.Tensor,
    xB_m: torch.Tensor,
    y_max_meters: float | None,
    epsilon: float,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, Any], float]:
    # Lazy import to avoid hard dependency for other experiments.
    try:
        import jax
        import jax.numpy as jnp
        from ott.geometry import pointcloud, costs
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except Exception as e:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "OTT-JAX (and JAX) are required for this comparison. "
            "Please install `jax` with CUDA and `ott-jax`."
        ) from e

    # Inputs to JAX on default device (GPU if available)
    xA = jnp.asarray(xA_m.detach().cpu().numpy())
    xB = jnp.asarray(xB_m.detach().cpu().numpy())

    a = jnp.ones(xA.shape[0], dtype=jnp.float32) / xA.shape[0]
    b = jnp.ones(xB.shape[0], dtype=jnp.float32) / xB.shape[0]

    y_max = float(y_max_meters) if y_max_meters is not None and y_max_meters > 0 else None

    # Custom Euclidean cost with y_max cutoff
    class EuclideanCutoff(costs.CostFn):  # type: ignore[misc]
        def __init__(self, y_max: float | None):
            super().__init__()
            self._y_max = y_max

        def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            # x: [Nx,D] or [D]; y: [My,D] or [D]
            x_ = jnp.atleast_2d(x)
            y_ = jnp.atleast_2d(y)
            d = jnp.linalg.norm(x_[:, None, :] - y_[None, :, :], axis=-1)
            if self._y_max is None:
                return d
            return jnp.where(d >= self._y_max, jnp.inf, d)

    geom = pointcloud.PointCloud(
        x=xA,
        y=xB,
        cost_fn=EuclideanCutoff(y_max),
        epsilon=epsilon,
        batch_size=int(batch_size),
    )
    prob = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn()

    # Run on GPU
    t0 = time.perf_counter()
    out = solver(prob)
    # Note: out.matrix may be large for big problems; this comparison assumes feasible sizes.
    P = out.matrix  # transport plan
    print("Computing discrete matching...")
    runtime = time.perf_counter() - t0

    # Discrete matching via thresholding the coupling
    matched_pairs = jnp.argwhere(P > threshold)
    num_pairs = int(matched_pairs.shape[0])
    if num_pairs == 0:
        return {"total_cost": 0.0, "avg_cost": None, "pairs": 0, "num_rows": int(xB.shape[0]), "num_cols": int(xA.shape[0])}, runtime

    rows = matched_pairs[:, 0]
    cols = matched_pairs[:, 1]
    pa = xA[cols]
    pb = xB[rows]
    d = jnp.linalg.norm(pa - pb, axis=1)

    # Respect feasibility cutoff in metrics too (defensive):
    if y_max is not None:
        mask = d < y_max
        d = d[mask]
        num_pairs = int(d.shape[0])

    total_cost = float(jnp.sum(d))
    avg_cost = (total_cost / num_pairs) if num_pairs > 0 else None

    return (
        {
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "pairs": num_pairs,
            "num_rows": int(xB.shape[0]),
            "num_cols": int(xA.shape[0]),
        },
        runtime,
    )


def _print_summary(
    *,
    solver_total: float,
    solver_avg: float | None,
    solver_runtime: float,
    ott_total: float,
    ott_avg: float | None,
    ott_runtime: float,
) -> None:
    def fmt(x: float | None, prec: int = 3) -> str:
        if x is None:
            return "n/a"
        return f"{x:.{prec}e}" if abs(x) >= 1e4 else f"{x:.{prec}f}"

    print("\n===================== COMPARISON SUMMARY =====================")
    print(f"{'Metric':<18} {'Euclidean Solver':>22} {'OTT-JAX Sinkhorn':>22}")
    print("-------------------------------------------------------------")
    print(f"{'Total Cost':<18} {fmt(solver_total):>22} {fmt(ott_total):>22}")
    print(f"{'Average Cost':<18} {fmt(solver_avg):>22} {fmt(ott_avg):>22}")
    print(f"{'Runtime (s)':<18} {solver_runtime:>22.2f} {ott_runtime:>22.2f}")
    print("==============================================================")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])
    config = _resolve_paths(_resolve_config(args))

    # Seeds and device
    _seed_all(int(config.seed))
    device = (
        torch.device(config.device)
        if config.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=== Compare Euclidean-Speed Solver vs OTT-JAX Sinkhorn ===")
    print(f"Input file      : {config.input}")
    print(f"Date            : {config.date}")
    print(f"Requested trips : {config.n} ({'random' if config.random_sample else 'first'})")
    print(f"Y max (meters)  : {config.y_max_meters}")
    print(f"Device (solver) : {device}")
    print(f"OTT epsilon     : {config.ott_epsilon}")
    print(f"OTT batch_size  : {config.ott_batch_size}")
    print(f"Threshold (P>t) : {config.threshold}")

    # Load and prepare data
    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
        logger=print,
    )
    xA_deg, xB_deg, tA, tB = prepare_tensors(df, mapping, device=device)

    # Project to meters
    xA_m, xB_m, lon0, lat0 = _project_lonlat_to_meters(
        xA_deg, xB_deg, origin_lon=config.origin_lon, origin_lat=config.origin_lat
    )
    print(
        f"Projection origin (lon,lat): ({lon0:.6f}, {lat0:.6f}); "
        f"xA{tuple(xA_m.shape)} xB{tuple(xB_m.shape)}"
    )

    # C handling: prefer provided; else estimate in planar metric
    if config.C is not None:
        if config.C <= 0:
            raise ValueError("C must be positive when provided")
        C = float(config.C)
        print(f"Using provided C value: C={C:.4f}")
    else:
        C = _estimate_c_euclidean(xA_m, xB_m, sample_size=config.c_sample, seed=config.seed)
        print(f"Estimated C value (Euclidean): C={C:.4f} (sample_size={config.c_sample})")

    # --- Warm-up phase (optional) ---
    # A warm-up run handles first-run costs like JIT compilation (JAX) or CUDA
    # kernel loading (PyTorch), ensuring the timed run is more representative
    # of steady-state performance. This is crucial for a fair comparison.
    if not config.no_warmup:
        print("\nPerforming warm-up runs (use --no-warmup to skip)...")
        _, warmup_solver_time = _run_space_efficient(
            xA_m=xA_m, xB_m=xB_m, C=C, k=config.k, delta=config.delta, device=device,
            seed=config.seed, times_A=tA, times_B=tB, stopping_condition=config.stopping_condition,
            speed_mps=config.speed_mps, y_max_meters=config.y_max_meters,
            future_only=bool(config.future_only), fill_policy=config.fill_policy,
        )
        _, warmup_ott_time = _run_ott_sinkhorn(
            xA_m=xA_m, xB_m=xB_m, y_max_meters=config.y_max_meters,
            epsilon=float(config.ott_epsilon), batch_size=int(config.ott_batch_size),
            threshold=float(config.threshold),
        )
        print(f"Warm-up finished: Solver ({warmup_solver_time:.2f}s), OTT-JAX ({warmup_ott_time:.2f}s)")
    else:
        print("\nSkipping warm-up runs.")

    # --- Timed execution phase ---

    # 1) Run space-efficient solver
    # This solver computes a discrete, optimal matching according to its algorithm.
    print("\nRunning Euclidean solver...")
    solver_result, solver_runtime = _run_space_efficient(
        xA_m=xA_m,
        xB_m=xB_m,
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
    )

    feasible_matches = float(solver_result.metrics.get("feasible_matches", 0.0))
    solver_total_cost = float(solver_result.matching_cost)
    solver_avg_cost = (solver_total_cost / feasible_matches) if feasible_matches > 0 else None

    # 2) Run OTT-JAX Sinkhorn
    # NOTE: Sinkhorn computes a probabilistic coupling matrix, not a discrete assignment.
    # To compare, we derive a discrete matching by thresholding the coupling (P > threshold).
    # This is a common but important methodological difference to be aware of.
    print("\nRunning OTT-JAX Sinkhorn...")
    ott_metrics, ott_runtime = _run_ott_sinkhorn(
        xA_m=xA_m,
        xB_m=xB_m,
        y_max_meters=config.y_max_meters,
        epsilon=float(config.ott_epsilon),
        batch_size=int(config.ott_batch_size),
        threshold=float(config.threshold),
    )

    # 3) Pretty summary
    _print_summary(
        solver_total=solver_total_cost,
        solver_avg=solver_avg_cost,
        solver_runtime=solver_runtime,
        ott_total=float(ott_metrics["total_cost"]),
        ott_avg=ott_metrics["avg_cost"],
        ott_runtime=ott_runtime,
    )

    # 4) Save JSON
    out_path = Path(config.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "solver": {
            "total_cost": solver_total_cost,
            "avg_cost": solver_avg_cost,
            "runtime": solver_runtime,
            "feasible_matches": feasible_matches,
        },
        "ott_jax": {
            "total_cost": float(ott_metrics["total_cost"]),
            "avg_cost": ott_metrics["avg_cost"],
            "runtime": ott_runtime,
            "pairs": int(ott_metrics["pairs"]),
            "num_rows": int(ott_metrics["num_rows"]),
            "num_cols": int(ott_metrics["num_cols"]),
        },
        "threshold": float(config.threshold),
        "epsilon": float(config.ott_epsilon),
        "batch_size": int(config.ott_batch_size),
        "y_max": float(config.y_max_meters) if config.y_max_meters is not None else None,
        "projection_origin": {"lon": lon0, "lat": lat0},
        "dataset": {
            "input": str(Path(config.input).resolve()),
            "date": config.date,
            "n_used": len(df),
            "random_sample": bool(config.random_sample),
            "seed": int(config.seed),
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaving results to {out_path}")


if __name__ == "__main__":
    main()
