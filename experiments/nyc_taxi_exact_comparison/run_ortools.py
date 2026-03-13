#!/usr/bin/env python3
"""Exact min-cost partial matching via OR-Tools min-cost flow — NYC taxi exact comparison.

Problem formulation
-------------------
We have n dropoff points A[i] (available at time tA[i]) and n pickup requests B[j]
(desired at time tB[j]).  A match (A[i], B[j]) is feasible when:
  - future_only : tA[i] <= tB[j]
  - speed       : dist(A[i], B[j]) / (tB[j] - tA[i]) <= speed_mps  (dt > 0)
  - y_max       : dist(A[i], B[j]) < y_max_meters

Objective: maximise the number of matched pairs, breaking ties by minimising total distance.

Min-cost flow graph
-------------------
Nodes  (total = 2n + 3):
  S       = 0            supply = n
  B[j]    = 1 + j        transit (j = 0 .. n-1)
  A[i]    = 1 + n + i    transit (i = 0 .. n-1)
  DUMMY   = 1 + 2n       drain for unmatched B
  T       = 2 + 2n       demand = n

Edges:
  S -> B[j]        cap=1, cost=0              (every B gets one allocation unit)
  B[j] -> A[i]     cap=1, cost=dist*SCALE     (feasible matches only)
  B[j] -> DUMMY    cap=1, cost=BIG_PENALTY    (unmatched B, strongly penalised)
  A[i] -> T        cap=1, cost=0              (each A used at most once)
  DUMMY -> T        cap=n, cost=0

BIG_PENALTY >> max real cost  →  solver maximises cardinality first, then minimises cost.
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

from loader import load_day        # noqa: E402
from prepare import prepare_tensors  # noqa: E402

EARTH_RADIUS_METERS = 6_371_000.0

# Cost precision: 1 unit = 1 mm  →  max real scaled cost = 10 000 m * 1 000 = 10 000 000
COST_SCALE = 1_000
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
# Feasible edge computation (vectorised via numpy, batched over B)
# ---------------------------------------------------------------------------

def _build_feasible_edges(
    xA_np: np.ndarray,   # [n, 2] float32, metres
    xB_np: np.ndarray,   # [n, 2] float32, metres
    tA_np: np.ndarray,   # [n] int64, Unix seconds
    tB_np: np.ndarray,   # [n] int64, Unix seconds
    speed_mps: float | None,
    y_max_meters: float | None,
    future_only: bool,
    batch_size: int = 512,
) -> tuple[list[int], list[int], list[int], int]:
    """Return (src_B_nodes, dst_A_nodes, scaled_int_costs, num_feasible_edges).

    Node indices are in the flow graph:
      B[j]  = 1 + j
      A[i]  = 1 + n + i
    """
    n = len(xA_np)
    B_OFFSET = 1
    A_OFFSET = 1 + n

    src_list: list[int] = []
    dst_list: list[int] = []
    cost_list: list[int] = []

    for j_start in range(0, n, batch_size):
        j_end = min(j_start + batch_size, n)
        xB_batch = xB_np[j_start:j_end]          # [b, 2]
        tB_batch = tB_np[j_start:j_end]          # [b]
        b = j_end - j_start

        # Euclidean distances  [b, n]
        diff = xB_batch[:, None, :] - xA_np[None, :, :]   # [b, n, 2]
        dists = np.sqrt((diff * diff).sum(axis=2))         # [b, n]  float64

        # Time delta [b, n]  (seconds, may be negative)
        dt = tB_batch[:, None].astype(np.float64) - tA_np[None, :].astype(np.float64)

        mask = np.ones((b, n), dtype=bool)

        if future_only:
            mask &= dt >= 0

        if y_max_meters is not None:
            mask &= dists < float(y_max_meters)

        if speed_mps is not None:
            pos_dt = dt > 0
            # same-second departure only feasible if driver is already at the spot (dist ≈ 0)
            mask &= pos_dt | (dists < 1e-3)
            max_dist = np.where(pos_dt, float(speed_mps) * dt, 0.0)
            mask &= dists <= max_dist + 1e-3   # small tolerance for float

        batch_b_idx, a_idx = np.where(mask)       # local j within batch, global i
        abs_j = j_start + batch_b_idx

        if a_idx.size > 0:
            pair_dists = dists[batch_b_idx, a_idx]
            pair_costs = (pair_dists * COST_SCALE).astype(np.int64)
            src_list.extend((B_OFFSET + abs_j).tolist())
            dst_list.extend((A_OFFSET + a_idx).tolist())
            cost_list.extend(pair_costs.tolist())

    return src_list, dst_list, cost_list, len(src_list)


# ---------------------------------------------------------------------------
# OR-Tools min-cost flow solve
# ---------------------------------------------------------------------------

def _solve(
    xA_np: np.ndarray,
    xB_np: np.ndarray,
    tA_np: np.ndarray,
    tB_np: np.ndarray,
    speed_mps: float | None,
    y_max_meters: float | None,
    future_only: bool,
    log,
) -> dict[str, Any]:
    from ortools.graph.python import min_cost_flow  # type: ignore[import]

    n = len(xA_np)
    BIG_PENALTY = int(float(y_max_meters) * COST_SCALE) if y_max_meters is not None else int(1e9)
    S = 0
    B_OFFSET = 1
    A_OFFSET = 1 + n
    DUMMY = 1 + 2 * n
    T = 2 + 2 * n
    num_nodes = T + 1

    log(f"Building feasibility graph: n={n}, nodes={num_nodes}")
    t_build_start = time.perf_counter()

    src: list[int] = []
    dst: list[int] = []
    cap: list[int] = []
    cost: list[int] = []

    # S -> B[j]
    for j in range(n):
        src.append(S);  dst.append(B_OFFSET + j);  cap.append(1);  cost.append(0)

    # B[j] -> A[i]  (feasible pairs)
    fe_src, fe_dst, fe_cost, num_feasible = _build_feasible_edges(
        xA_np, xB_np, tA_np, tB_np, speed_mps, y_max_meters, future_only
    )
    src.extend(fe_src);  dst.extend(fe_dst)
    cap.extend([1] * num_feasible);  cost.extend(fe_cost)

    # B[j] -> DUMMY  (penalty for unmatched B)
    for j in range(n):
        src.append(B_OFFSET + j);  dst.append(DUMMY)
        cap.append(1);  cost.append(BIG_PENALTY)

    # A[i] -> T
    for i in range(n):
        src.append(A_OFFSET + i);  dst.append(T)
        cap.append(1);  cost.append(0)

    # DUMMY -> T
    src.append(DUMMY);  dst.append(T);  cap.append(n);  cost.append(0)

    build_time = time.perf_counter() - t_build_start
    log(f"Graph built: {len(src)} arcs ({num_feasible} feasible B->A)  [{build_time:.2f}s]")

    smcf = min_cost_flow.SimpleMinCostFlow()
    for s, e, c, u in zip(src, dst, cap, cost):
        smcf.add_arc_with_capacity_and_unit_cost(s, e, c, u)
    smcf.set_node_supply(S, n)
    smcf.set_node_supply(T, -n)

    log("Solving min-cost flow...")
    t_solve_start = time.perf_counter()
    status = smcf.solve()
    solve_time = time.perf_counter() - t_solve_start
    log(f"Solve finished in {solve_time:.3f}s  status={status}")

    if status != smcf.OPTIMAL:
        raise RuntimeError(
            f"OR-Tools min-cost flow did not reach OPTIMAL (status={status}). "
            "Check feasibility or increase BIG_PENALTY."
        )

    # Extract matches
    total_cost_scaled = 0
    feasible_matches = 0
    free_B = 0

    for arc in range(smcf.num_arcs()):
        if smcf.flow(arc) == 0:
            continue
        s_node = smcf.tail(arc)
        e_node = smcf.head(arc)
        if B_OFFSET <= s_node < B_OFFSET + n:
            if A_OFFSET <= e_node < A_OFFSET + n:
                # Real match B[j] -> A[i]
                feasible_matches += 1
                total_cost_scaled += smcf.unit_cost(arc)
            elif e_node == DUMMY:
                # Unmatched B
                free_B += 1

    total_cost_m = total_cost_scaled / COST_SCALE
    total_cost_km = total_cost_m / 1000.0

    return {
        "build_time_sec": build_time,
        "solve_time_sec": solve_time,
        "num_arcs": smcf.num_arcs(),
        "num_nodes": num_nodes,
        "num_feasible_edges": num_feasible,
        "feasible_matches": feasible_matches,
        "free_B": free_B,
        "matching_cost_m": total_cost_m,
        "matching_cost_km": total_cost_km,
        "avg_cost_m": total_cost_m / feasible_matches if feasible_matches > 0 else None,
        "avg_cost_km": (total_cost_km / feasible_matches) if feasible_matches > 0 else None,
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
    speed_mps: float | None = 8.0
    y_max_meters: float | None = 10000.0
    future_only: bool = True
    out: str | None = None
    origin_lon: float | None = None
    origin_lat: float | None = None


DEFAULT_CONFIG = ExperimentConfig()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NYC taxi exact solver via OR-Tools min-cost flow")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--random-sample", dest="random_sample", action="store_true", default=None)
    p.add_argument("--no-random-sample", dest="random_sample", action="store_false", default=None)
    p.add_argument("--seed", type=int, default=None)
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

    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = (NYC_TAXI_DIR / input_path).resolve()
    config = replace(config, input=str(input_path))

    random.seed(config.seed)
    np.random.seed(config.seed)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("=== NYC Taxi Exact Comparison: ortools ===")
    log(f"n={config.n}  seed={config.seed}  speed_mps={config.speed_mps}")

    df, mapping = load_day(
        config.input,
        date=config.date,
        n=config.n,
        random_sample=bool(config.random_sample),
        seed=config.seed,
        logger=log,
    )
    log(f"Loaded {len(df)} trips.")

    # Load on CPU (OR-Tools is CPU-only)
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
        log=log,
    )
    total_runtime = time.perf_counter() - t_total_start

    output: dict[str, Any] = {
        "solver": "ortools",
        "params": {
            "n_requested": config.n,
            "n_used": len(df),
            "seed": config.seed,
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
            "num_arcs": solve_info["num_arcs"],
            "num_nodes": solve_info["num_nodes"],
            "num_feasible_edges": solve_info["num_feasible_edges"],
        },
        "metrics": {
            "matching_cost_m": solve_info["matching_cost_m"],
            "matching_cost_km": solve_info["matching_cost_km"],
            "avg_cost_m": solve_info["avg_cost_m"],
            "avg_cost_km": solve_info["avg_cost_km"],
            "feasible_matches": solve_info["feasible_matches"],
            "free_B": solve_info["free_B"],
        },
    }

    log(f"\n=== Match Summary ===")
    log(f"Feasible matches : {solve_info['feasible_matches']}")
    log(f"Free B (unmatched): {solve_info['free_B']}")
    log(f"Total cost (m)   : {solve_info['matching_cost_m']:.4f}")
    log(f"Total cost (km)  : {solve_info['matching_cost_km']:.6f}")
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
