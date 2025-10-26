#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Allow running without an install when invoked from repo root
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from spef_ot import match  # noqa: E402


def generate_points(n: int, d: int, device: torch.device, seed: int, cache: bool = True):
    datasets_dir = Path(__file__).resolve().parent / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    cache_path = datasets_dir / f"points_n{n}_d{d}_seed{seed}.pt"

    if cache and cache_path.exists():
        data = torch.load(cache_path, map_location=device)
        return data["xa"].to(device), data["xb"].to(device)

    torch.manual_seed(seed)
    xa = torch.rand(n, d, device=device)
    xb = torch.rand(n, d, device=device)

    if cache:
        torch.save({"xa": xa.cpu(), "xb": xb.cpu()}, cache_path)

    return xa, xb


def compute_C(xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    """Compute scalar C consistent with archived experiments.

    Uses max Euclidean distance from a random B point to A, then squares.
    """
    n = xa.shape[0]
    rand_idx = torch.randint(0, n, (1,), device=xa.device)
    b_pt = xb[rand_idx]
    dists = torch.cdist(b_pt, xa)  # [1, n]
    return dists.max() ** 2


def run_once(n: int, d: int, k: int, delta: float, seed: int, device: torch.device):
    xa, xb = generate_points(n, d, device, seed, cache=True)
    C = compute_C(xa, xb)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = match(
        xa,
        xb,
        kernel="euclidean_sq",
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    runtime = t1 - t0
    matching_cost = float(result.matching_cost)
    cost_per_n = matching_cost / n

    return {
        "algorithm": "spef_ot.match(euclidean_sq)",
        "n": n,
        "d": d,
        "k": k,
        "delta": delta,
        "seed": seed,
        "device": str(device),
        "runtime": runtime,
        "matching_cost": matching_cost,
        "cost_per_n": cost_per_n,
        "iterations": int(result.iterations),
        "C": float(C),
        "metrics": result.metrics,
    }


def main():
    p = argparse.ArgumentParser(description="Run random Euclidean experiment using spef_ot")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = p.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Random Euclidean Experiment")
    print(f"Parameters: n={args.n}, d={args.d}, k={args.k}, delta={args.delta}, seed={args.seed}")
    print(f"Device: {device}")
    print("=" * 60)

    res = run_once(args.n, args.d, args.k, args.delta, args.seed, device)

    print("RESULTS:")
    print(json.dumps(res, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

