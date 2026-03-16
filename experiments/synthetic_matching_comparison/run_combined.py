#!/usr/bin/env python3
"""Synthetic matching comparison: spef_scaled vs POT exact EMD using plain L2 distances."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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

torch.set_float32_matmul_precision("highest")

import spef_ot.kernels.euclidean_l2  # noqa: F401 - registers euclidean_l2
from spef_ot import scaling_match


def _estimate_C(xA: np.ndarray, xB: np.ndarray, seed: int) -> float:
    rng = np.random.RandomState(seed)
    n = len(xA)
    idx = rng.choice(n, size=min(64, n), replace=False)
    diff = xA[idx].astype(np.float64) - xB[idx].astype(np.float64)
    dists = np.sqrt((diff ** 2).sum(axis=1))
    C = float(4.0 * dists.max())
    print(f"[C estimate] max_sample_l2={dists.max():.6f}  C={C:.6f}", flush=True)
    return C


def _run_spef(xA: np.ndarray, xB: np.ndarray, *, n: int, C: float,
              k: int, delta: float, device: str, seed: int) -> dict:
    xA_t = torch.from_numpy(xA).to(device)
    xB_t = torch.from_numpy(xB).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = scaling_match(
        xA_t, xB_t,
        kernel="euclidean_l2",
        C=C,
        k=k,
        target_delta=delta,
        device=device,
        seed=seed,
        fill_policy="greedy",
        verbose=False,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0
    cost = float(result.matching_cost)
    feasible = int((result.Mb != -1).sum().item())
    print(f"[spef] cost={cost:.4f}  phases={result.phases}  "
          f"matches={feasible}  runtime={runtime:.3f}s", flush=True)
    return {
        "cost": cost,
        "runtime_sec": runtime,
        "phases": int(result.phases),
        "iterations": int(result.iterations),
        "feasible_matches": feasible,
    }


def _run_pot(xA: np.ndarray, xB: np.ndarray, *, n: int) -> dict:
    print(f"[pot] building {n}x{n} L2 cost matrix...", flush=True)
    t_build = time.perf_counter()
    M = np.empty((n, n), dtype=np.float64)
    batch = 512
    for r in range(0, n, batch):
        re = min(r + batch, n)
        diff = xB[r:re, None, :].astype(np.float64) - xA[None, :, :].astype(np.float64)
        M[r:re, :] = np.sqrt((diff ** 2).sum(axis=2))
    build_time = time.perf_counter() - t_build
    print(f"[pot] matrix built in {build_time:.2f}s, solving EMD...", flush=True)
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(n, dtype=np.float64) / n
    t_solve = time.perf_counter()
    gamma = ot.emd(a, b, M, numItermax=10000000)
    solve_time = time.perf_counter() - t_solve
    opt_cost = float(np.sum(gamma * M) * n)
    print(f"[pot] opt_cost={opt_cost:.4f}  solve={solve_time:.3f}s", flush=True)
    return {
        "opt_cost": opt_cost,
        "build_time_sec": build_time,
        "solve_time_sec": solve_time,
        "runtime_sec": build_time + solve_time,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int,   required=True)
    p.add_argument("--seed",   type=int,   default=1)
    p.add_argument("--device", type=str,   default="cuda")
    p.add_argument("--k",      type=int,   default=512)
    p.add_argument("--delta",  type=float, default=0.001)
    p.add_argument("--dim",    type=int,   default=2)
    p.add_argument("--out",    type=str,   default=None)
    args = p.parse_args()

    n, seed, dim = args.n, args.seed, args.dim

    # Generate points scaled to [0, sqrt(n)]^dim so average L2 cost ~ O(1)
    rng = np.random.RandomState(seed)
    side = float(np.sqrt(n))
    xA = (rng.uniform(0, 1, size=(n, dim)) * side).astype(np.float32)
    xB = (rng.uniform(0, 1, size=(n, dim)) * side).astype(np.float32)

    print(f"\n{'='*60}", flush=True)
    print(f"  n={n}  seed={seed}  delta={args.delta}  dim={dim}  device={args.device}")
    print(f"{'='*60}\n", flush=True)

    C = _estimate_C(xA, xB, seed)

    print("\n--- SPEF scaled ---", flush=True)
    spef = _run_spef(xA, xB, n=n, C=C, k=args.k, delta=args.delta,
                     device=args.device, seed=seed)

    print("\n--- POT EMD ---", flush=True)
    pot = _run_pot(xA, xB, n=n)

    # Theoretical bound from NeurIPS paper: additive error <= 3*epsilon*n = C*delta*n
    theory_bound = C * args.delta * n
    additive_gap = spef["cost"] - pot["opt_cost"]
    approx_ratio = spef["cost"] / pot["opt_cost"] if pot["opt_cost"] > 0 else None
    theory_satisfied = bool(additive_gap <= theory_bound)

    print(f"\n--- Comparison ---", flush=True)
    print(f"  spef_cost     = {spef['cost']:.4f}", flush=True)
    print(f"  opt_cost      = {pot['opt_cost']:.4f}", flush=True)
    print(f"  approx_ratio  = {approx_ratio:.4f}", flush=True)
    print(f"  additive_gap  = {additive_gap:.4f}", flush=True)
    print(f"  3en_bound     = {theory_bound:.4f}", flush=True)
    print(f"  satisfied     = {theory_satisfied}", flush=True)

    output: dict[str, Any] = {
        "solver": "combined_synthetic",
        "params": {
            "n": n, "seed": seed, "delta": args.delta,
            "C": C, "k": args.k, "dim": dim, "device": args.device,
        },
        "spef": spef,
        "pot": pot,
        "comparison": {
            "approx_ratio": approx_ratio,
            "additive_gap": additive_gap,
            "theory_bound_3en": theory_bound,
            "theory_satisfied": theory_satisfied,
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote results to {args.out}", flush=True)
    else:
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
