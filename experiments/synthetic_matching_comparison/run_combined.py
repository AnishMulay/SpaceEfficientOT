#!/usr/bin/env python3
"""Run scaled SPEF and exact POT EMD on shared synthetic Euclidean data."""
from __future__ import annotations

import argparse
import json
import os
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
try:
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
except Exception:
    pass

import spef_ot.kernels.euclidean_sq  # noqa: F401
from spef_ot import scaling_match

POT_BATCH_ROWS = 512
PAIR_SAMPLE_SIZE = 64


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scaled SPEF + POT EMD on synthetic squared-Euclidean data"
    )
    parser.add_argument("--n", type=int, required=True, help="Number of points per side")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--k", type=int, default=512, help="Tile size")
    parser.add_argument("--delta", type=float, default=0.001, help="Target delta")
    parser.add_argument("--dim", type=int, default=2, help="Point dimensionality")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    return parser


def _estimate_C(xA: np.ndarray, xB: np.ndarray, rng: np.random.RandomState) -> float:
    idxA = rng.randint(0, xA.shape[0], size=PAIR_SAMPLE_SIZE)
    idxB = rng.randint(0, xB.shape[0], size=PAIR_SAMPLE_SIZE)
    diffs = xA[idxA].astype(np.float64) - xB[idxB].astype(np.float64)
    sq_dists = np.sum(diffs * diffs, axis=1)
    max_sq_dist = float(np.max(sq_dists))
    return 4.0 * max_sq_dist


def _build_cost_matrix(xA: np.ndarray, xB: np.ndarray) -> np.ndarray:
    n = xA.shape[0]
    xA64 = xA.astype(np.float64, copy=False)
    xB64 = xB.astype(np.float64, copy=False)
    M = np.empty((n, n), dtype=np.float64)
    for row_start in range(0, n, POT_BATCH_ROWS):
        row_end = min(row_start + POT_BATCH_ROWS, n)
        diff = xB64[row_start:row_end, None, :] - xA64[None, :, :]
        M[row_start:row_end, :] = np.sum(diff * diff, axis=2, dtype=np.float64)
    return M


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def main() -> None:
    args = _build_parser().parse_args()

    rng = np.random.RandomState(args.seed)
    xA = rng.uniform(0.0, 1.0, size=(args.n, args.dim)).astype(np.float32)
    xB = rng.uniform(0.0, 1.0, size=(args.n, args.dim)).astype(np.float32)

    C = _estimate_C(xA, xB, rng)
    print(f"Estimated C: {C:.10f}", flush=True)

    xA_t = torch.from_numpy(xA)
    xB_t = torch.from_numpy(xB)
    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    spef_start = time.perf_counter()
    spef_result = scaling_match(
        xA_t,
        xB_t,
        kernel="euclidean_sq",
        C=C,
        k=args.k,
        target_delta=args.delta,
        device=device,
        seed=args.seed,
        fill_policy="greedy",
        verbose=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    spef_runtime_sec = time.perf_counter() - spef_start

    spef_cost = _to_float(spef_result.matching_cost)
    spef_phases = int(spef_result.phases)
    spef_iterations = int(spef_result.iterations)
    spef_feasible_matches = int(
        spef_result.metrics.get(
            "feasible_matches",
            int((spef_result.Mb != -1).sum().item()),
        )
    )

    pot_build_start = time.perf_counter()
    M = _build_cost_matrix(xA, xB)
    pot_build_time_sec = time.perf_counter() - pot_build_start

    a = np.ones(args.n, dtype=np.float64) / float(args.n)
    b = np.ones(args.n, dtype=np.float64) / float(args.n)

    pot_solve_start = time.perf_counter()
    gamma = ot.emd(a, b, M, numItermax=10000000)
    pot_solve_time_sec = time.perf_counter() - pot_solve_start
    pot_runtime_sec = pot_build_time_sec + pot_solve_time_sec
    opt_cost = float(np.sum(gamma * M, dtype=np.float64) * args.n)

    additive_gap = spef_cost - opt_cost
    approx_ratio = spef_cost / opt_cost
    theory_bound_3en = C * args.delta * args.n
    theory_satisfied = bool(additive_gap <= theory_bound_3en)

    output = {
        "solver": "combined_synthetic",
        "params": {
            "n": args.n,
            "seed": args.seed,
            "delta": args.delta,
            "C": C,
            "k": args.k,
            "dim": args.dim,
            "device": str(device),
        },
        "spef": {
            "runtime_sec": spef_runtime_sec,
            "cost": spef_cost,
            "phases": spef_phases,
            "iterations": spef_iterations,
            "feasible_matches": spef_feasible_matches,
        },
        "pot": {
            "runtime_sec": pot_runtime_sec,
            "build_time_sec": pot_build_time_sec,
            "solve_time_sec": pot_solve_time_sec,
            "opt_cost": opt_cost,
        },
        "comparison": {
            "approx_ratio": approx_ratio,
            "additive_gap": additive_gap,
            "theory_bound_3en": theory_bound_3en,
            "theory_satisfied": theory_satisfied,
        },
    }

    print(json.dumps(output, indent=2), flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"Wrote results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
