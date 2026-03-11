#!/usr/bin/env python3
"""Aggregate all per-run result.json files into a single CSV.

Run this after all SLURM jobs have finished:
  python experiments/nyc_taxi_exact_comparison/batch/aggregate_results.py

Output: experiments/nyc_taxi_exact_comparison/batch/exact_comparison_results.csv

Each row corresponds to one completed run. Failed / timed-out runs are included
with status="failed"/"timeout" and empty metric columns so you can see what is missing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

BATCH_DIR   = Path(__file__).resolve().parent
RESULTS_DIR = BATCH_DIR / "results"
DEFAULT_OUT = BATCH_DIR / "exact_comparison_results.csv"

FIELDNAMES = [
    # Identity
    "solver",
    "n_requested",
    "n_used",
    "seed",
    # Status
    "status",
    "run_dir",
    # Fixed params (for sanity-checking)
    "speed_mps",
    "y_max_meters",
    "future_only",
    "delta",
    "C",
    "k",
    "stopping_condition",
    "device",
    # Timing
    "runtime_sec",
    "build_time_sec",    # ortools only
    "solve_time_sec",    # ortools only
    "phases",            # spef_scaled only
    "total_inner_loops", # spef only
    # Solution quality
    "matching_cost_m",
    "matching_cost_km",
    "avg_cost_m",
    "avg_cost_km",
    "feasible_matches",
    "free_B",
    # ortools graph stats
    "num_feasible_edges",
    "num_arcs",
    "num_nodes",
    # SPEF filter stats
    "removed_by_future",
    "removed_by_speed",
    "removed_by_ymax",
]


def _safe(d: dict, *keys: str) -> Any:
    """Drill into nested dicts safely, returning None if any key is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _row_from_result(result_path: Path) -> Dict[str, Any]:
    """Build a CSV row from a result.json file. Returns partial row on error."""
    run_dir = result_path.parent
    row: Dict[str, Any] = {f: None for f in FIELDNAMES}
    row["run_dir"] = str(run_dir.relative_to(BATCH_DIR))

    # Try to read meta for status override
    meta_path = run_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    if not result_path.exists():
        row["status"] = meta.get("status", "missing")
        # Try to recover identity from config_used
        cfg_path = run_dir / "config_used.json"
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                row["solver"]   = cfg.get("solver")
                row["n_requested"] = cfg.get("n")
                row["seed"]     = cfg.get("seed")
            except Exception:
                pass
        return row

    try:
        with result_path.open("r", encoding="utf-8") as f:
            data: dict = json.load(f)
    except Exception as ex:
        row["status"] = f"unreadable:{ex}"
        return row

    row["status"]  = "success"
    row["solver"]  = data.get("solver")

    p = data.get("params", {})
    row["n_requested"]       = p.get("n_requested")
    row["n_used"]            = p.get("n_used")
    row["seed"]              = p.get("seed")
    row["speed_mps"]         = p.get("speed_mps")
    row["y_max_meters"]      = p.get("y_max_meters")
    row["future_only"]       = p.get("future_only")
    row["delta"]             = p.get("delta")
    row["C"]                 = p.get("C")
    row["k"]                 = p.get("k")
    row["stopping_condition"] = p.get("stopping_condition")
    row["device"]            = p.get("device")

    perf = data.get("performance", {})
    row["runtime_sec"]       = perf.get("runtime_sec")
    row["build_time_sec"]    = perf.get("build_time_sec")
    row["solve_time_sec"]    = perf.get("solve_time_sec")
    row["phases"]            = perf.get("phases")
    row["total_inner_loops"] = perf.get("total_inner_loops")
    row["num_feasible_edges"] = perf.get("num_feasible_edges")
    row["num_arcs"]          = perf.get("num_arcs")
    row["num_nodes"]         = perf.get("num_nodes")

    m = data.get("metrics", {})
    row["matching_cost_m"]   = m.get("matching_cost_m")
    row["matching_cost_km"]  = m.get("matching_cost_km")
    row["avg_cost_m"]        = m.get("avg_cost_m")
    row["avg_cost_km"]       = m.get("avg_cost_km")
    row["feasible_matches"]  = m.get("feasible_matches")
    row["free_B"]            = m.get("free_B")
    row["removed_by_future"] = m.get("removed_by_future")
    row["removed_by_speed"]  = m.get("removed_by_speed")
    row["removed_by_ymax"]   = m.get("removed_by_ymax")

    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate per-run result.json files into exact_comparison_results.csv"
    )
    ap.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help=f"Root of per-run result directories (default: {RESULTS_DIR})"
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Output CSV path (default: {DEFAULT_OUT})"
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.out)

    # Collect all result.json files (walk entire tree)
    result_files = sorted(results_dir.rglob("result.json"))

    # Also include run dirs where result.json is absent but meta.json exists
    # (captures failed/timed-out jobs)
    meta_only_dirs = sorted(
        p.parent for p in results_dir.rglob("meta.json")
        if not (p.parent / "result.json").exists()
    )

    rows = []
    for rf in result_files:
        rows.append(_row_from_result(rf))

    for run_dir in meta_only_dirs:
        # Build a partial row
        dummy_result = run_dir / "result.json"  # doesn't exist
        rows.append(_row_from_result(dummy_result))

    if not rows:
        print("No results found. Have the jobs finished?")
        return

    # Sort by solver, n, seed for readability
    def _sort_key(r: dict):
        solver_order = {"spef_unscaled": 0, "spef_scaled": 1, "ortools": 2}
        s = solver_order.get(str(r.get("solver", "")), 99)
        n = int(r["n_requested"]) if r.get("n_requested") is not None else 0
        seed = int(r["seed"]) if r.get("seed") is not None else 0
        return (s, n, seed)

    rows.sort(key=_sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    success = sum(1 for r in rows if r.get("status") == "success")
    failed  = sum(1 for r in rows if r.get("status") not in ("success", None))
    print(f"Wrote {len(rows)} rows ({success} success, {failed} non-success) → {out_path}")


if __name__ == "__main__":
    main()
