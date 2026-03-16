#!/usr/bin/env python3
"""Aggregate synthetic comparison results into a flat CSV and summary table."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

BATCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BATCH_DIR / "results"
OUT_PATH = RESULTS_DIR / "aggregated_results.csv"
FIELDNAMES = [
    "n",
    "seed",
    "delta",
    "C",
    "dim",
    "spef_cost",
    "spef_runtime_sec",
    "spef_phases",
    "spef_feasible_matches",
    "opt_cost",
    "pot_runtime_sec",
    "approx_ratio",
    "additive_gap",
    "theory_bound_3en",
    "theory_satisfied",
    "status",
]


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        next_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, next_key))
        else:
            flat[next_key] = value
    return flat


def _status_for(result_path: Path) -> str:
    meta_path = result_path.parent / "meta.json"
    if not meta_path.exists():
        return "success"
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return str(meta.get("status", "success"))
    except Exception:
        return "success"


def _row_from_result(result_path: Path) -> dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    flat = _flatten_dict(data)
    return {
        "n": flat.get("params_n"),
        "seed": flat.get("params_seed"),
        "delta": flat.get("params_delta"),
        "C": flat.get("params_C"),
        "dim": flat.get("params_dim"),
        "spef_cost": flat.get("spef_cost"),
        "spef_runtime_sec": flat.get("spef_runtime_sec"),
        "spef_phases": flat.get("spef_phases"),
        "spef_feasible_matches": flat.get("spef_feasible_matches"),
        "opt_cost": flat.get("pot_opt_cost"),
        "pot_runtime_sec": flat.get("pot_runtime_sec"),
        "approx_ratio": flat.get("comparison_approx_ratio"),
        "additive_gap": flat.get("comparison_additive_gap"),
        "theory_bound_3en": flat.get("comparison_theory_bound_3en"),
        "theory_satisfied": flat.get("comparison_theory_satisfied"),
        "status": _status_for(result_path),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def main() -> None:
    result_files = sorted(RESULTS_DIR.rglob("result.json"))
    if not result_files:
        print("No results found.")
        return

    rows = [_row_from_result(path) for path in result_files]
    rows.sort(key=lambda row: (int(row["n"]), int(row["seed"])))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["n"])].append(row)

    print(f"Wrote {len(rows)} rows -> {OUT_PATH}")
    print("")
    print(f"{'N':>8}  {'approx_ratio mean±std':>24}  {'spef mean s':>12}  {'pot mean s':>11}")
    for n in sorted(grouped):
        approx_values = [float(row["approx_ratio"]) for row in grouped[n]]
        spef_values = [float(row["spef_runtime_sec"]) for row in grouped[n]]
        pot_values = [float(row["pot_runtime_sec"]) for row in grouped[n]]
        approx_mean, approx_std = _mean_std(approx_values)
        print(
            f"{n:8d}  "
            f"{approx_mean:12.6f} ± {approx_std:8.6f}  "
            f"{mean(spef_values):12.3f}  "
            f"{mean(pot_values):11.3f}"
        )


if __name__ == "__main__":
    main()
