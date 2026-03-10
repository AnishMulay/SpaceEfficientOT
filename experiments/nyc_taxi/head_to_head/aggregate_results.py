#!/usr/bin/env python3
"""
aggregate_results.py
====================
Post-processing step.  Run this after all SLURM jobs finish:

    python aggregate_results.py                          # uses default paths
    python aggregate_results.py --raw results/raw_results.csv

Reads raw_results.csv and writes two outputs:
  1. summary_results.csv  — mean ± std per (n, method) cell, one row per cell
  2. Prints a human-readable table to stdout

The summary CSV is the one you paste into Claude (or any plotting tool)
for the next step: plotting and analysis.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Metrics to aggregate (column name → display label)
# ---------------------------------------------------------------------------

METRICS = {
    "runtime_sec":      "Runtime (s)",
    "total_cost_km":    "Total Cost (km)",
    "avg_cost_km":      "Avg Cost/Match (km)",
    "match_rate":       "Match Rate",
    "feasible_matches": "Feasible Matches",
}

METHOD_ORDER = ["exact", "unscaled", "scaled"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "nan", "None") else None
    except (ValueError, TypeError):
        return None


def _stats(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {
        "count": len(a),
        "mean":  float(np.mean(a)),
        "std":   float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "min":   float(np.min(a)),
        "max":   float(np.max(a)),
    }


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> list[dict]:
    """
    Group by (n, method), compute stats for each metric.
    Only rows with status='success' are included in stats.
    Returns a list of flat dicts suitable for CSV writing.
    """
    from collections import defaultdict

    # cell_data[(n, method)][metric] = [values...]
    cell_data: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cell_counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_rows = len(rows)
    success_rows = 0

    for row in rows:
        status = row.get("status", "")
        n_raw  = _safe_float(row.get("n"))
        method = row.get("method", "")

        if n_raw is None or not method:
            continue

        n = int(n_raw)
        cell_counts[(n, method)]["total"] = cell_counts[(n, method)].get("total", 0) + 1

        if status != "success":
            cell_counts[(n, method)][f"status_{status}"] = (
                cell_counts[(n, method)].get(f"status_{status}", 0) + 1
            )
            continue

        success_rows += 1
        for col in METRICS:
            v = _safe_float(row.get(col))
            if v is not None:
                cell_data[(n, method)][col].append(v)

    print(f"Loaded {total_rows} rows, {success_rows} successful.", file=sys.stderr)

    # Build output rows
    all_n      = sorted({n for n, _ in cell_data} | {n for n, _ in cell_counts})
    all_methods = METHOD_ORDER + sorted(
        m for m in {me for _, me in cell_data} if m not in METHOD_ORDER
    )

    out_rows = []
    for n in all_n:
        for method in all_methods:
            key = (n, method)
            counts = cell_counts.get(key, {})
            data   = cell_data.get(key, {})

            if not counts:
                continue

            row_out: dict = {
                "n":             n,
                "method":        method,
                "n_jobs_total":  counts.get("total", 0),
                "n_jobs_success": sum(
                    v for k, v in counts.items()
                    if k == "total"   # subtract non-success
                ) - sum(v for k, v in counts.items() if k != "total"),
            }
            # cleaner: just count success from data
            row_out["n_jobs_success"] = len(data.get("runtime_sec", []))

            for col in METRICS:
                vals = data.get(col, [])
                if vals:
                    s = _stats(vals)
                    row_out[f"{col}_mean"]  = round(s["mean"], 6)
                    row_out[f"{col}_std"]   = round(s["std"],  6)
                    row_out[f"{col}_min"]   = round(s["min"],  6)
                    row_out[f"{col}_max"]   = round(s["max"],  6)
                    row_out[f"{col}_n"]     = s["count"]
                else:
                    for suffix in ("mean", "std", "min", "max", "n"):
                        row_out[f"{col}_{suffix}"] = ""

            out_rows.append(row_out)

    return out_rows


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    """Print a compact table of mean ± std for key metrics."""
    key_metrics = ["runtime_sec", "avg_cost_km", "match_rate"]

    header = f"{'n':>7}  {'method':<10}" + "".join(
        f"  {METRICS[m]:>22}" for m in key_metrics
    )
    print("\n" + header)
    print("─" * len(header))

    prev_n = None
    for row in rows:
        n = row["n"]
        if n != prev_n and prev_n is not None:
            print()
        prev_n = n
        method = row["method"]
        line = f"{n:>7}  {method:<10}"
        for m in key_metrics:
            mn  = row.get(f"{m}_mean", "")
            std = row.get(f"{m}_std",  "")
            if mn != "":
                cell = f"{float(mn):.4f} ± {float(std):.4f}"
            else:
                cell = "n/a"
            line += f"  {cell:>22}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate raw experiment results")
    ap.add_argument("--raw", default=None,
                    help="Path to raw_results.csv "
                         "(default: results/raw_results.csv next to this script)")
    ap.add_argument("--out", default=None,
                    help="Output summary CSV path "
                         "(default: results/summary_results.csv)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    raw_path = Path(args.raw) if args.raw else here / "results" / "raw_results.csv"
    out_path = Path(args.out) if args.out else here / "results" / "summary_results.csv"

    if not raw_path.exists():
        print(f"[ERROR] raw results file not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    rows     = _load_raw(raw_path)
    summary  = aggregate(rows)

    if not summary:
        print("[WARNING] No successful rows found — nothing to write.", file=sys.stderr)
        sys.exit(0)

    # Write summary CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"Summary written → {out_path}  ({len(summary)} rows)")
    _print_table(summary)


if __name__ == "__main__":
    main()
