#!/usr/bin/env python3
"""
aggregate_results.py
====================
Merges all per-job CSVs from batch/results/, then computes mean +/- std
per (n, method) cell across trials.

Run after all SLURM jobs finish:
    python aggregate_results.py

Writes:
    batch/results/raw_combined.csv   -- all rows merged into one file
    batch/results/summary.csv        -- mean +/- std per (n, method)

Paste summary.csv here for plotting and analysis.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Metrics to aggregate
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
# Load + merge all per-job CSVs
# ---------------------------------------------------------------------------

def _load_all(results_dir: Path) -> list[dict]:
    csv_files = sorted(results_dir.glob("results_*.csv"))
    if not csv_files:
        print(f"[ERROR] No results_*.csv files found in {results_dir}",
              file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for path in csv_files:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            all_rows.extend(rows)

    print(f"Loaded {len(all_rows)} rows from {len(csv_files)} files.",
          file=sys.stderr)
    return all_rows


def _write_combined(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    cell_data: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cell_counts: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))

    success_rows = 0
    for row in rows:
        status = row.get("status", "")
        n_raw  = _safe_float(row.get("n"))
        method = row.get("method", "")
        if n_raw is None or not method:
            continue
        n = int(n_raw)
        cell_counts[(n, method)]["total"] += 1

        if status != "success":
            cell_counts[(n, method)][f"status_{status}"] += 1
            continue

        success_rows += 1
        for col in METRICS:
            v = _safe_float(row.get(col))
            if v is not None:
                cell_data[(n, method)][col].append(v)

    print(f"{success_rows} successful rows across "
          f"{len(cell_data)} (n, method) cells.", file=sys.stderr)

    all_n = sorted({n for n, _ in {**cell_data, **cell_counts}})
    all_methods = METHOD_ORDER + sorted(
        m for m in {me for _, me in {**cell_data, **cell_counts}}
        if m not in METHOD_ORDER
    )

    out_rows = []
    for n in all_n:
        for method in all_methods:
            key    = (n, method)
            counts = cell_counts.get(key, {})
            data   = cell_data.get(key, {})
            if not counts:
                continue

            n_success = len(data.get("runtime_sec", []))
            row_out = {
                "n":              n,
                "method":         method,
                "n_jobs_total":   counts.get("total", 0),
                "n_jobs_success": n_success,
            }

            for col in METRICS:
                vals = data.get(col, [])
                if vals:
                    s = _stats(vals)
                    row_out[f"{col}_mean"] = round(s["mean"], 6)
                    row_out[f"{col}_std"]  = round(s["std"],  6)
                    row_out[f"{col}_min"]  = round(s["min"],  6)
                    row_out[f"{col}_max"]  = round(s["max"],  6)
                    row_out[f"{col}_n"]    = s["count"]
                else:
                    for suffix in ("mean", "std", "min", "max", "n"):
                        row_out[f"{col}_{suffix}"] = ""

            out_rows.append(row_out)

    return out_rows


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    cols = ["runtime_sec", "avg_cost_km", "match_rate"]
    hdr  = f"{'n':>7}  {'method':<10}" + "".join(
        f"  {METRICS[c]:>24}" for c in cols
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    prev_n = None
    for row in rows:
        n = row["n"]
        if n != prev_n and prev_n is not None:
            print()
        prev_n = n
        line = f"{n:>7}  {row['method']:<10}"
        for c in cols:
            mn  = row.get(f"{c}_mean", "")
            std = row.get(f"{c}_std",  "")
            if mn != "":
                cell = f"{float(mn):.4f} +/- {float(std):.4f}"
            else:
                cell = "n/a"
            line += f"  {cell:>24}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge per-job CSVs and compute summary statistics")
    ap.add_argument("--results-dir", default=None,
                    help="Directory containing results_*.csv files "
                         "(default: batch/results/ next to this script)")
    args = ap.parse_args()

    here        = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir \
                  else here / "batch" / "results"

    combined_path = results_dir / "raw_combined.csv"
    summary_path  = results_dir / "summary.csv"

    rows    = _load_all(results_dir)
    _write_combined(rows, combined_path)
    print(f"Combined CSV -> {combined_path}", file=sys.stderr)

    summary = aggregate(rows)
    if not summary:
        print("[WARNING] No successful rows found.", file=sys.stderr)
        sys.exit(0)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"Summary CSV  -> {summary_path}  ({len(summary)} rows)")
    _print_table(summary)


if __name__ == "__main__":
    main()