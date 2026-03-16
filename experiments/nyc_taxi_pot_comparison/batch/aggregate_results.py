#!/usr/bin/env python3
"""Aggregate per-run combined result.json files into a flat CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BATCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BATCH_DIR / "results"
DEFAULT_OUT = BATCH_DIR / "pot_comparison_results.csv"

SOLVER_ORDER = {"spef_unscaled": 0, "spef_scaled": 1, "pot_partial": 2}


def _flatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten_dict(value))
        else:
            flat[key] = value
    return flat


def _row_from_nested_run(
    *,
    run_dir: Path,
    parent: dict[str, Any],
    nested: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "run_dir": str(run_dir.relative_to(BATCH_DIR)),
        "combined_solver": parent.get("solver"),
        "combined_runtime_sec": parent.get("performance", {}).get("runtime_sec"),
    }
    row.update(_flatten_dict(nested))
    return row


def _rows_from_result(result_path: Path) -> list[dict[str, Any]]:
    run_dir = result_path.parent
    meta_path = run_dir / "meta.json"
    status = "success"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            status = meta.get("status", status)
        except Exception:
            pass

    if not result_path.exists():
        cfg_path = run_dir / "config_used.json"
        row: dict[str, Any] = {
            "status": status,
            "run_dir": str(run_dir.relative_to(BATCH_DIR)),
            "combined_solver": "combined",
        }
        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                row.update(_flatten_dict(cfg))
                if "n_requested" not in row and "n" in row:
                    row["n_requested"] = row["n"]
            except Exception:
                pass
        return [row]

    try:
        with result_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as ex:
        return [{
            "status": f"unreadable:{ex}",
            "run_dir": str(run_dir.relative_to(BATCH_DIR)),
        }]

    if data.get("solver") != "combined":
        row = {
            "status": status,
            "run_dir": str(run_dir.relative_to(BATCH_DIR)),
        }
        row.update(_flatten_dict(data))
        return [row]

    rows: list[dict[str, Any]] = []
    for nested in data.get("runs", {}).values():
        if not isinstance(nested, dict):
            continue
        rows.append(_row_from_nested_run(run_dir=run_dir, parent=data, nested=nested, status=status))
    return rows


def _collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "solver",
        "n_requested",
        "n_used",
        "seed",
        "status",
        "run_dir",
        "combined_solver",
        "combined_runtime_sec",
    ]
    seen: set[str] = set()
    fieldnames: list[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    return fieldnames


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate combined result.json files into a CSV")
    ap.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help=f"Root of per-run result directories (default: {RESULTS_DIR})",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    result_files = sorted(results_dir.rglob("result.json"))
    meta_only_dirs = sorted(
        p.parent for p in results_dir.rglob("meta.json")
        if not (p.parent / "result.json").exists()
    )

    rows: list[dict[str, Any]] = []
    for path in result_files:
        rows.extend(_rows_from_result(path))
    for run_dir in meta_only_dirs:
        rows.extend(_rows_from_result(run_dir / "result.json"))

    if not rows:
        print("No results found. Have the jobs finished?")
        return

    def _sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
        solver_rank = SOLVER_ORDER.get(str(row.get("solver", "")), 99)
        n = int(row["n_requested"]) if row.get("n_requested") is not None else 0
        seed = int(row["seed"]) if row.get("seed") is not None else 0
        return (n, seed, solver_rank)

    rows.sort(key=_sort_key)
    fieldnames = _collect_fieldnames(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success = sum(1 for row in rows if row.get("status") == "success")
    failed = sum(1 for row in rows if row.get("status") not in ("success", None))
    print(f"Wrote {len(rows)} rows ({success} success, {failed} non-success) -> {out_path}")


if __name__ == "__main__":
    main()
