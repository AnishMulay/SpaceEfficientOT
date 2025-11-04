#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "experiments" / "nyc_taxi_speed_haversine" / "batch" / "results" / "nyc_haversine"
CONFIGS_DIR = ROOT / "experiments" / "nyc_taxi_speed_haversine" / "batch" / "configs"
OUT_MD = ROOT / "experiments" / "nyc_taxi_speed_haversine" / "batch" / "REPORT.md"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(float(x))


def _bar(value: Optional[float], max_value: float, width: int = 14) -> str:
    # Unicode bar for a quick visual sense; handles None/missing
    if value is None or max_value <= 0:
        return "".ljust(width)
    # Normalize and clamp
    frac = min(max(value / max_value, 0.0), 1.0)
    # Use full block characters for simplicity
    filled = int(round(frac * width))
    return ("█" * filled + "░" * (width - filled))


@dataclass
class RunRow:
    run_name: str
    status: str
    n: Optional[int]
    stopping_condition: Optional[int]
    delta: Optional[float]
    speed_mps: Optional[float]
    y_max_meters: Optional[float]
    k: Optional[int]
    runtime_sec: Optional[float]
    matching_cost_km: Optional[float]
    avg_cost_m: Optional[float]


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    try:
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k)
        return cur if cur is not None else default
    except Exception:
        return default


def _params_from_dirname(name: str) -> Dict[str, Any]:
    # Fallback parser for names like n100000_sc5000_v8_d0.001_y10000
    parts = name.split("_")
    out: Dict[str, Any] = {}
    for p in parts:
        if p.startswith("n") and p[1:].isdigit():
            out["n"] = int(p[1:])
        elif p.startswith("sc") and p[2:].isdigit():
            out["stopping_condition"] = int(p[2:])
        elif p.startswith("v"):
            try:
                out["speed_mps"] = float(p[1:])
            except Exception:
                pass
        elif p.startswith("d"):
            try:
                out["delta"] = float(p[1:])
            except Exception:
                pass
        elif p.startswith("y"):
            try:
                out["y_max_meters"] = float(p[1:])
            except Exception:
                pass
    return out


def collect_runs(runs_dir: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    if not runs_dir.exists():
        return rows
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        result_path = child / "result.json"
        cfg_used_path = child / "config_used.json"

        status = "unknown"
        if meta_path.exists():
            try:
                status = _read_json(meta_path).get("status", "unknown")
            except Exception:
                status = "unknown"

        # Assemble params from multiple sources: result.json, config_used.json, then directory name
        params: Dict[str, Any] = {}
        res_params: Dict[str, Any] = {}
        if result_path.exists():
            try:
                data = _read_json(result_path)
                res_params = dict(_safe_get(data, "params", default={}) or {})
            except Exception:
                pass
        cfg_params: Dict[str, Any] = {}
        if cfg_used_path.exists():
            try:
                cfg_params = dict(_read_json(cfg_used_path))
            except Exception:
                pass
        # Start with result params, backfill missing from config, then from name tokens
        params = {**_params_from_dirname(child.name), **cfg_params, **res_params}

        # metrics
        runtime_sec: Optional[float] = None
        match_km: Optional[float] = None
        avg_m: Optional[float] = None
        if result_path.exists():
            try:
                data = _read_json(result_path)
                runtime_sec = _safe_get(data, "performance", "runtime_sec")
                match_km = _safe_get(data, "metrics", "matching_cost_km")
                avg_m = _safe_get(data, "metrics", "avg_cost_m")
            except Exception:
                pass

        rows.append(
            RunRow(
                run_name=child.name,
                status=status,
                n=to_int(params.get("n")),
                stopping_condition=to_int(params.get("stopping_condition")),
                delta=to_float(params.get("delta")),
                speed_mps=to_float(params.get("speed_mps")),
                y_max_meters=to_float(params.get("y_max_meters")),
                k=to_int(params.get("k")),
                runtime_sec=to_float(runtime_sec),
                matching_cost_km=to_float(match_km),
                avg_cost_m=to_float(avg_m),
            )
        )
    return rows


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def md_header(level: int, text: str) -> str:
    return "#" * level + f" {text}\n\n"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    # GitHub-style markdown table
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    out += "\n"
    return out


def md_details(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n" + body + "\n</details>\n\n"


def build_report(rows: List[RunRow]) -> str:
    if not rows:
        return md_header(1, "NYC Haversine Speed Experiments") + "No runs found.\n"

    # Normalize sets
    Ns = sorted({r.n for r in rows if r.n is not None})
    deltas = sorted({r.delta for r in rows if r.delta is not None})
    y_maxs = sorted({r.y_max_meters for r in rows if r.y_max_meters is not None})

    total_runs = len(rows)
    successes = sum(1 for r in rows if r.status == "success")
    incomplete = sum(1 for r in rows if r.status != "success")

    # Top matter
    md: List[str] = []
    md.append(md_header(1, "NYC Taxi — Haversine Speed Batch Results"))
    md.append(
        "This report summarizes batch experiments for NYC taxi matching with a haversine speed kernel.\n"
    )
    md.append("Parameters vary across runs: `n`, `stopping_condition`, `delta`, `speed_mps`, `y_max_meters`, and `k`.\n\n")

    # Overview
    md.append(md_header(2, "Overview"))
    overview_rows = [
        ["Total runs", str(total_runs)],
        ["Successful runs", str(successes)],
        ["Incomplete (started/failed/timeout)", str(incomplete)],
        ["Unique n", ", ".join(str(n) for n in Ns) or "-"],
        ["Unique delta", ", ".join(_fmt_float(d, 4) for d in deltas) or "-"],
        ["Unique y_max_meters", ", ".join(_fmt_float(y, 0) for y in y_maxs) or "-"],
    ]
    md.append(md_table(["Key", "Value"], overview_rows))

    # Success grid per n x delta
    md.append(md_header(3, "Success Grid (by n and delta)"))
    grid_headers = ["n/delta"] + [str(_fmt_float(d, 4)) for d in deltas]
    grid_rows: List[List[str]] = []
    for n in Ns:
        row = [str(n)]
        for d in deltas:
            sel = [r for r in rows if r.n == n and r.delta == d]
            ok = sum(1 for r in sel if r.status == "success")
            tot = len(sel)
            row.append(f"{ok}/{tot}")
        grid_rows.append(row)
    md.append(md_table(grid_headers, grid_rows))

    # Per-n sections
    for n in Ns:
        md.append(md_header(2, f"n = {n}"))
        for d in deltas:
            subset = [r for r in rows if r.n == n and r.delta == d]
            if not subset:
                continue
            md.append(md_header(3, f"delta = {_fmt_float(d, 4)}"))
            # Group by y_max
            yvals = sorted({r.y_max_meters for r in subset if r.y_max_meters is not None})
            for y in yvals:
                sub2 = [r for r in subset if r.y_max_meters == y]
                if not sub2:
                    continue
                md.append(md_header(4, f"y_max_meters = {_fmt_float(y, 0)}"))

                # Sort rows by stopping_condition
                sub2.sort(key=lambda r: (r.stopping_condition or 0))

                # Compute scales for bars
                max_rt = max((r.runtime_sec or 0) for r in sub2)
                max_tot = max((r.matching_cost_km or 0) for r in sub2)
                max_avg = max((r.avg_cost_m or 0) for r in sub2)

                table_rows: List[List[str]] = []
                table_headers = [
                    "stopping_condition",
                    "status",
                    "runtime_sec",
                    "rt_bar",
                    "total_km",
                    "tot_bar",
                    "avg_m",
                    "avg_bar",
                ]
                for r in sub2:
                    table_rows.append(
                        [
                            str(r.stopping_condition or "-"),
                            ("✓" if r.status == "success" else r.status),
                            _fmt_float(r.runtime_sec, 2),
                            f"`{_bar(r.runtime_sec, max_rt)}`",
                            _fmt_float(r.matching_cost_km, 3),
                            f"`{_bar(r.matching_cost_km, max_tot)}`",
                            _fmt_float(r.avg_cost_m, 3),
                            f"`{_bar(r.avg_cost_m, max_avg)}`",
                        ]
                    )

                md.append(md_table(table_headers, table_rows))

        # Incomplete runs for this n
        inc = [r for r in rows if r.n == n and r.status != "success"]
        if inc:
            body_rows = []
            for r in sorted(inc, key=lambda x: (x.delta or 0, x.y_max_meters or 0, x.stopping_condition or 0)):
                body_rows.append(
                    [
                        r.run_name,
                        r.status,
                        str(r.stopping_condition or "-"),
                        _fmt_float(r.delta, 4),
                        _fmt_float(r.y_max_meters, 0),
                    ]
                )
            body = md_table(["run", "status", "stopping_condition", "delta", "y_max_meters"], body_rows)
            md.append(md_details("Incomplete runs (started/failed/timeout)", body))

    # Footer
    md.append(md_header(2, "Notes"))
    md.append(
        "- Values shown: total matching cost (km), average matching cost (m), runtime (sec), and completion status.\n"
    )
    md.append(
        "- Bars provide relative scale within each (n, delta, y_max_meters) group for quick visual comparison.\n"
    )
    md.append("- Status values reflect meta.json: ✓ = success; others indicate no final artifact.\n")

    return "".join(md)


def main() -> None:
    rows = collect_runs(RUNS_DIR)
    report = build_report(rows)
    OUT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
