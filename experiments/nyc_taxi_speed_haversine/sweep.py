#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
RUN_PY = EXP_DIR / "run.py"


def _parse_int_list(arg: str) -> list[int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_float_list(arg: str) -> list[float]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [float(p) for p in parts]


def _ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class RunSpec:
    n: int
    speed_mps: float
    stopping_condition: int


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep NYC taxi Haversine Speed experiment across n, speed, with per-n stopping conditions",
        epilog=(
            "Example: python experiments/NYC_Taxi_Speed_Haversine/sweep.py "
            "--n 100000,200000,300000 --speeds 8,10,12 --delta 0.001 --C 100000 "
            "--timeout-sec 1800"
        ),
    )

    # Parameter grids
    p.add_argument("--n", dest="n_list", type=_parse_int_list, default=None,
                   help="Comma-separated list of n values (e.g., 100000,200000,300000)")
    p.add_argument("--speeds", dest="speeds", type=_parse_float_list, default=None,
                   help="Comma-separated list of speeds in m/s (e.g., 8,10,12)")
    p.add_argument("--stopping", dest="stopping", type=_parse_int_list, default=None,
                   help="Optional: comma-separated list of global stopping_condition values for all n; "
                        "if omitted, per-n defaults are used")

    # Fixed params
    p.add_argument("--delta", type=float, default=0.001, help="Delta scaling (default: 0.001)")
    p.add_argument("--C", dest="C", type=float, default=100000.0,
                   help="Scaling constant C in meters (default: 100000)")
    p.add_argument("--k", type=int, default=512, help="Tile size for solver batches (default: 512)")
    p.add_argument("--y-max-meters", dest="y_max_meters", type=float, default=100000.0,
                   help="Clamp Haversine distance to this threshold before integerization (default: 100000)")
    p.add_argument("--future-only", dest="future_only", action="store_true", default=False,
                   help="Force future-only constraint in run.py")
    p.add_argument("--fill-policy", dest="fill_policy", choices=("none", "greedy"), default="none",
                   help="Fill policy for solver (default: none)")

    # Data/config passthroughs
    p.add_argument("--input", type=str, default=None, help="Optional path to NYC taxi CSV/Parquet")
    p.add_argument("--date", type=str, default=None, help="Filter date (YYYY-MM-DD)")
    p.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda)")
    p.add_argument("--seed", type=int, default=None, help="Seed to pass to run.py")

    # Execution controls
    p.add_argument("--timeout-sec", type=int, default=1800,
                   help="Max seconds per run (default: 1800 = 30 minutes)")
    p.add_argument("--no-warmup", dest="no_warmup", action="store_true", default=True,
                   help="Disable warm-up run in run.py (default: enabled)")
    p.add_argument("--preview-count", dest="preview_count", type=int, default=0,
                   help="Preview rows to print inside run.py (default: 0)")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=False,
                   help="Skip runs whose output JSON already exists")

    # Output controls
    p.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                   help="Directory to place sweep outputs (default: experiments/NYC_Taxi_Speed_Haversine/outputs/sweep_<ts>)")

    return p


def _default_grid() -> tuple[list[int], list[float]]:
    # Defaults per request: large-N only
    n_list = [100000, 200000, 300000]
    speeds = [8.0, 10.0, 12.0]
    return n_list, speeds


def _make_out_dir(base: str | None) -> Path:
    if base is None:
        base_path = EXP_DIR / "outputs" / f"sweep_{_ts()}"
    else:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = (EXP_DIR / base_path).resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def _run_one(spec: RunSpec, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    run_name = f"run_n{spec.n}_sc{spec.stopping_condition}_v{int(spec.speed_mps)}"
    out_path = out_dir / f"{run_name}.json"

    if args.skip_existing and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {"status": "skipped", "run": run_name, "spec": asdict(spec), "result": data}
        except Exception:
            pass  # fall through and rerun on read error

    cmd: list[str] = [sys.executable, str(RUN_PY)]

    # Required grid params
    cmd += ["--n", str(spec.n)]
    cmd += ["--stopping-condition", str(spec.stopping_condition)]
    cmd += ["--speed-mps", str(spec.speed_mps)]

    # Fixed params
    cmd += ["--delta", str(args.delta)]
    cmd += ["--C", str(args.C)]
    cmd += ["--k", str(args.k)]
    cmd += ["--y-max-meters", str(args.y_max_meters)]
    cmd += ["--fill-policy", str(args.fill_policy)]

    # Prefer to explicitly enforce future-only if requested
    if args.future_only:
        cmd += ["--future-only"]

    # Avoid preview noise and warmup overhead
    cmd += ["--preview-count", str(int(args.preview_count))]
    if args.no_warmup:
        cmd += ["--no-warmup"]

    # Optional passthroughs
    if args.input:
        cmd += ["--input", args.input]
    if args.date:
        cmd += ["--date", args.date]
    if args.device:
        cmd += ["--device", args.device]
    if args.seed is not None:
        cmd += ["--seed", str(int(args.seed))]

    # Write per-run output to file; ignore stdout logs from run.py
    cmd += ["--out", str(out_path)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(args.timeout_sec),
        )
    except subprocess.TimeoutExpired as ex:
        return {
            "status": "timeout",
            "run": run_name,
            "spec": asdict(spec),
            "timeout_sec": int(args.timeout_sec),
            "stdout_tail": ex.stdout[-2000:] if ex.stdout else None,
            "stderr_tail": ex.stderr[-2000:] if ex.stderr else None,
        }
    except Exception as ex:  # runtime or OSErrors
        return {
            "status": "failed",
            "run": run_name,
            "spec": asdict(spec),
            "error": f"{type(ex).__name__}: {ex}",
        }

    # Non-zero return code but not timeout
    if proc.returncode != 0:
        return {
            "status": "failed",
            "run": run_name,
            "spec": asdict(spec),
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else None,
        }

    # Read the JSON artifact we asked run.py to write
    try:
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", "run": run_name, "spec": asdict(spec), "result": data}
    except Exception as ex:
        return {
            "status": "failed",
            "run": run_name,
            "spec": asdict(spec),
            "error": f"Could not read output JSON: {type(ex).__name__}: {ex}",
        }


def _summarize_markdown(runs: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# NYC Taxi Haversine Speed – Sweep Summary")
    lines.append("")
    lines.append("| n | speed_mps | stopping_condition | status | runtime_sec | iterations | feasible | free_B | removed_future | removed_speed | removed_ymax | total_cost_km |")
    lines.append("|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        spec = r.get("spec", {})
        status = r.get("status", "?")
        if status == "success":
            res = r.get("result", {})
            perf = res.get("performance", {})
            metrics = res.get("metrics", {})
            lines.append(
                f"| {spec.get('n')} | {spec.get('speed_mps')} | {spec.get('stopping_condition')} | {status} | "
                f"{perf.get('runtime_sec', '')} | {perf.get('iterations', '')} | {metrics.get('feasible_matches', '')} | {metrics.get('free_B', '')} | "
                f"{metrics.get('removed_by_future', '')} | {metrics.get('removed_by_speed', '')} | {metrics.get('removed_by_ymax', '')} | {metrics.get('matching_cost_km', '')} |"
            )
        else:
            lines.append(
                f"| {spec.get('n')} | {spec.get('speed_mps')} | {spec.get('stopping_condition')} | {status} |  |  |  |  |  |  |  |  |"
            )
    lines.append("")
    lines.append("Notes: status=timeout indicates the 10-minute cap was exceeded.")
    return "\n".join(lines)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    n_list, speeds = _default_grid()
    if args.n_list is not None:
        n_list = args.n_list
    if args.speeds is not None:
        speeds = args.speeds

    out_dir = _make_out_dir(args.out_dir)

    # Build per-n stopping conditions
    if args.stopping is not None:
        # Global override for all n
        stopping_map: dict[int, list[int]] = {int(n): list(args.stopping) for n in n_list}
    else:
        # Per-n defaults per request
        default_map = {
            # Large-N defaults per request
            100000: [5000, 10000, 15000],
            200000: [10000, 15000, 20000],
            300000: [100000, 25000],
        }
        # Validate that all n have a mapping; if not, instruct user to pass --stopping
        missing = [n for n in n_list if int(n) not in default_map]
        if missing:
            raise ValueError(
                f"No default stopping conditions for n={missing}. Provide --stopping to override globally."
            )
        stopping_map = {int(n): list(default_map[int(n)]) for n in n_list}

    # Create the full spec list using per-n stopping conditions
    specs: list[RunSpec] = []
    for n in n_list:
        for v in speeds:
            for sc in stopping_map[int(n)]:
                specs.append(RunSpec(n=int(n), speed_mps=float(v), stopping_condition=int(sc)))

    all_results: list[dict[str, Any]] = []
    total = len(specs)
    for idx, spec in enumerate(specs, start=1):
        result = _run_one(spec, args, out_dir)
        all_results.append(result)
        status = result.get("status", "?")
        print(
            f"[{idx}/{total}] Completed: n={spec.n}, speed={spec.speed_mps}, stopping={spec.stopping_condition} -> {status}"
        )

    # Write pretty Markdown
    summary_md = out_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(_summarize_markdown(all_results))

    print(f"Wrote sweep outputs to: {out_dir}")
    print(f" - Pretty summary: {summary_md}")
    print(f" - Per-run JSONs: {out_dir}/run_*.json")


if __name__ == "__main__":
    main()
