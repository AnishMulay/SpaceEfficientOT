#!/usr/bin/env python3
"""Batch job dispatcher — NYC taxi POT comparison experiment.

Each SLURM job calls this script with a single JSON config file.
It dispatches the combined runner, writes result.json + meta.json to a unique
per-run directory, and exits.

No shared CSV is written here. Use aggregate_results.py after all jobs finish.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_DIR.parents[1]
RUN_COMBINED_PY = EXP_DIR / "run_combined.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(obj: Any, final_path: Path) -> None:
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(final_path)


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _slurm_env() -> Dict[str, Any]:
    keys = [
        "SLURM_JOB_ID", "SLURM_JOB_NAME",
        "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_NODELIST", "SLURM_JOB_PARTITION",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def _run_id(cfg: Dict[str, Any]) -> str:
    key_fields = (
        "solver", "input", "date", "n", "seed",
        "speed_mps", "y_max_meters", "future_only",
        "delta", "C", "k", "stopping_condition",
        "fill_policy", "device",
    )
    sel = {k: cfg.get(k) for k in key_fields}
    payload = json.dumps(sel, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()  # nosec — non-crypto id


def _run_dir_name(cfg: Dict[str, Any]) -> str:
    solver = cfg.get("solver", "unknown")
    n = cfg.get("n", "?")
    seed = cfg.get("seed", "?")
    return f"{solver}_n{n}_seed{seed}"


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _build_combined_cmd(cfg: Dict[str, Any], out_path: Path) -> list[str]:
    cmd = [sys.executable, str(RUN_COMBINED_PY)]

    def add(flag: str, key: str) -> None:
        val = cfg.get(key)
        if val is not None:
            cmd.extend([flag, str(val)])

    add("--input", "input")
    add("--date", "date")
    add("--n", "n")
    add("--seed", "seed")
    add("--device", "device")
    add("--k", "k")
    add("--delta", "delta")
    add("--stopping-condition", "stopping_condition")
    add("--C", "C")
    add("--speed-mps", "speed_mps")
    add("--y-max-meters", "y_max_meters")
    add("--fill-policy", "fill_policy")
    add("--origin-lon", "origin_lon")
    add("--origin-lat", "origin_lat")

    if cfg.get("random_sample") is True:
        cmd.append("--random-sample")
    elif cfg.get("random_sample") is False:
        cmd.append("--no-random-sample")

    if cfg.get("future_only") is True:
        cmd.append("--future-only")
    elif cfg.get("future_only") is False:
        cmd.append("--no-future-only")

    cmd.extend(["--out", str(out_path)])
    return cmd


def _extract_duration_sec(data: Dict[str, Any]) -> float | None:
    perf = data.get("performance")
    if isinstance(perf, dict):
        runtime = perf.get("runtime_sec")
        if runtime is not None:
            return float(runtime)

    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_once(
    config_path: Path,
    results_dir: Path,
    timeout_sec: int,
    overwrite: bool,
    print_cmd: bool,
) -> Tuple[int, Path]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)

    solver = cfg.get("solver", "")
    if solver != "combined":
        print(f"ERROR: unknown solver {solver!r}", file=sys.stderr)
        return 1, results_dir / "result.json"

    run_name = _run_dir_name(cfg)
    run_id = _run_id(cfg)
    date_dir = str(cfg.get("date") or "unknown_date")

    out_dir = results_dir / date_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = out_dir / "result.json"
    artifact_tmp = out_dir / "result.json.tmp"
    config_used = out_dir / "config_used.json"
    meta_path = out_dir / "meta.json"

    # Skip if already succeeded and not overwriting
    if artifact.exists() and not overwrite:
        try:
            _read_json(artifact)
            print(f"Skipping (already done): {artifact}")
            return 0, artifact
        except Exception:
            pass

    # Build command
    cmd = _build_combined_cmd(cfg, artifact_tmp)

    if print_cmd:
        print("CMD:", " ".join(cmd), flush=True)

    meta: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "solver": solver,
        "host": socket.gethostname(),
        "slurm": _slurm_env(),
        "config_path": str(config_path),
        "start": _now_iso(),
        "timeout_sec": timeout_sec,
        "status": "started",
        "cmd": cmd,
    }
    _write_json_atomic(cfg, config_used)
    _write_json_atomic(meta, meta_path)

    # Run subprocess
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as ex:
        meta.update({
            "status": "timeout",
            "end": _now_iso(),
            "stdout_tail": ex.stdout[-4000:] if ex.stdout else None,
            "stderr_tail": ex.stderr[-4000:] if ex.stderr else None,
        })
        _write_json_atomic(meta, meta_path)
        artifact_tmp.unlink(missing_ok=True)
        return 124, artifact
    except Exception as ex:
        meta.update({"status": "failed", "end": _now_iso(), "error": f"{type(ex).__name__}: {ex}"})
        _write_json_atomic(meta, meta_path)
        artifact_tmp.unlink(missing_ok=True)
        return 1, artifact

    if proc.returncode != 0:
        meta.update({
            "status": "failed",
            "end": _now_iso(),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        })
        _write_json_atomic(meta, meta_path)
        artifact_tmp.unlink(missing_ok=True)
        return proc.returncode, artifact

    # Validate and promote artifact
    try:
        _read_json(artifact_tmp)
        artifact_tmp.replace(artifact)
    except Exception as ex:
        meta.update({
            "status": "failed",
            "end": _now_iso(),
            "error": f"Invalid result JSON: {type(ex).__name__}: {ex}",
        })
        _write_json_atomic(meta, meta_path)
        artifact_tmp.unlink(missing_ok=True)
        return 1, artifact

    duration = None
    try:
        data = _read_json(artifact)
        duration = _extract_duration_sec(data)
    except Exception:
        pass

    meta.update({"status": "success", "end": _now_iso(), "duration_sec": duration, "returncode": 0})
    _write_json_atomic(meta, meta_path)
    return 0, artifact


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one NYC taxi POT-comparison job from a JSON config")
    ap.add_argument("--config", required=True, help="Path to JSON config file")
    ap.add_argument("--results-dir", default=str(EXP_DIR / "batch" / "results"),
                    help="Root directory for per-run outputs (default: batch/results)")
    ap.add_argument("--timeout-sec", type=int, default=7200,
                    help="Max seconds for the experiment (default: 7200 = 2h)")
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if result.json exists")
    ap.add_argument("--print-cmd", action="store_true", help="Print the exact command being run")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    results_dir = Path(args.results_dir).resolve()

    code, artifact = run_once(cfg_path, results_dir, args.timeout_sec, args.overwrite, args.print_cmd)

    if code == 0:
        print(f"Success: {artifact}", flush=True)
    elif code == 124:
        print("Timeout: exceeded max runtime; see meta.json", file=sys.stderr, flush=True)
    else:
        print(f"Failed (code={code}): see meta.json", file=sys.stderr, flush=True)

    sys.exit(code)


if __name__ == "__main__":
    main()
