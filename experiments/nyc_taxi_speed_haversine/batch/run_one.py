#!/usr/bin/env python3
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
RUN_PY = EXP_DIR / "run.py"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(obj: Any, final_path: Path) -> None:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp_path.replace(final_path)


def _fmt_float_compact(x: float) -> str:
    # Compact, stable float formatting for names (avoids long trailing zeros)
    s = format(float(x), ".6g")
    return s.replace("+", "").lower()


def _run_name_from_config(cfg: Dict[str, Any]) -> str:
    n = cfg.get("n")
    sc = cfg.get("stopping_condition")
    speed = cfg.get("speed_mps")
    delta = cfg.get("delta")
    y_max = cfg.get("y_max_meters")
    parts = []
    if n is not None:
        parts.append(f"n{int(n)}")
    if sc is not None:
        parts.append(f"sc{int(sc)}")
    if speed is not None:
        parts.append(f"v{_fmt_float_compact(float(speed))}")
    if delta is not None:
        parts.append(f"d{_fmt_float_compact(float(delta))}")
    if y_max is not None:
        parts.append(f"y{_fmt_float_compact(float(y_max))}")
    if not parts:
        return "run"
    return "_".join(parts)


def _run_id(cfg: Dict[str, Any]) -> str:
    # Stable ID based on a subset of parameters that affect the run artifact
    key_fields = (
        "input",
        "date",
        "n",
        "delta",
        "stopping_condition",
        "C",
        "k",
        "speed_mps",
        "y_max_meters",
        "future_only",
        "fill_policy",
        "device",
        "seed",
        "preview_count",
    )
    sel = {k: cfg.get(k) for k in key_fields}
    payload = json.dumps(sel, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()  # nosec - non-crypto id


def _build_cmd(cfg: Dict[str, Any], out_path: Path, force_no_warmup: bool = True) -> list[str]:
    cmd: list[str] = [sys.executable, str(RUN_PY)]

    def add(flag: str, value: Any) -> None:
        nonlocal cmd
        if value is None:
            return
        cmd += [flag, str(value)]

    # Scalar flags mapping 1:1 with run.py
    add("--input", cfg.get("input"))
    add("--date", cfg.get("date"))
    add("--n", cfg.get("n"))
    add("--seed", cfg.get("seed"))
    add("--device", cfg.get("device"))
    add("--k", cfg.get("k"))
    add("--delta", cfg.get("delta"))
    add("--stopping-condition", cfg.get("stopping_condition"))
    add("--c-sample", cfg.get("c_sample"))
    add("--C", cfg.get("C"))
    add("--speed-mps", cfg.get("speed_mps"))
    add("--y-max-meters", cfg.get("y_max_meters"))
    add("--fill-policy", cfg.get("fill_policy"))
    add("--preview-count", cfg.get("preview_count"))

    # Booleans with explicit positive/negative flags
    if cfg.get("random_sample") is True:
        cmd += ["--random-sample"]
    elif cfg.get("random_sample") is False:
        cmd += ["--no-random-sample"]

    if cfg.get("future_only") is True:
        cmd += ["--future-only"]
    elif cfg.get("future_only") is False:
        cmd += ["--no-future-only"]

    # Always disable warm-up to save time on HPC
    if force_no_warmup:
        cmd += ["--no-warmup"]

    # Artifact path
    cmd += ["--out", str(out_path)]
    return cmd


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _slurm_env() -> Dict[str, Any]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_ARRAY_TASK_MAX",
        "SLURM_ARRAY_TASK_MIN",
        "SLURM_SUBMIT_DIR",
        "SLURM_NODELIST",
        "SLURM_JOB_PARTITION",
    ]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def _parse_config(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    if isinstance(data, dict) and "params" in data and isinstance(data["params"], dict):
        return data["params"]
    if isinstance(data, dict):
        return data
    raise ValueError("Config JSON must be an object or contain a 'params' object")


def run_once(config_path: Path, results_dir: Path, timeout_sec: int, overwrite: bool, print_cmd: bool) -> Tuple[int, Path]:
    cfg = _parse_config(config_path)

    # Defaults useful for quick testing
    cfg.setdefault("delta", 0.0001)
    cfg.setdefault("preview_count", 0)

    run_name = _run_name_from_config(cfg)
    run_id = _run_id(cfg)

    out_dir = results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = out_dir / "result.json"
    artifact_tmp = out_dir / "result.json.tmp"
    config_used = out_dir / "config_used.json"
    meta_path = out_dir / "meta.json"

    # Idempotency: if a valid result already exists, return early unless overwrite
    if artifact.exists() and not overwrite:
        try:
            _read_json(artifact)  # validate JSON
            return 0, artifact
        except Exception:
            # Corrupt or partial file; proceed to re-run
            pass

    cmd = _build_cmd(cfg, artifact_tmp, force_no_warmup=True)
    if print_cmd:
        print("CMD:", " ".join(cmd))

    meta: Dict[str, Any] = {
        "run_name": run_name,
        "run_id": run_id,
        "host": socket.gethostname(),
        "slurm": _slurm_env(),
        "config_path": str(config_path),
        "start": _now_iso(),
        "timeout_sec": int(timeout_sec),
        "status": "started",
        "cmd": cmd,
    }

    # Persist the exact config used
    _write_json_atomic(cfg, config_used)
    _write_json_atomic(meta, meta_path)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_sec),
        )
    except subprocess.TimeoutExpired as ex:
        meta.update(
            {
                "status": "timeout",
                "end": _now_iso(),
                "duration_sec": None,
                "stdout_tail": ex.stdout[-4000:] if ex.stdout else None,
                "stderr_tail": ex.stderr[-4000:] if ex.stderr else None,
            }
        )
        _write_json_atomic(meta, meta_path)
        if artifact_tmp.exists():
            try:
                artifact_tmp.unlink()
            except Exception:
                pass
        return 124, artifact
    except Exception as ex:
        meta.update(
            {
                "status": "failed",
                "end": _now_iso(),
                "error": f"{type(ex).__name__}: {ex}",
            }
        )
        _write_json_atomic(meta, meta_path)
        if artifact_tmp.exists():
            try:
                artifact_tmp.unlink()
            except Exception:
                pass
        return 1, artifact

    # Non-zero return code without timeout
    if proc.returncode != 0:
        meta.update(
            {
                "status": "failed",
                "end": _now_iso(),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:] if proc.stdout else None,
                "stderr_tail": proc.stderr[-4000:] if proc.stderr else None,
            }
        )
        _write_json_atomic(meta, meta_path)
        if artifact_tmp.exists():
            try:
                artifact_tmp.unlink()
            except Exception:
                pass
        return proc.returncode, artifact

    # Success: move tmp → final and finalize meta
    try:
        _ = _read_json(artifact_tmp)  # sanity check JSON before rename
        artifact_tmp.replace(artifact)
    except Exception as ex:
        meta.update(
            {
                "status": "failed",
                "end": _now_iso(),
                "error": f"Invalid or unreadable JSON artifact: {type(ex).__name__}: {ex}",
            }
        )
        _write_json_atomic(meta, meta_path)
        if artifact_tmp.exists():
            try:
                artifact_tmp.unlink()
            except Exception:
                pass
        return 1, artifact

    # Measure duration if run.py reports it in artifact
    duration = None
    try:
        data = _read_json(artifact)
        perf = data.get("performance", {}) if isinstance(data, dict) else {}
        duration = perf.get("runtime_sec")
    except Exception:
        pass

    meta.update(
        {
            "status": "success",
            "end": _now_iso(),
            "duration_sec": duration,
            "returncode": 0,
        }
    )
    _write_json_atomic(meta, meta_path)
    return 0, artifact


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a single NYC Haversine Speed experiment from JSON config")
    ap.add_argument("--config", required=True, help="Path to JSON config file")
    ap.add_argument(
        "--results-dir",
        default=str(EXP_DIR / "batch" / "results" / "nyc_haversine"),
        help="Directory where per-run outputs are written",
    )
    ap.add_argument("--timeout-sec", type=int, default=3600, help="Max seconds to allow for the experiment (default: 3600)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing successful result.json")
    ap.add_argument("--print-cmd", action="store_true", help="Print the exact python command executed")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    results_dir = Path(args.results_dir).resolve()

    code, artifact = run_once(cfg_path, results_dir, args.timeout_sec, args.overwrite, args.print_cmd)
    if code == 0:
        print(f"Success: wrote {artifact}")
    elif code == 124:
        print("Timeout: exceeded max runtime; see meta.json for details")
    else:
        print("Failed: see meta.json for details")
    sys.exit(code)


if __name__ == "__main__":
    main()
