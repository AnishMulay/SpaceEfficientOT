from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .base import SlackKernel, Workspace
from .haversine import _haversine_distance
from .registry import register_kernel
from ..core.problem import Problem
from ..core.state import SolverState


def _coerce_optional_tensor(
    value: Any,
    *,
    name: str,
    expected_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1).contiguous()
    if tensor.numel() != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {tensor.numel()}.")
    return tensor


def _raw_haversine_speed_slack(
    xb_deg: torch.Tensor,
    xA_rad: torch.Tensor,
    yA: torch.Tensor,
    yB_idx: torch.Tensor,
    scale: torch.Tensor,
    tA: torch.Tensor,
    tB_tile: torch.Tensor,
    future_only: bool,
    use_speed: bool,
    inv_speed: float,
    use_ymax: bool,
    y_max_m: float,
    inf_m: float,
) -> torch.Tensor:
    xb_rad = torch.deg2rad(xb_deg)
    dist_m = _haversine_distance(xb_rad, xA_rad).to(xb_deg.dtype)

    dt = tB_tile.view(-1, 1) - tA.view(1, -1)

    if future_only:
        mask_future = dt < 0
    else:
        mask_future = torch.zeros_like(dist_m, dtype=torch.bool)

    if use_speed:
        time_limit = dist_m * inv_speed
        mask_speed = dt.to(dist_m.dtype) < time_limit
    else:
        mask_speed = torch.zeros_like(dist_m, dtype=torch.bool)

    mask = mask_future | mask_speed
    inf = dist_m.new_full((), inf_m)
    cost_m = torch.where(mask, inf, dist_m)

    if use_ymax:
        y_max = dist_m.new_full((), y_max_m)
        cost_m = torch.where(mask, cost_m, torch.minimum(cost_m, y_max))

    scaled = torch.floor(cost_m * scale).to(torch.int64)
    return scaled - yA.unsqueeze(0) - yB_idx.unsqueeze(1)


if hasattr(torch, "compile"):
    try:
        _compiled_haversine_speed_slack = torch.compile(
            _raw_haversine_speed_slack, mode="reduce-overhead", dynamic=True
        )
    except Exception:  # pragma: no cover - compilation fallback
        _compiled_haversine_speed_slack = _raw_haversine_speed_slack
else:  # pragma: no cover - older PyTorch fallback
    _compiled_haversine_speed_slack = _raw_haversine_speed_slack


@dataclass
class _HaversineSpeedWorkspace:
    xA_deg: torch.Tensor
    xB_deg: torch.Tensor
    xA_rad: torch.Tensor
    scale: torch.Tensor
    times_A: torch.Tensor
    times_B: torch.Tensor
    future_only: bool
    use_speed: bool
    use_ymax: bool
    inv_speed: float
    speed_mps: float
    y_max_m: float
    inf_m: float


class HaversineSpeedKernel(SlackKernel):
    """
    Slack kernel that combines Haversine distance with optional temporal feasibility
    and distance caps to produce integer slack values.
    """

    def __init__(
        self,
        *,
        times_A: Any = None,
        times_B: Any = None,
        speed_mps: Optional[float] = None,
        y_max_meters: Optional[float] = None,
        future_only: bool = True,
    ) -> None:
        super().__init__(
            times_A=times_A,
            times_B=times_B,
            speed_mps=speed_mps,
            y_max_meters=y_max_meters,
            future_only=future_only,
        )
        self._times_A = times_A
        self._times_B = times_B
        self._speed_mps = speed_mps
        self._y_max_meters = y_max_meters
        self._future_only = bool(future_only)

    def prepare(self, problem: Problem) -> Workspace:
        if problem.xA.shape[-1] != 2 or problem.xB.shape[-1] != 2:
            raise ValueError("HaversineSpeedKernel expects coordinates with shape [..., 2] (lon, lat).")

        xA = problem.xA.to(dtype=torch.float32)
        xB = problem.xB.to(dtype=torch.float32)

        xA_rad = torch.deg2rad(xA)
        scale_value = 3.0 / (problem.C_value * problem.delta_value)
        scale = torch.tensor(scale_value, dtype=xA.dtype, device=problem.device)

        times_A_opt = _coerce_optional_tensor(
            self._times_A,
            name="times_A",
            expected_length=problem.n,
            device=problem.device,
            dtype=torch.int64,
        )
        times_B_opt = _coerce_optional_tensor(
            self._times_B,
            name="times_B",
            expected_length=problem.m,
            device=problem.device,
            dtype=torch.int64,
        )

        speed_value = 0.0 if self._speed_mps is None else float(self._speed_mps)
        y_max_value = 0.0 if self._y_max_meters is None else float(self._y_max_meters)

        if speed_value > 0.0 and (times_A_opt is None or times_B_opt is None):
            raise ValueError("speed_mps requires both times_A and times_B to be provided.")

        future_only = bool(self._future_only and times_A_opt is not None and times_B_opt is not None)
        use_speed = bool(speed_value > 0.0 and times_A_opt is not None and times_B_opt is not None)
        use_ymax = bool(y_max_value > 0.0)
        inv_speed = (1.0 / speed_value) if use_speed else 0.0

        times_A = (
            times_A_opt
            if times_A_opt is not None
            else torch.zeros(problem.n, dtype=torch.int64, device=problem.device)
        )
        times_B = (
            times_B_opt
            if times_B_opt is not None
            else torch.zeros(problem.m, dtype=torch.int64, device=problem.device)
        )

        inf_m = float(min(1e12, (2**62) / max(scale_value, 1.0)))

        return _HaversineSpeedWorkspace(
            xA_deg=xA,
            xB_deg=xB,
            xA_rad=xA_rad,
            scale=scale,
            times_A=times_A,
            times_B=times_B,
            future_only=future_only,
            use_speed=use_speed,
            use_ymax=use_ymax,
            inv_speed=inv_speed,
            speed_mps=speed_value,
            y_max_m=y_max_value,
            inf_m=inf_m,
        )

    def compute_slack_tile(
        self,
        idxB: torch.Tensor,
        state: SolverState,
        workspace: _HaversineSpeedWorkspace,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K_actual = int(idxB.numel())
        if K_actual == 0:
            return out[:0] if out is not None else state.yA.new_empty((0, workspace.xA_deg.shape[0]))

        def _compute(idx: torch.Tensor, out_buf: Optional[torch.Tensor] = None) -> torch.Tensor:
            xb = workspace.xB_deg.index_select(0, idx).to(dtype=workspace.xA_deg.dtype)
            yB_idx = state.yB.index_select(0, idx)
            tB_tile = workspace.times_B.index_select(0, idx)
            slack = _compiled_haversine_speed_slack(
                xb,
                workspace.xA_rad,
                state.yA,
                yB_idx,
                workspace.scale,
                workspace.times_A,
                tB_tile,
                workspace.future_only,
                workspace.use_speed,
                workspace.inv_speed,
                workspace.use_ymax,
                workspace.y_max_m,
                workspace.inf_m,
            )
            if out_buf is not None:
                out_buf.copy_(slack)
                return out_buf
            return slack

        if out is None:
            return _compute(idxB)

        K_target = int(out.shape[0])
        if K_target == K_actual:
            return _compute(idxB, out[:K_actual])

        pad_count = K_target - K_actual
        last_idx = idxB[-1]
        idxB_padded = torch.cat((idxB, last_idx.repeat(pad_count)))
        slack_k = _compute(idxB_padded, out)
        return slack_k[:K_actual]

    def pair_cost(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        problem: Problem,
        workspace: _HaversineSpeedWorkspace,
    ) -> torch.Tensor:
        xb = workspace.xB_deg.index_select(0, rows).to(dtype=torch.float64)
        xa = workspace.xA_deg.index_select(0, cols).to(dtype=torch.float64)
        xb_rad = torch.deg2rad(xb)
        xa_rad = torch.deg2rad(xa)
        return _haversine_distance(xb_rad, xa_rad).to(torch.float64)

    def finalize(
        self,
        problem: Problem,
        state: SolverState,
        workspace: _HaversineSpeedWorkspace,
    ) -> Optional[dict[str, Any]]:
        removed_by_future = 0
        removed_by_speed = 0
        removed_by_ymax = 0

        matched_mask = state.Mb != -1
        if matched_mask.any():
            rows = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
            cols = state.Mb.index_select(0, rows)

            xb = workspace.xB_deg.index_select(0, rows).to(dtype=torch.float64)
            xa = workspace.xA_deg.index_select(0, cols).to(dtype=torch.float64)
            xb_rad = torch.deg2rad(xb)
            xa_rad = torch.deg2rad(xa)
            dist = _haversine_distance(xb_rad, xa_rad).to(torch.float64)

            violation_mask = torch.zeros_like(dist, dtype=torch.bool)

            if workspace.future_only or workspace.use_speed:
                tB = workspace.times_B.index_select(0, rows).to(dtype=torch.int64)
                tA = workspace.times_A.index_select(0, cols).to(dtype=torch.int64)
                dt = tB - tA
            else:
                dt = None

            if workspace.future_only and dt is not None:
                mask_future = dt < 0
                violation_mask |= mask_future
                removed_by_future = int(mask_future.sum().item())
            else:
                mask_future = torch.zeros_like(dist, dtype=torch.bool)

            if workspace.use_speed and dt is not None:
                time_limit = dist / workspace.speed_mps
                mask_speed = dt.to(dist.dtype) < time_limit
                violation_mask |= mask_speed
                removed_by_speed = int(mask_speed.sum().item())
            else:
                mask_speed = torch.zeros_like(dist, dtype=torch.bool)

            if workspace.use_ymax:
                mask_ymax = dist >= workspace.y_max_m
                violation_mask |= mask_ymax
                removed_by_ymax = int(mask_ymax.sum().item())
            else:
                mask_ymax = torch.zeros_like(dist, dtype=torch.bool)

            if violation_mask.any():
                bad_idx = torch.nonzero(violation_mask, as_tuple=False).reshape(-1)
                bad_rows = rows.index_select(0, bad_idx)
                bad_cols = cols.index_select(0, bad_idx)
                state.Mb[bad_rows] = -1
                state.Ma[bad_cols] = -1

        feasible_matches = int((state.Mb != -1).sum().item())
        free_B = problem.m - feasible_matches

        return {
            "feasible_matches": float(feasible_matches),
            "free_B": float(free_B),
            "removed_by_future": float(removed_by_future),
            "removed_by_speed": float(removed_by_speed),
            "removed_by_ymax": float(removed_by_ymax),
        }


register_kernel("haversine_speed", HaversineSpeedKernel)
