from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .base import SlackKernel, Workspace
from .registry import register_kernel
from ..core.problem import Problem
from ..core.state import SolverState

_EARTH_RADIUS_KM = 6_371.0
_MASK_SENTINEL_INT = 10**12


def _haversine_distance(xb_rad: torch.Tensor, xA_rad: torch.Tensor) -> torch.Tensor:
    lon_b = xb_rad[:, 0].unsqueeze(1)
    lat_b = xb_rad[:, 1].unsqueeze(1)
    lon_a = xA_rad[:, 0].unsqueeze(0)
    lat_a = xA_rad[:, 1].unsqueeze(0)

    dlon = lon_a - lon_b
    dlat = lat_a - lat_b

    sin_dlat = torch.sin(dlat * 0.5)
    sin_dlon = torch.sin(dlon * 0.5)
    cos_lat_b = torch.cos(lat_b)
    cos_lat_a = torch.cos(lat_a)

    a = sin_dlat.square() + cos_lat_b * cos_lat_a * sin_dlon.square()
    a = torch.clamp(a, min=0.0, max=1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=0.0)))
    return (_EARTH_RADIUS_KM * c).to(xb_rad.dtype)


def _raw_haversine_slack(
    xb_deg: torch.Tensor,
    xA_rad: torch.Tensor,
    yA: torch.Tensor,
    yB_idx: torch.Tensor,
    scale_factor: torch.Tensor,
    tA: torch.Tensor,
    tB_tile: torch.Tensor,
    apply_mask: bool,
    cmax_int: int,
) -> torch.Tensor:
    xb_rad = torch.deg2rad(xb_deg)
    dist = _haversine_distance(xb_rad, xA_rad)
    scaled = torch.floor(dist * scale_factor).to(torch.int64)

    sentinel = scaled.new_full((), _MASK_SENTINEL_INT)
    mask = None

    if apply_mask:
        mask = tB_tile.view(-1, 1) < tA.view(1, -1)
        scaled = torch.where(mask, sentinel, scaled)

    if cmax_int >= 0:
        clamp_val = scaled.new_full((), int(cmax_int))
        if apply_mask and mask is not None:
            scaled = torch.where(mask, sentinel, torch.minimum(scaled, clamp_val))
        else:
            scaled = torch.minimum(scaled, clamp_val)

    return scaled - yA.unsqueeze(0) - yB_idx.unsqueeze(1)


if hasattr(torch, "compile"):
    try:
        _compiled_haversine_slack = torch.compile(
            _raw_haversine_slack, mode="reduce-overhead", dynamic=True
        )
    except Exception:  # pragma: no cover - compilation fallback
        _compiled_haversine_slack = _raw_haversine_slack
else:  # pragma: no cover - older PyTorch fallback
    _compiled_haversine_slack = _raw_haversine_slack


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


@dataclass
class _HaversineWorkspace:
    xA_deg: torch.Tensor
    xB_deg: torch.Tensor
    xA_rad: torch.Tensor
    scale: torch.Tensor
    times_A: Optional[torch.Tensor]
    times_B: Optional[torch.Tensor]
    apply_mask: bool
    cmax_int: int


class HaversineKernel(SlackKernel):
    """
    Slack kernel for Haversine distances with optional time-window masking and capping.
    """

    def __init__(
        self,
        *,
        times_A: Any = None,
        times_B: Any = None,
        cmax_int: Optional[int] = None,
    ) -> None:
        super().__init__(times_A=times_A, times_B=times_B, cmax_int=cmax_int)
        self._times_A = times_A
        self._times_B = times_B
        self._cmax_int = -1 if cmax_int is None else int(cmax_int)

    def prepare(self, problem: Problem) -> Workspace:
        if problem.xA.shape[-1] != 2 or problem.xB.shape[-1] != 2:
            raise ValueError("HaversineKernel expects coordinates with shape [..., 2] (lon, lat).")

        xA = problem.xA.to(dtype=torch.float32)
        xB = problem.xB.to(dtype=torch.float32)

        xA_rad = torch.deg2rad(xA)
        scale_value = 3.0 / (problem.C_value * problem.delta_value)
        scale = torch.tensor(scale_value, dtype=xA.dtype, device=problem.device)

        times_A = _coerce_optional_tensor(
            self._times_A,
            name="times_A",
            expected_length=problem.n,
            device=problem.device,
            dtype=torch.int64,
        )
        times_B = _coerce_optional_tensor(
            self._times_B,
            name="times_B",
            expected_length=problem.m,
            device=problem.device,
            dtype=torch.int64,
        )
        apply_mask = times_A is not None and times_B is not None

        return _HaversineWorkspace(
            xA_deg=xA,
            xB_deg=xB,
            xA_rad=xA_rad,
            scale=scale,
            times_A=times_A,
            times_B=times_B,
            apply_mask=apply_mask,
            cmax_int=self._cmax_int,
        )

    def compute_slack_tile(
        self,
        idxB: torch.Tensor,
        state: SolverState,
        workspace: _HaversineWorkspace,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_k = idxB.numel()
        if current_k == 0:
            if out is not None:
                return out[:0]
            return state.yA.new_empty((0, workspace.xA_deg.shape[0]))

        xb = workspace.xB_deg.index_select(0, idxB).to(dtype=workspace.xA_deg.dtype)
        yB_idx = state.yB.index_select(0, idxB)

        if workspace.apply_mask:
            tB_tile = workspace.times_B.index_select(0, idxB)
            tA = workspace.times_A
        else:
            tB_tile = yB_idx.new_zeros(current_k, dtype=torch.int64)
            tA = state.yA.new_zeros(workspace.xA_deg.shape[0], dtype=torch.int64)

        slack = _compiled_haversine_slack(
            xb,
            workspace.xA_rad,
            state.yA,
            yB_idx,
            workspace.scale,
            tA,
            tB_tile,
            workspace.apply_mask,
            workspace.cmax_int,
        )

        if out is not None and out.shape[0] >= slack.shape[0]:
            out_view = out[: slack.shape[0]]
            out_view.copy_(slack)
            return out_view

        return slack

    def pair_cost(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        problem: Problem,
        workspace: _HaversineWorkspace,
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
        workspace: _HaversineWorkspace,
    ) -> Optional[dict[str, Any]]:
        if workspace.cmax_int >= 0:
            matched_mask = state.Mb != -1
            if matched_mask.any():
                rows = torch.nonzero(matched_mask, as_tuple=False).view(-1)
                cols = state.Mb.index_select(0, rows)
                xb = workspace.xB_deg.index_select(0, rows).to(dtype=torch.float64)
                xa = workspace.xA_deg.index_select(0, cols).to(dtype=torch.float64)
                xb_rad = torch.deg2rad(xb)
                xa_rad = torch.deg2rad(xa)
                dist = _haversine_distance(xb_rad, xa_rad).to(torch.float64)
                scale = workspace.scale.to(dtype=torch.float64)
                int_costs = torch.floor(dist * scale).to(torch.int64)
                infeasible = int_costs >= workspace.cmax_int
                if infeasible.any():
                    bad_idx = torch.nonzero(infeasible, as_tuple=False).view(-1)
                    bad_rows = rows.index_select(0, bad_idx)
                    bad_cols = cols.index_select(0, bad_idx)
                    state.Mb[bad_rows] = -1
                    state.Ma[bad_cols] = -1

        feasible_matches = int((state.Mb != -1).sum().item())
        free_B = problem.m - feasible_matches

        return {
            "feasible_matches": float(feasible_matches),
            "free_B": float(free_B),
        }


register_kernel("haversine", HaversineKernel)
