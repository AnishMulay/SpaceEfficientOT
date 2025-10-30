from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .base import SlackKernel, Workspace
from .registry import register_kernel
from ..core.problem import Problem
from ..core.state import SolverState


@dataclass
class _EuclideanSpeedWorkspace:
    # Coordinates in meters (already projected in the experiment)
    xA_m: torch.Tensor  # [N, 2] float32
    xB_m: torch.Tensor  # [M, 2] float32
    xA_T: torch.Tensor  # [2, N] float32
    xA2: torch.Tensor   # [N] float32, squared norms
    scale: torch.Tensor  # scalar float32
    times_A: torch.Tensor  # [N] int64
    times_B: torch.Tensor  # [M] int64
    future_only: bool
    use_speed: bool
    use_ymax: bool
    inv_speed: float
    speed_mps: float
    y_max_m: float
    inf_m: float


def _compute_distances_tile(
    xB_tile: torch.Tensor,
    xA_T: torch.Tensor,
    xA2: torch.Tensor,
) -> torch.Tensor:
    # xB_tile: [K, 2] float32, xA_T: [2, N] float32
    # Dist^2 = ||xb||^2 + ||xa||^2 − 2 xb·xa
    xb2 = (xB_tile.square()).sum(dim=1)  # [K]
    prod = xB_tile @ xA_T  # [K, N]
    d2 = xb2.unsqueeze(1) + xA2.unsqueeze(0) - 2.0 * prod
    d2 = torch.clamp(d2, min=0.0)
    return torch.sqrt(d2)


def _raw_euclidean_speed_slack(
    xB_tile: torch.Tensor,  # [K,2] f32
    xA_T: torch.Tensor,     # [2,N] f32
    xA2: torch.Tensor,      # [N] f32
    yA: torch.Tensor,       # [N] i64
    yB_idx: torch.Tensor,   # [K] i64
    scale: torch.Tensor,    # scalar f32
    tA: torch.Tensor,       # [N] i64
    tB_tile: torch.Tensor,  # [K] i64
    future_only: bool,
    use_speed: bool,
    inv_speed: float,
    use_ymax: bool,
    y_max_m: float,
    inf_m: float,
) -> torch.Tensor:
    # Compute Euclidean distances in meters
    dist_m = _compute_distances_tile(xB_tile, xA_T, xA2)  # [K, N], float32

    # Time differences
    dt = tB_tile.view(-1, 1) - tA.view(1, -1)  # int64 broadcast

    # Masks for infeasible pairs
    if future_only:
        mask_future = dt < 0
    else:
        mask_future = torch.zeros_like(dist_m, dtype=torch.bool)

    if use_speed:
        # Compare in float64 to reduce near-boundary flipping
        dist64 = dist_m.to(dtype=torch.float64)
        time_needed = dist64 * inv_speed  # seconds
        mask_speed = dt.to(dtype=torch.float64) < time_needed
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
        _compiled_euclidean_speed_slack = torch.compile(
            _raw_euclidean_speed_slack, mode="reduce-overhead", dynamic=True
        )
    except Exception:  # pragma: no cover - fallback
        _compiled_euclidean_speed_slack = _raw_euclidean_speed_slack
else:  # pragma: no cover - older PyTorch fallback
    _compiled_euclidean_speed_slack = _raw_euclidean_speed_slack


class EuclideanSpeedKernel(SlackKernel):
    """
    Slack kernel using Euclidean distances in meters with optional temporal
    feasibility and distance caps. Inputs must already be projected to meters.
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
            raise ValueError("EuclideanSpeedKernel expects coordinates [...,2] (meters).")

        # Expect float32 coordinates already in meters
        xA_m = problem.xA.to(dtype=torch.float32)
        xB_m = problem.xB.to(dtype=torch.float32)

        xA_T = xA_m.transpose(0, 1).contiguous()  # [2,N]
        xA2 = (xA_m.square()).sum(dim=1).contiguous()  # [N]

        scale_value = 3.0 / (problem.C_value * problem.delta_value)
        scale = torch.tensor(scale_value, dtype=xA_m.dtype, device=problem.device)

        # Coerce optional times
        def _coerce(value: Any, name: str, length: int) -> torch.Tensor:
            if value is None:
                return torch.zeros(length, dtype=torch.int64, device=problem.device)
            out = torch.as_tensor(value, device=problem.device, dtype=torch.int64).reshape(-1)
            if int(out.numel()) != length:
                raise ValueError(f"{name} must have length {length}, got {int(out.numel())}.")
            return out

        times_A = _coerce(self._times_A, "times_A", problem.n)
        times_B = _coerce(self._times_B, "times_B", problem.m)

        speed_value = 0.0 if self._speed_mps is None else float(self._speed_mps)
        y_max_value = 0.0 if self._y_max_meters is None else float(self._y_max_meters)

        future_only = bool(self._future_only and (self._times_A is not None and self._times_B is not None))
        use_speed = bool(speed_value > 0.0 and (self._times_A is not None and self._times_B is not None))
        use_ymax = bool(y_max_value > 0.0)
        inv_speed = (1.0 / speed_value) if use_speed else 0.0

        # Large sentinel for masked pairs after scaling to int (pre-scale meters)
        inf_m = float(min(1e12, (2**62) / max(scale_value, 1.0)))

        return _EuclideanSpeedWorkspace(
            xA_m=xA_m,
            xB_m=xB_m,
            xA_T=xA_T,
            xA2=xA2,
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
        workspace: _EuclideanSpeedWorkspace,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K_actual = int(idxB.numel())
        if K_actual == 0:
            return out[:0] if out is not None else state.yA.new_empty((0, workspace.xA_m.shape[0]))

        def _compute(idx: torch.Tensor, out_buf: Optional[torch.Tensor] = None) -> torch.Tensor:
            xB_tile = workspace.xB_m.index_select(0, idx)
            yB_idx = state.yB.index_select(0, idx)
            tB_tile = workspace.times_B.index_select(0, idx)
            slack = _compiled_euclidean_speed_slack(
                xB_tile,
                workspace.xA_T,
                workspace.xA2,
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

        # Pad to K_target by repeating last index; compute once and return slice
        pad = K_target - K_actual
        last = idxB[-1]
        idx_pad = torch.cat((idxB, last.repeat(pad)))
        slack_k = _compute(idx_pad, out)
        return slack_k[:K_actual]

    def pair_cost(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        problem: Problem,
        workspace: _EuclideanSpeedWorkspace,
    ) -> torch.Tensor:
        xb = workspace.xB_m.index_select(0, rows).to(dtype=torch.float64)
        xa = workspace.xA_m.index_select(0, cols).to(dtype=torch.float64)
        d2 = ((xb - xa).square()).sum(dim=1)
        return torch.sqrt(torch.clamp(d2, min=0.0))

    def finalize(
        self,
        problem: Problem,
        state: SolverState,
        workspace: _EuclideanSpeedWorkspace,
    ) -> Optional[dict[str, Any]]:
        removed_by_future = 0
        removed_by_speed = 0
        removed_by_ymax = 0

        matched_mask = state.Mb != -1
        if matched_mask.any():
            rows = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
            cols = state.Mb.index_select(0, rows)

            xb = workspace.xB_m.index_select(0, rows).to(dtype=torch.float64)
            xa = workspace.xA_m.index_select(0, cols).to(dtype=torch.float64)
            d2 = ((xb - xa).square()).sum(dim=1)
            dist = torch.sqrt(torch.clamp(d2, min=0.0))

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

            # Speed-based removal disabled to mirror haversine_speed finalize
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


register_kernel("euclidean_speed", EuclideanSpeedKernel)

