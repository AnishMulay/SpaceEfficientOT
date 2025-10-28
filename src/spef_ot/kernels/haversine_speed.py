from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .base import SlackKernel, Workspace
from .haversine import _haversine_distance
from .haversine import _EARTH_RADIUS_METERS
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
    EB_tile: torch.Tensor,
    EA_T: torch.Tensor,
    radius_m: float,
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
    # Pairwise great-circle distance via unit-sphere dot product
    # cos(theta) = EB · EA; distances = R * arccos(cos(theta))
    cos_angles = EB_tile @ EA_T  # [K, N]
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    dist_m = (torch.acos(cos_angles) * float(radius_m)).to(EB_tile.dtype)

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
    EA_T: torch.Tensor
    EB: torch.Tensor
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
    radius_m: float


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

        # Precompute unit-sphere embeddings for lon/lat in degrees
        def _embed_lonlat_deg(coords_deg: torch.Tensor) -> torch.Tensor:
            lon_rad = torch.deg2rad(coords_deg[:, 0])
            lat_rad = torch.deg2rad(coords_deg[:, 1])
            cos_lat = torch.cos(lat_rad)
            sin_lat = torch.sin(lat_rad)
            x = cos_lat * torch.cos(lon_rad)
            y = cos_lat * torch.sin(lon_rad)
            z = sin_lat
            return torch.stack((x, y, z), dim=1)

        EA = _embed_lonlat_deg(xA).contiguous()
        EB = _embed_lonlat_deg(xB).contiguous()
        EA_T = EA.transpose(0, 1).contiguous()
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
            EA_T=EA_T,
            EB=EB,
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
            radius_m=float(_EARTH_RADIUS_METERS),
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
            EB_tile = workspace.EB.index_select(0, idx)
            yB_idx = state.yB.index_select(0, idx)
            tB_tile = workspace.times_B.index_select(0, idx)
            slack = _compiled_haversine_speed_slack(
                EB_tile,
                workspace.EA_T,
                workspace.radius_m,
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
        EB_rows = workspace.EB.index_select(0, rows).to(dtype=torch.float64)
        EA_cols = workspace.EA_T.transpose(0, 1).index_select(0, cols).to(dtype=torch.float64)
        cos_vals = (EB_rows * EA_cols).sum(dim=1)
        cos_vals = torch.clamp(cos_vals, -1.0, 1.0)
        return torch.acos(cos_vals) * float(workspace.radius_m)

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

            # Compute great-circle distances using the same GEMM-based approach
            EB_rows = workspace.EB.index_select(0, rows).to(dtype=torch.float64)
            EA_cols = workspace.EA_T.transpose(0, 1).index_select(0, cols).to(dtype=torch.float64)
            cos_vals = (EB_rows * EA_cols).sum(dim=1)
            cos_vals = torch.clamp(cos_vals, -1.0, 1.0)
            dist = torch.acos(cos_vals) * float(workspace.radius_m)

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

            # Speed-based removal disabled per request; keep metric at 0
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

    # Optional per-iteration diagnostics hook used by the solver when available
    def iteration_diagnostics(
        self,
        problem: Problem,
        state: SolverState,
        workspace: _HaversineSpeedWorkspace,
        *,
        sentinel_count: int = 6,
        near_epsilon_sec: float = 1.0,
    ) -> Optional[dict[str, Any]]:
        try:
            # Only meaningful if speed constraint is active
            if not workspace.use_speed or workspace.speed_mps <= 0.0:
                return None

            device = problem.device
            generator = state.generator

            free_mask = state.Mb == -1
            if not bool(free_mask.any()):
                return None
            free_idx = torch.nonzero(free_mask, as_tuple=False).squeeze(1)
            S = min(int(sentinel_count), int(free_idx.numel()))
            if S <= 0:
                return None

            if S == int(free_idx.numel()):
                samp = free_idx
            else:
                perm = torch.randperm(int(free_idx.numel()), device=device, generator=generator)
                samp = free_idx.index_select(0, perm[:S])

            # Distances vs all A in fp32 and fp64 via dot+acos
            EB_rows_f32 = workspace.EB.index_select(0, samp).to(dtype=torch.float32)
            EA_T_f32 = workspace.EA_T.to(dtype=torch.float32)
            cos32 = EB_rows_f32 @ EA_T_f32
            cos32 = torch.clamp(cos32, -1.0, 1.0)
            dist32 = torch.acos(cos32) * float(workspace.radius_m)  # [S, N]

            EB_rows_f64 = EB_rows_f32.to(dtype=torch.float64)
            EA_T_f64 = EA_T_f32.to(dtype=torch.float64)
            cos64 = EB_rows_f64 @ EA_T_f64
            cos64 = torch.clamp(cos64, -1.0, 1.0)
            dist64 = torch.acos(cos64) * float(workspace.radius_m)  # [S, N]

            tA = workspace.times_A.to(dtype=torch.int64)
            tB_sel = workspace.times_B.index_select(0, samp).to(dtype=torch.int64)
            speed = float(workspace.speed_mps)
            N = int(tA.numel())
            eps = float(near_epsilon_sec)

            sentinels: list[dict[str, Any]] = []

            for row in range(S):
                b_idx = int(samp[row].item())
                tB_val = int(tB_sel[row].item())

                # dt over columns [N]
                dt = (tB_sel[row].view(1, 1) - tA.view(1, -1)).squeeze(0)
                dt_nonneg = dt >= 0

                d32 = dist32[row, :]
                d64 = dist64[row, :]

                need32 = (d32 / speed).to(dtype=torch.float32)
                need64 = (d64 / speed).to(dtype=torch.float64)
                valid_k32 = dt_nonneg & (dt.to(dtype=need32.dtype) >= need32)
                valid_o64 = dt_nonneg & (dt.to(dtype=need64.dtype) >= need64)

                eligible = int(dt_nonneg.sum().item())
                future_invalid = N - eligible
                kernel_allowed = int(valid_k32.sum().item())
                oracle_valid = int(valid_o64.sum().item())

                if eligible > 0:
                    margin32 = dt[dt_nonneg].to(dtype=torch.float32) - need32[dt_nonneg]
                    margin64 = dt[dt_nonneg].to(dtype=torch.float64) - need64[dt_nonneg]
                    def _q(vals: torch.Tensor, probs: list[float]) -> list[float]:
                        q = torch.quantile(vals, torch.tensor(probs, device=vals.device))
                        return [float(v.item()) for v in q]
                    probs = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
                    q32 = _q(margin32, probs)
                    q64 = _q(margin64, probs)
                    near = int((margin64.abs() <= eps).sum().item())
                else:
                    q32 = q64 = []
                    near = 0

                miss_K_mask = valid_o64 & (~valid_k32)
                miss_O_mask = (~valid_o64) & valid_k32
                miss_K = int(miss_K_mask.sum().item())
                miss_O = int(miss_O_mask.sum().item())
                near_mask64 = (dt.to(dtype=torch.float64) - need64).abs() <= eps

                # Examples (cap to 5)
                ex_K = []
                if miss_K > 0:
                    idxs = torch.nonzero(miss_K_mask, as_tuple=False).squeeze(1)[:5]
                    for a_idx_t in idxs:
                        a_i = int(a_idx_t.item())
                        ex_K.append(
                            {
                                "a_idx": a_i,
                                "dt_s": int(dt[a_i].item()),
                                "dist32_m": float(d32[a_i].item()),
                                "dist64_m": float(d64[a_i].item()),
                                "need_time_s": float((d64[a_i] / speed).item()),
                                "margin32_s": float((dt[a_i].to(dtype=torch.float32) - need32[a_i]).item()),
                                "margin64_s": float((dt[a_i].to(dtype=torch.float64) - need64[a_i]).item()),
                            }
                        )

                ex_O = []
                if miss_O > 0:
                    idxs = torch.nonzero(miss_O_mask, as_tuple=False).squeeze(1)[:5]
                    for a_idx_t in idxs:
                        a_i = int(a_idx_t.item())
                        ex_O.append(
                            {
                                "a_idx": a_i,
                                "dt_s": int(dt[a_i].item()),
                                "dist32_m": float(d32[a_i].item()),
                                "dist64_m": float(d64[a_i].item()),
                                "need_time_s": float((d64[a_i] / speed).item()),
                                "margin32_s": float((dt[a_i].to(dtype=torch.float32) - need32[a_i]).item()),
                                "margin64_s": float((dt[a_i].to(dtype=torch.float64) - need64[a_i]).item()),
                            }
                        )

                # Zero-slack and top-5 smallest slacks for this row
                slack_row = self.compute_slack_tile(samp[row : row + 1], state, workspace)  # [1, N]
                slack_row = slack_row[0]
                zero_count = int((slack_row == 0).sum().item())
                k_top = min(5, N)
                top_vals, top_idx = torch.topk(slack_row, k_top, largest=False)
                top_list = []
                for j in range(k_top):
                    a_i = int(top_idx[j].item())
                    top_list.append(
                        {
                            "a_idx": a_i,
                            "slack": int(top_vals[j].item()),
                            "dt_s": int(dt[a_i].item()),
                            "dist_km": float((d64[a_i] / 1000.0).item()),
                            "need_time_s": float((d64[a_i] / speed).item()),
                            "margin64_s": float((dt[a_i].to(dtype=torch.float64) - need64[a_i]).item()),
                        }
                    )

                # Time formatting (UTC ISO)
                import datetime as _dt
                tB_iso = _dt.datetime.utcfromtimestamp(tB_val).strftime("%Y-%m-%dT%H:%M:%SZ")

                row_payload = {
                    "b_idx": b_idx,
                    "tB_epoch": tB_val,
                    "tB_iso": tB_iso,
                    "N": N,
                    "eligible_by_time": eligible,
                    "future_invalid": future_invalid,
                    "oracle_valid": oracle_valid,
                    "kernel_allowed": kernel_allowed,
                    "miss_kernel": miss_K,
                    "miss_kernel_near": int((miss_K_mask & near_mask64).sum().item()),
                    "false_positive": miss_O,
                    "margin64_quantiles": (
                        {
                            "min": q64[0],
                            "p01": q64[1],
                            "p10": q64[2],
                            "p50": q64[3],
                            "p90": q64[4],
                            "p99": q64[5],
                            "max": q64[6],
                        }
                        if q64
                        else None
                    ),
                    "margin32_quantiles": (
                        {
                            "min": q32[0],
                            "p01": q32[1],
                            "p10": q32[2],
                            "p50": q32[3],
                            "p90": q32[4],
                            "p99": q32[5],
                            "max": q32[6],
                        }
                        if q32
                        else None
                    ),
                    "near_boundary_count": near,
                    "zero_slack_count": zero_count,
                    "top_slack": top_list,
                    "miss_kernel_examples": ex_K,
                    "false_positive_examples": ex_O,
                }
                sentinels.append(row_payload)

            return {
                "iteration": state.iteration,
                "sentinel_count": S,
                "speed_mps": float(workspace.speed_mps),
                "A_count": N,
                "sentinels": sentinels,
            }
        except Exception:
            # Diagnostics must never interfere with solving
            return None


register_kernel("haversine_speed", HaversineSpeedKernel)
