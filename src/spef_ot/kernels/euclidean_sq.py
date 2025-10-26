from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import SlackKernel, Workspace
from .registry import register_kernel
from ..core.problem import Problem
from ..core.state import SolverState


def _raw_slack_kernel(
    xb: torch.Tensor,
    xAT: torch.Tensor,
    xa2_cached: torch.Tensor,
    yA: torch.Tensor,
    yB_idx: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    xb2 = (xb * xb).sum(dim=1, keepdim=True)
    w = xb2 + xa2_cached - 2.0 * (xb @ xAT)
    scaled = torch.floor(w * scale)
    slack = scaled.to(torch.int64) - yA.unsqueeze(0) - yB_idx.unsqueeze(1)
    return slack


if hasattr(torch, "compile"):
    try:
        _compiled_slack_kernel = torch.compile(
            _raw_slack_kernel, mode="reduce-overhead", dynamic=True
        )
    except Exception:  # pragma: no cover - compilation fallback
        _compiled_slack_kernel = _raw_slack_kernel
else:  # pragma: no cover - older PyTorch fallback
    _compiled_slack_kernel = _raw_slack_kernel


@dataclass
class _EuclideanWorkspace:
    xA: torch.Tensor
    xB: torch.Tensor
    xAT: torch.Tensor
    xa2_cached: torch.Tensor
    scale: torch.Tensor


class SquaredEuclideanKernel(SlackKernel):
    """
    Slack kernel implementing squared Euclidean costs using a fused kernel.
    """

    def prepare(self, problem: Problem) -> Workspace:
        xA = problem.xA
        xB = problem.xB
        dtype = xA.dtype
        scale_value = 3.0 / (problem.C_value * problem.delta_value)
        scale = torch.tensor(scale_value, dtype=dtype, device=problem.device)

        xAT = xA.transpose(0, 1).contiguous()
        xa2 = (xA * xA).sum(dim=1, keepdim=True).transpose(0, 1).contiguous()

        return _EuclideanWorkspace(
            xA=xA,
            xB=xB,
            xAT=xAT,
            xa2_cached=xa2,
            scale=scale,
        )

    def compute_slack_tile(
        self,
        idxB: torch.Tensor,
        state: SolverState,
        workspace: _EuclideanWorkspace,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        K_actual = int(idxB.numel())
        if K_actual == 0:
            return out[:0] if out is not None else state.yA.new_empty((0, workspace.xA.shape[0]))

        if out is None:
            # No prealloc provided; compute at actual size.
            xb = workspace.xB.index_select(0, idxB).to(dtype=workspace.xA.dtype)
            yB_idx = state.yB.index_select(0, idxB)
            slack = _compiled_slack_kernel(
                xb,
                workspace.xAT,
                workspace.xa2_cached,
                state.yA,
                yB_idx,
                workspace.scale,
            )
            return slack

        # Use constant-K execution by padding to out.shape[0]
        K_target = int(out.shape[0])
        if K_target == K_actual:
            xb = workspace.xB.index_select(0, idxB).to(dtype=workspace.xA.dtype)
            yB_idx = state.yB.index_select(0, idxB)
            slack = _compiled_slack_kernel(
                xb,
                workspace.xAT,
                workspace.xa2_cached,
                state.yA,
                yB_idx,
                workspace.scale,
            )
            out_view = out[:K_actual]
            out_view.copy_(slack)
            return out_view

        # Pad indices to K_target by repeating the last valid index
        pad_count = K_target - K_actual
        last_idx = idxB[-1]
        idxB_padded = torch.cat((idxB, last_idx.repeat(pad_count)))

        xb = workspace.xB.index_select(0, idxB_padded).to(dtype=workspace.xA.dtype)
        yB_idx = state.yB.index_select(0, idxB_padded)

        slack_k = _compiled_slack_kernel(
            xb,
            workspace.xAT,
            workspace.xa2_cached,
            state.yA,
            yB_idx,
            workspace.scale,
        )
        out.copy_(slack_k)
        return out[:K_actual]

    def pair_cost(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        problem: Problem,
        workspace: _EuclideanWorkspace,
    ) -> torch.Tensor:
        xb = workspace.xB.index_select(0, rows).to(dtype=torch.float64)
        xa = workspace.xA.index_select(0, cols).to(dtype=torch.float64)
        diff = xb - xa
        return (diff * diff).sum(dim=1)


register_kernel("euclidean_sq", SquaredEuclideanKernel)
