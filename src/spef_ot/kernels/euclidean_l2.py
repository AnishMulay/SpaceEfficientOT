from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import torch
from .base import SlackKernel, Workspace
from .registry import register_kernel
from ..core.problem import Problem
from ..core.state import SolverState


def _raw_euclidean_l2_slack(
    xb: torch.Tensor,
    xAT: torch.Tensor,
    xA2: torch.Tensor,
    yA: torch.Tensor,
    yB_idx: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    xb2 = (xb * xb).sum(dim=1, keepdim=True)
    d2 = xb2 + xA2 - 2.0 * (xb @ xAT)
    dist = torch.sqrt(torch.clamp(d2, min=0.0))
    scaled = torch.floor(dist * scale).to(torch.int64)
    return scaled - yA.unsqueeze(0) - yB_idx.unsqueeze(1)


if hasattr(torch, "compile"):
    try:
        _compiled_euclidean_l2_slack = torch.compile(
            _raw_euclidean_l2_slack, mode="reduce-overhead", dynamic=True
        )
    except Exception:
        _compiled_euclidean_l2_slack = _raw_euclidean_l2_slack
else:
    _compiled_euclidean_l2_slack = _raw_euclidean_l2_slack


@dataclass
class _EuclideanL2Workspace:
    xA: torch.Tensor
    xB: torch.Tensor
    xAT: torch.Tensor
    xA2: torch.Tensor
    scale: torch.Tensor


class EuclideanL2Kernel(SlackKernel):
    """Plain L2 (Euclidean) distance kernel. No constraints."""

    def prepare(self, problem: Problem) -> Workspace:
        xA = problem.xA.to(dtype=torch.float32)
        xB = problem.xB.to(dtype=torch.float32)
        scale_value = 3.0 / (problem.C_value * problem.delta_value)
        scale = torch.tensor(scale_value, dtype=torch.float32, device=problem.device)
        xAT = xA.transpose(0, 1).contiguous()
        xA2 = (xA * xA).sum(dim=1, keepdim=True).transpose(0, 1).contiguous()
        return _EuclideanL2Workspace(xA=xA, xB=xB, xAT=xAT, xA2=xA2, scale=scale)

    def compute_slack_tile(self, idxB, state, workspace, out=None):
        K_actual = int(idxB.numel())
        if K_actual == 0:
            return out[:0] if out is not None else state.yA.new_empty((0, workspace.xA.shape[0]))

        K_target = int(out.shape[0]) if out is not None else K_actual
        pad = K_target - K_actual
        if pad > 0:
            idx_pad = torch.cat((idxB, idxB[-1].repeat(pad)))
        else:
            idx_pad = idxB

        xb = workspace.xB.index_select(0, idx_pad).to(dtype=torch.float32)
        yB_idx = state.yB.index_select(0, idx_pad)
        slack = _compiled_euclidean_l2_slack(
            xb, workspace.xAT, workspace.xA2, state.yA, yB_idx, workspace.scale
        )
        if out is not None:
            out.copy_(slack)
            return out[:K_actual]
        return slack[:K_actual]

    def pair_cost(self, rows, cols, problem, workspace):
        xb = workspace.xB.index_select(0, rows).to(dtype=torch.float64)
        xa = workspace.xA.index_select(0, cols).to(dtype=torch.float64)
        diff = xb - xa
        return torch.sqrt(torch.clamp((diff * diff).sum(dim=1), min=0.0))

    def finalize(self, problem, state, workspace):
        return {}


register_kernel("euclidean_l2", EuclideanL2Kernel)
