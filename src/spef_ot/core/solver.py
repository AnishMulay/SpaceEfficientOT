from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple, Union

import torch

from .problem import Problem
from .state import SolverState
from ..kernels.base import SlackKernel
from ..kernels.registry import get_kernel


@dataclass
class MatchResult:
    Mb: torch.Tensor
    yA: torch.Tensor
    yB: torch.Tensor
    matching_cost: torch.Tensor
    iterations: int
    metrics: dict[str, float]


def _unique_with_first(x: torch.Tensor, *, input_sorted: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mirror of the helper used in the archive implementation.
    Returns unique values along with the index of their first occurrence.
    """
    if x.numel() == 0:
        return x, x.new_empty((0,), dtype=torch.long)

    unique_vals, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    first = counts.cumsum(0)
    first = torch.cat((first.new_zeros(1), first[:-1]))
    if not input_sorted:
        _, order = torch.sort(inverse, stable=True)
        first = order[first]
    return unique_vals, first


def match(
    xA: torch.Tensor,
    xB: torch.Tensor,
    *,
    kernel: Union[str, type[SlackKernel], SlackKernel] = "euclidean_sq",
    C: Union[float, torch.Tensor] = 1.0,
    k: int = 1,
    delta: Union[float, torch.Tensor] = 1.0,
    device: Union[str, torch.device, None] = None,
    seed: int = 1,
    stopping_condition: int | None = None,
    fill_policy: str = "greedy",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    **kernel_kwargs: Any,
) -> MatchResult:
    """
    Solve the space-efficient matching problem between point sets `xA` and `xB`.
    """

    if fill_policy not in ("greedy", "none"):
        raise ValueError(f"fill_policy must be either 'greedy' or 'none', got {fill_policy!r}.")

    if device is None:
        device = xA.device if xA.device.type != "cpu" or xA.is_cuda else xB.device
    torch_device = torch.device(device)

    xA_dev = xA.to(device=torch_device)
    xB_dev = xB.to(device=torch_device)

    kernel_instance = get_kernel(kernel, **kernel_kwargs)

    problem = Problem(
        xA=xA_dev,
        xB=xB_dev,
        C=C,
        delta=delta,
        device=torch_device,
    )

    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(seed))

    state = SolverState(
        yA=torch.zeros(problem.n, dtype=torch.int64, device=torch_device),
        yB=torch.ones(problem.m, dtype=torch.int64, device=torch_device),
        Ma=torch.full((problem.n,), -1, dtype=torch.int64, device=torch_device),
        Mb=torch.full((problem.m,), -1, dtype=torch.int64, device=torch_device),
        iteration=0,
        metrics={},
        generator=generator,
    )

    workspace = kernel_instance.prepare(problem)

    k = int(max(1, k))
    slack_tile = torch.empty((k, problem.n), dtype=torch.int64, device=torch_device)

    f = float(problem.n)
    default_threshold = problem.m * problem.delta_value / problem.C_value
    f_threshold = (
        float(stopping_condition)
        if stopping_condition is not None
        else default_threshold
    )

    inner_loops = 0

    while f > f_threshold:
        free_mask = state.Mb == -1
        if not bool(free_mask.any()):
            break
        ind_b_all_free = torch.nonzero(free_mask, as_tuple=False).squeeze(1)
        if progress_callback is not None:
            free_count = int(ind_b_all_free.numel())
            matched = problem.m - free_count
            progress_callback(
                "iteration",
                {
                    "iteration": state.iteration,
                    "free_b": free_count,
                    "matched_b": matched,
                    "objective_gap": f,
                    "threshold": f_threshold,
                },
            )

        for start_idx in range(0, ind_b_all_free.numel(), k):
            end_idx = min(start_idx + k, ind_b_all_free.numel())
            idxB = ind_b_all_free[start_idx:end_idx]
            if idxB.numel() == 0:
                continue
            if progress_callback is not None:
                progress_callback(
                    "tile",
                    {
                        "iteration": state.iteration,
                        "tile_index": start_idx // k,
                        "tile_size": int(idxB.numel()),
                        "tile_start": start_idx,
                        "tile_end": end_idx,
                    },
                )

            # Pass the full preallocated buffer to enable constant-K execution
            slack_values = kernel_instance.compute_slack_tile(idxB, state, workspace, out=slack_tile)

            zero_rows, zero_cols = torch.where(slack_values == 0)

            ind_a_push = zero_rows.new_empty((0,), dtype=torch.int64)
            ind_b_push_local = zero_rows.new_empty((0,), dtype=torch.int64)

            if zero_rows.numel() > 0:
                row_changes = torch.nonzero(zero_rows[1:] != zero_rows[:-1], as_tuple=False).squeeze(1) + 1
                group_starts = torch.cat((zero_rows.new_zeros(1), row_changes))
                group_ends = torch.cat((group_starts[1:], zero_rows.new_tensor([zero_rows.numel()])))
                group_sizes = group_ends - group_starts

                rand = torch.rand(group_sizes.shape, device=torch_device, generator=generator)
                pick_offsets = (rand * group_sizes.to(rand.dtype)).to(torch.int64)
                picks = group_starts + pick_offsets

                candidate_cols = zero_cols.index_select(0, picks)
                candidate_rows_local = zero_rows.index_select(0, picks)

                ind_a_push, first_positions = _unique_with_first(candidate_cols, input_sorted=False)
                ind_b_push_local = candidate_rows_local.index_select(0, first_positions)

            if ind_a_push.numel() > 0:
                ind_b_push = idxB.index_select(0, ind_b_push_local)

                ma_prev = state.Ma.index_select(0, ind_a_push)
                release_mask = ma_prev != -1
                num_release = int(release_mask.sum().item())
                if num_release > 0:
                    active_idx = torch.nonzero(release_mask, as_tuple=False).squeeze(1)
                    release_a = ind_a_push.index_select(0, active_idx)
                    release_b = ma_prev.index_select(0, active_idx)
                    state.Mb[release_b] = -1

                state.Ma[ind_a_push] = ind_b_push
                state.Mb[ind_b_push] = ind_a_push
                state.yA[ind_a_push] -= 1

                f -= float(ind_a_push.numel() - num_release)

            min_slack = torch.min(slack_values, dim=1).values
            non_zero_mask = min_slack != 0
            if non_zero_mask.any():
                ind_b_not_pushed = idxB.index_select(
                    0, torch.nonzero(non_zero_mask, as_tuple=False).squeeze(1)
                )
                delta_yB = min_slack.index_select(
                    0, torch.nonzero(non_zero_mask, as_tuple=False).squeeze(1)
                )
                state.yB.index_add_(0, ind_b_not_pushed, delta_yB)

            inner_loops += 1

    state.iteration += 1

    # Final fill for unmatched rows
    Ma = state.Ma
    Mb = state.Mb

    if fill_policy == "greedy":
        ind_a = 0
        for ind_b in range(problem.m):
            if Mb[ind_b].item() == -1:
                while Ma[ind_a].item() != -1:
                    ind_a += 1
                Mb[ind_b] = ind_a
                Ma[ind_a] = ind_b

    # Allow kernel to finish any clean-up (e.g. removing capped matches)
    finalize_metrics = kernel_instance.finalize(problem, state, workspace) or {}
    state.metrics.update(finalize_metrics)

    matched_mask = Mb != -1
    if matched_mask.any():
        rows = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
        cols = Mb.index_select(0, rows)
        pair_costs = kernel_instance.pair_cost(rows, cols, problem, workspace)
        matching_cost = pair_costs.sum(dtype=torch.float64)
    else:
        matching_cost = torch.tensor(0.0, dtype=torch.float64, device=torch_device)

    result = MatchResult(
        Mb=Mb.detach().cpu(),
        yA=state.yA.detach().cpu(),
        yB=state.yB.detach().cpu(),
        matching_cost=matching_cost.detach().cpu(),
        iterations=state.iteration,
        metrics={**state.metrics, "inner_loops": float(inner_loops)},
    )

    return result
