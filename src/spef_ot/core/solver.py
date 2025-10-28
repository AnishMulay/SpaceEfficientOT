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

        # Sentinel-B deep dive: sample a few free B rows and report feasibility
        if progress_callback is not None:
            try:
                # Only run when kernel workspace exposes necessary fields (e.g., HaversineSpeed)
                has_fields = all(
                    hasattr(workspace, name)
                    for name in (
                        "EB",
                        "EA_T",
                        "radius_m",
                        "times_A",
                        "times_B",
                    )
                )
                use_speed = bool(getattr(workspace, "use_speed", False))
                speed_mps = float(getattr(workspace, "speed_mps", 0.0))
                if has_fields and use_speed and speed_mps > 0.0:
                    import datetime as _dt

                    # Recompute current free set after tile updates
                    free_mask_cur = state.Mb == -1
                    free_idx = torch.nonzero(free_mask_cur, as_tuple=False).squeeze(1)
                    S = min(6, int(free_idx.numel()))
                    sentinels: list[dict[str, Any]] = []
                    if S > 0:
                        # Sample without replacement using solver RNG
                        if S == int(free_idx.numel()):
                            samp = free_idx
                        else:
                            perm = torch.randperm(int(free_idx.numel()), device=torch_device, generator=generator)
                            samp = free_idx.index_select(0, perm[:S])

                        # Build batch EB rows and compute distances against all A in fp32 and fp64
                        EB_rows_f32 = workspace.EB.index_select(0, samp).to(dtype=torch.float32)
                        EA_T_f32 = workspace.EA_T.to(dtype=torch.float32)
                        cos32 = EB_rows_f32 @ EA_T_f32  # [S, N]
                        cos32 = torch.clamp(cos32, -1.0, 1.0)
                        dist32 = torch.acos(cos32) * float(workspace.radius_m)

                        EB_rows_f64 = EB_rows_f32.to(dtype=torch.float64)
                        EA_T_f64 = EA_T_f32.to(dtype=torch.float64)
                        cos64 = EB_rows_f64 @ EA_T_f64
                        cos64 = torch.clamp(cos64, -1.0, 1.0)
                        dist64 = torch.acos(cos64) * float(workspace.radius_m)

                        tA = workspace.times_A.to(dtype=torch.int64)
                        tB_sel = workspace.times_B.index_select(0, samp).to(dtype=torch.int64)

                        N = int(tA.numel())
                        eps = 1.0  # seconds for near-boundary

                        # For each sentinel b, compute per-row stats
                        for row in range(S):
                            b_idx = int(samp[row].item())
                            tB_val = int(tB_sel[row].item())
                            # Broadcast dt over columns
                            dt = (tB_sel[row].view(1, 1) - tA.view(1, -1)).squeeze(0)  # [N], int64
                            dt_nonneg = dt >= 0

                            dist32_row = dist32[row, :]
                            dist64_row = dist64[row, :]

                            # Kernel allowed (float32 distance)
                            need32 = (dist32_row / speed_mps).to(dtype=torch.float32)
                            valid_k32 = dt_nonneg & (dt.to(dtype=need32.dtype) >= need32)

                            # Oracle valid (float64 distance)
                            need64 = (dist64_row / speed_mps).to(dtype=torch.float64)
                            valid_o64 = dt_nonneg & (dt.to(dtype=need64.dtype) >= need64)

                            eligible = int(dt_nonneg.sum().item())
                            future_invalid = N - eligible

                            kernel_allowed = int(valid_k32.sum().item())
                            oracle_valid = int(valid_o64.sum().item())

                            # Margins on eligible set only
                            if eligible > 0:
                                margin32 = dt[dt_nonneg].to(dtype=torch.float32) - need32[dt_nonneg]
                                margin64 = dt[dt_nonneg].to(dtype=torch.float64) - need64[dt_nonneg]
                                # Quantiles
                                def _q(vals: torch.Tensor, probs: list[float]) -> list[float]:
                                    q = torch.quantile(vals, torch.tensor(probs, device=vals.device))
                                    return [float(v.item()) for v in q]

                                q_probs = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
                                q32 = _q(margin32, q_probs)
                                q64 = _q(margin64, q_probs)
                                near = int((margin64.abs() <= eps).sum().item())
                            else:
                                q32 = q64 = []
                                near = 0

                            # Mismatches
                            miss_K_mask = valid_o64 & (~valid_k32)
                            miss_O_mask = (~valid_o64) & valid_k32
                            miss_K = int(miss_K_mask.sum().item())
                            miss_O = int(miss_O_mask.sum().item())

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
                                            "dist32_m": float(dist32_row[a_i].item()),
                                            "dist64_m": float(dist64_row[a_i].item()),
                                            "need_time_s": float((dist64_row[a_i] / speed_mps).item()),
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
                                            "dist32_m": float(dist32_row[a_i].item()),
                                            "dist64_m": float(dist64_row[a_i].item()),
                                            "need_time_s": float((dist64_row[a_i] / speed_mps).item()),
                                            "margin32_s": float((dt[a_i].to(dtype=torch.float32) - need32[a_i]).item()),
                                            "margin64_s": float((dt[a_i].to(dtype=torch.float64) - need64[a_i]).item()),
                                        }
                                    )

                            # Compute zero-slack count and smallest slacks for this sentinel row
                            slack_row = kernel_instance.compute_slack_tile(samp[row : row + 1], state, workspace)  # [1, N]
                            slack_row = slack_row[0]
                            zero_count = int((slack_row == 0).sum().item())
                            # Top-5 smallest slacks
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
                                        "dist_km": float((dist64_row[a_i] / 1000.0).item()),
                                        "need_time_s": float((dist64_row[a_i] / speed_mps).item()),
                                        "margin64_s": float((dt[a_i].to(dtype=torch.float64) - need64[a_i]).item()),
                                    }
                                )

                            tB_iso = _dt.datetime.utcfromtimestamp(tB_val).strftime("%Y-%m-%dT%H:%M:%SZ")
                            # Near-boundary count restricted to misses (use 64-bit margin)
                            near_mask64 = (dt.to(dtype=torch.float64) - need64).abs() <= eps
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

                        progress_callback(
                            "sentinel",
                            {
                                "iteration": state.iteration,
                                "sentinel_count": S,
                                "speed_mps": speed_mps,
                                "A_count": N,
                                "sentinels": sentinels,
                            },
                        )
            except Exception:
                # Never allow diagnostics to break the solver
                pass

        # Completed one outer iteration of the solver
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
    # Pre-finalize debug: report B size, matched count, and sample of yB duals
    try:
        total_b = int(problem.m)
        Mb_cur = Mb
        matched_mask = Mb_cur != -1
        matched_count = int(matched_mask.sum().item())
        print(f"[pre-finalize] B_total={total_b} matched_B={matched_count}")
        if matched_count > 0:
            idxs = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
            sample_n = min(1000, int(idxs.numel()))
            if sample_n > 0:
                perm = torch.randperm(int(idxs.numel()), device=idxs.device, generator=generator)
                sel = perm[:sample_n]
                sample_b = idxs.index_select(0, sel)
                sample_yB = state.yB.index_select(0, sample_b).detach().cpu().tolist()
                sample_b_cpu = sample_b.detach().cpu().tolist()
                print("[pre-finalize] sampled_B_indices:", sample_b_cpu)
                print("[pre-finalize] sampled_yB_duals:", sample_yB)
    except Exception:
        pass

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
