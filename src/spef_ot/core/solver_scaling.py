from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Tuple, Union

import torch

# Reuse existing structures to maintain compatibility
from .problem import Problem
from .state import SolverState
from ..kernels.base import SlackKernel
from ..kernels.registry import get_kernel


@dataclass
class ScalingMatchResult:
    Mb: torch.Tensor
    yA: torch.Tensor
    yB: torch.Tensor
    matching_cost: torch.Tensor
    iterations: int
    phases: int
    metrics: dict[str, float]


def _unique_with_first(x: torch.Tensor, *, input_sorted: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper: Returns unique values along with the index of their first occurrence.
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


def _solve_inner(
    problem: Problem,
    state: SolverState,
    kernel_instance: SlackKernel,
    workspace: Any,
    k: int,
    stopping_condition: float | None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
    phase_id: int,
    fill_policy: str = "none" # Inner loop usually doesn't fill, we fill at the very end
) -> int:
    """
    The inner solver loop. Runs until the objective gap `f` meets the threshold defined 
    by the current problem's epsilon/delta.
    
    Returns:
        int: Number of inner loop iterations (tiles processed).
    """
    # Recalculate k to ensure it's valid
    k = int(max(1, k))
    slack_tile = torch.empty((k, problem.n), dtype=torch.int64, device=problem.device)

    # Initial objective gap 'f'. 
    # In the scaling context, we just need to drive matches to completion relative to the new C/delta.
    # Note: 'f' approximates the number of augmenting paths needed.
    # We estimate 'f' based on free nodes, though the original logic started f=n.
    # Here, we continue from where we left off.
    
    # Calculate threshold based on CURRENT problem's parameters
    default_threshold = problem.m * problem.delta_value / problem.C_value
    f_threshold = (
        float(stopping_condition)
        if stopping_condition is not None
        else default_threshold
    )
    
    # In scaling, we might want to run until *all* free nodes that CAN be matched ARE matched.
    # However, we stick to the original stopping logic for consistency.
    # We reset f to the number of free nodes + some buffer to drive the loop? 
    # Original solver sets f = n. Let's stick to that to ensure we process enough.
    f = float(problem.n) 
    
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
                    "phase": phase_id,
                    "iteration": state.iteration,
                    "free_b": free_count,
                    "matched_b": matched,
                    "objective_gap": f,
                    "threshold": f_threshold,
                },
            )

        a_claimed = torch.zeros((problem.n,), dtype=torch.bool, device=problem.device)

        # Iterate over free B nodes in tiles
        for start_idx in range(0, ind_b_all_free.numel(), k):
            end_idx = min(start_idx + k, ind_b_all_free.numel())
            idxB = ind_b_all_free[start_idx:end_idx]
            
            if idxB.numel() == 0:
                continue

            # Compute slacks for this tile using the CURRENT kernel (prepared with current epsilon)
            # This returns integer slacks relative to the current scale.
            slack_values = kernel_instance.compute_slack_tile(idxB, state, workspace, out=slack_tile)

            zero_rows, zero_cols = torch.where(slack_values == 0)

            # Deduplication logic (same as original solver)
            if zero_cols.numel() > 0:
                keep_mask = ~a_claimed.index_select(0, zero_cols)
                if not bool(keep_mask.all()):
                    idx_keep = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
                    if idx_keep.numel() == 0:
                        zero_rows = zero_rows.new_empty((0,), dtype=zero_rows.dtype)
                        zero_cols = zero_cols.new_empty((0,), dtype=zero_cols.dtype)
                    else:
                        zero_rows = zero_rows.index_select(0, idx_keep)
                        zero_cols = zero_cols.index_select(0, idx_keep)

            ind_a_push = zero_rows.new_empty((0,), dtype=torch.int64)
            ind_b_push_local = zero_rows.new_empty((0,), dtype=torch.int64)

            # Randomized selection of matches (same as original solver)
            if zero_rows.numel() > 0:
                row_changes = torch.nonzero(zero_rows[1:] != zero_rows[:-1], as_tuple=False).squeeze(1) + 1
                group_starts = torch.cat((zero_rows.new_zeros(1), row_changes))
                group_ends = torch.cat((group_starts[1:], zero_rows.new_tensor([zero_rows.numel()])))
                group_sizes = group_ends - group_starts

                rand = torch.rand(group_sizes.shape, device=problem.device, generator=state.generator)
                pick_offsets = (rand * group_sizes.to(rand.dtype)).to(torch.int64)
                picks = group_starts + pick_offsets

                candidate_cols = zero_cols.index_select(0, picks)
                candidate_rows_local = zero_rows.index_select(0, picks)

                ind_a_push, first_positions = _unique_with_first(candidate_cols, input_sorted=False)
                ind_b_push_local = candidate_rows_local.index_select(0, first_positions)

            # Apply matches
            add_count = 0
            num_release = 0
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

                a_claimed[ind_a_push] = True

                add_count = int(ind_a_push.numel())
                f -= float(add_count - num_release)

            # Dual updates
            min_slack = torch.min(slack_values, dim=1).values
            non_zero_mask = min_slack != 0
            if non_zero_mask.any():
                nz_rows = torch.nonzero(non_zero_mask, as_tuple=False).squeeze(1)
                ind_b_not_pushed = idxB.index_select(0, nz_rows)
                delta_yB = min_slack.index_select(0, nz_rows)
                state.yB.index_add_(0, ind_b_not_pushed, delta_yB)

            inner_loops += 1
        
        state.iteration += 1

    return inner_loops


def scaling_match(
    xA: torch.Tensor,
    xB: torch.Tensor,
    *,
    kernel: Union[str, type[SlackKernel], SlackKernel] = "euclidean_sq",
    C: Union[float, torch.Tensor] = 1.0,
    k: int = 512,
    target_delta: float = 0.001,
    initial_delta: float | None = None,
    device: Union[str, torch.device, None] = None,
    seed: int = 1,
    stopping_condition: int | None = None,
    fill_policy: str = "greedy",
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    verbose: bool = True,
    **kernel_kwargs: Any,
) -> ScalingMatchResult:
    """
    Solves the matching problem using Epsilon-Scaling (Cost Scaling).
    
    The algorithm starts with a coarse epsilon (initial_delta) and iteratively 
    refines it by half until it reaches `target_delta`.
    
    At each step:
    1. Epsilon is halved.
    2. Potentials yA, yB are doubled (to maintain integer representation relative to new scale).
    3. Existing matches are checked for feasibility under the new epsilon. Violations are pruned.
    4. The solver runs to fix the solution using the warm-start.
    """
    
    if fill_policy not in ("greedy", "none"):
        raise ValueError(f"fill_policy must be either 'greedy' or 'none', got {fill_policy!r}.")

    if device is None:
        device = xA.device if xA.device.type != "cpu" or xA.is_cuda else xB.device
    torch_device = torch.device(device)

    xA_dev = xA.to(device=torch_device)
    xB_dev = xB.to(device=torch_device)
    
    n, m = xA_dev.shape[0], xB_dev.shape[0]

    # Initialize State once
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(seed))
    
    state = SolverState(
        yA=torch.zeros(n, dtype=torch.int64, device=torch_device),
        yB=torch.ones(m, dtype=torch.int64, device=torch_device),
        Ma=torch.full((n,), -1, dtype=torch.int64, device=torch_device),
        Mb=torch.full((m,), -1, dtype=torch.int64, device=torch_device),
        iteration=0,
        metrics={},
        generator=generator,
    )

    # Determine scaling schedule
    # If initial_delta not provided, pick something large enough (e.g. C / 10 or similar, or just 1.0)
    # A safe bet is usually to start where the original solver would have started, 
    # but for scaling to be effective, we want a larger delta.
    # Let's default initial_delta to target_delta * 2^4 (16x coarser) if not provided, just as a heuristic.
    if initial_delta is None:
        # Heuristic: 4-5 scales
        initial_delta = target_delta * 16.0 
    
    current_delta = initial_delta
    phase = 0
    total_inner_loops = 0
    
    # We create the kernel instance once (it might have internal caches, 
    # but prepare() should handle problem changes)
    kernel_instance = get_kernel(kernel, **kernel_kwargs)

    t_start_global = time.perf_counter()

    while True:
        # Ensure we don't overshoot target
        if current_delta < target_delta:
            current_delta = target_delta
        
        if verbose:
            print(f"\n=== Scaling Phase {phase}: delta={current_delta:.6f} ===")

        # 1. Create Problem Definition for this scale
        # The Problem class is immutable, so we make a new one. 
        # This re-calculates C_value/delta_value relationships used by the kernel.
        problem = Problem(
            xA=xA_dev,
            xB=xB_dev,
            C=C,
            delta=current_delta,
            device=torch_device,
        )
        
        # 2. Prepare workspace for this problem
        # IMPORTANT: The kernel uses problem.delta to determine discretization.
        workspace = kernel_instance.prepare(problem)
        
        # 3. Scaling & Pruning (Skip on first phase)
        if phase > 0:
            # A. Scale Up Potentials
            # Since we halved delta, the integer representation of the same physical cost doubles.
            # So we must double the potentials to keep them physically consistent.
            state.yA *= 2
            state.yB *= 2
            
            # B. Prune Violations
            # Now that potentials are scaled and delta is tighter, some edges might have slack != 0.
            # We must identify and break them.
            
            # Get currently matched pairs
            matched_b_indices = torch.nonzero(state.Mb != -1, as_tuple=False).squeeze(1)
            
            if matched_b_indices.numel() > 0:
                # We calculate slack for these matches.
                # To do this efficiently without a custom kernel method, we use compute_slack_tile 
                # on the matched rows.
                # Note: compute_slack_tile computes slack against ALL A or a subset. 
                # We need specific (b, a) slack.
                # Optimization: Process in chunks
                
                violations_count = 0
                chunk_size = k
                
                for start in range(0, matched_b_indices.numel(), chunk_size):
                    end = min(start + chunk_size, matched_b_indices.numel())
                    b_chunk = matched_b_indices[start:end]
                    
                    # Compute full slack row for these B
                    # This returns [chunk, n] matrix
                    slacks = kernel_instance.compute_slack_tile(b_chunk, state, workspace)
                    
                    # We only care about the column corresponding to the match
                    # Get matched A for these B
                    a_matched = state.Mb.index_select(0, b_chunk)
                    
                    # Extract the diagonal (the slack of the matched edges)
                    # slacks is [B_chunk, N]. We want slacks[i, a_matched[i]]
                    matched_slacks = slacks.gather(1, a_matched.unsqueeze(1)).squeeze(1)
                    
                    # Identify violations (slack != 0)
                    # In this integer domain, valid matches must have 0 slack.
                    violation_mask = matched_slacks != 0
                    
                    if violation_mask.any():
                        bad_local_indices = torch.nonzero(violation_mask, as_tuple=False).squeeze(1)
                        bad_b = b_chunk.index_select(0, bad_local_indices)
                        bad_a = a_matched.index_select(0, bad_local_indices)
                        
                        # Unmatch
                        state.Mb[bad_b] = -1
                        state.Ma[bad_a] = -1
                        
                        violations_count += int(bad_local_indices.numel())
                
                if verbose:
                    print(f"  [Pruning] Scaled potentials x2. Checked {matched_b_indices.numel()} matches. Pruned {violations_count} violations.")

        # 4. Solve for this scale
        loops = _solve_inner(
            problem=problem,
            state=state,
            kernel_instance=kernel_instance,
            workspace=workspace,
            k=k,
            stopping_condition=stopping_condition,
            progress_callback=progress_callback if verbose else None,
            phase_id=phase,
            fill_policy="none" # Defer filling to the very end
        )
        total_inner_loops += loops
        
        # Check termination
        if current_delta <= target_delta + 1e-9: # Float comparison tolerance
            break
            
        # Prepare next delta (halving)
        current_delta = current_delta / 2.0
        phase += 1

    # Finalize (Greedy Fill if requested)
    if fill_policy == "greedy":
        if verbose:
            print("\n=== Finalizing (Greedy Fill) ===")
        ind_a = 0
        for ind_b in range(problem.m):
            if state.Mb[ind_b].item() == -1:
                while ind_a < problem.n and state.Ma[ind_a].item() != -1:
                    ind_a += 1
                if ind_a < problem.n:
                    state.Mb[ind_b] = ind_a
                    state.Ma[ind_a] = ind_b

    # Metrics
    finalize_metrics = kernel_instance.finalize(problem, state, workspace) or {}
    state.metrics.update(finalize_metrics)
    state.metrics["total_phases"] = float(phase + 1)
    state.metrics["total_inner_loops"] = float(total_inner_loops)
    state.metrics["total_runtime"] = time.perf_counter() - t_start_global

    # Compute final cost
    matched_mask = state.Mb != -1
    if matched_mask.any():
        rows = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
        cols = state.Mb.index_select(0, rows)
        pair_costs = kernel_instance.pair_cost(rows, cols, problem, workspace)
        matching_cost = pair_costs.sum(dtype=torch.float64)
    else:
        matching_cost = torch.tensor(0.0, dtype=torch.float64, device=torch_device)

    return ScalingMatchResult(
        Mb=state.Mb.detach().cpu(),
        yA=state.yA.detach().cpu(),
        yB=state.yB.detach().cpu(),
        matching_cost=matching_cost.detach().cpu(),
        iterations=state.iteration,
        phases=phase + 1,
        metrics=state.metrics,
    )