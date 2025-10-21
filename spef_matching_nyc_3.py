import numpy as np
import torch
import time
from haversine_utils import haversine_distance_cpu

# Global times for NYC mask
_tA_global = None
_tB_global = None

_EARTH_RADIUS_METERS = 6371000.0


def _unit_sphere_embedding(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Return unit-sphere embedding (x, y, z) for lat/lon in degrees."""
    lat_rad = torch.deg2rad(lat_deg)
    lon_rad = torch.deg2rad(lon_deg)
    cos_lat = torch.cos(lat_rad)
    sin_lat = torch.sin(lat_rad)
    return torch.stack(
        (
            cos_lat * torch.cos(lon_rad),
            cos_lat * torch.sin(lon_rad),
            sin_lat,
        ),
        dim=1,
    )


def _prepare_haversine_cache(
    xA: torch.Tensor,
    xB: torch.Tensor,
    C,
    delta,
    R: float = _EARTH_RADIUS_METERS,
):
    """
    Precompute unit-sphere embeddings for A and B once so each slack tile
    can reuse them without re-evaluating trig functions.
    """
    target_dtype = xA.dtype
    EA = _unit_sphere_embedding(xA[:, 1], xA[:, 0]).to(dtype=target_dtype)
    EB = _unit_sphere_embedding(xB[:, 1], xB[:, 0]).to(dtype=target_dtype)
    EA = EA.contiguous()
    EB = EB.contiguous()
    scale = 3.0 / (float(C) * float(delta))

    return {
        "EA": EA,
        "EB": EB,
        "EA_T": EA.transpose(0, 1).contiguous(),
        "R": torch.tensor(R, device=xA.device, dtype=target_dtype),
        "scale": torch.tensor(scale, device=xA.device, dtype=target_dtype),
    }


def _compute_haversine_cost_tile(idxB, geo_cache):
    if geo_cache is None:
        raise ValueError("geo_cache must be provided for Haversine slack computation.")

    EA_T = geo_cache["EA_T"]
    EB = geo_cache["EB"]
    R = geo_cache["R"]
    scale = geo_cache["scale"]

    EB_tile = EB.index_select(0, idxB)
    cos_angles = EB_tile @ EA_T
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    distances = R * torch.acos(cos_angles)

    scaled_cost = torch.floor(distances * scale).to(torch.int64)
    return scaled_cost


def unique(x, input_sorted = False):
    unique, inverse_ind, unique_count = torch.unique(x, return_inverse=True, return_counts=True)
    unique_ind = unique_count.cumsum(0)
    if not unique_ind.size()[0] == 0:
        unique_ind = torch.cat((torch.tensor([0], dtype=x.dtype, device=x.device), unique_ind[:-1]))
    if not input_sorted:
        _, sort2ori_ind = torch.sort(inverse_ind, stable=True)
        unique_ind = sort2ori_ind[unique_ind]
    return unique, unique_ind

def compute_slack_tile(
    idxB,
    xA,
    xB,
    yA,
    yB,
    C,
    delta,
    slack_tile=None,
    geo_cache=None,
    tile_times=None,
    xAT=None,
    xa2_cached=None,
    cmax_int=None,
):
    current_k = len(idxB)
    if current_k == 0:
        if slack_tile is not None:
            return slack_tile[:0]
        return torch.empty(0, xA.shape[0], dtype=torch.int64, device=xA.device)
    
    yB_idx = yB.index_select(0, idxB)

    cost_tile = _compute_haversine_cost_tile(idxB, geo_cache)

    times = tile_times if tile_times is not None else ((_tA_global), (_tB_global))
    invalid_mask = None
    if times is not None and times[0] is not None and times[1] is not None:
        tA_all, tB_all = times  # int64 [N], [N]
        tB_sel = tB_all.index_select(0, idxB)  # [K]
        invalid = tB_sel.view(-1, 1) < tA_all.view(1, -1)  # [K,N] bool
        if invalid.any():
            invalid_mask = invalid
            if cmax_int is not None:
                sentinel = cost_tile.new_full((), torch.iinfo(cost_tile.dtype).max)
                cost_tile = cost_tile.masked_fill(invalid, sentinel)
            else:
                bigM = cost_tile.new_full((), 10**12)
                cost_tile = cost_tile.masked_fill(invalid, bigM)

    if cmax_int is not None:
        cost_tile = torch.clamp_max(cost_tile, int(cmax_int))

    slack_int = cost_tile - yA.unsqueeze(0) - yB_idx.unsqueeze(1)

    if invalid_mask is not None:
        bigM_slack = slack_int.new_full((), 10**12)
        slack_int = slack_int.masked_fill(invalid_mask, bigM_slack)

    if slack_tile is not None and current_k <= slack_tile.shape[0]:
        slack_view = slack_tile[:current_k]
        slack_view.copy_(slack_int)
        return slack_view

    return slack_int  # [K,N] int64


def spef_matching_2(
    xA,
    xB,
    C,
    k,
    delta,
    device,
    seed=1,
    tA=None,
    tB=None,
    cmax_int=None,
    stopping_condition=None
):
    dtyp = torch.int64
    n = xA.shape[0]
    m = xB.shape[0]

    xA = xA.to(device=device)
    xB = xB.to(device=device)
    global _tA_global, _tB_global
    if tA is not None and tB is not None:
        _tA_global = tA.to(device=device, dtype=torch.int64)
        _tB_global = tB.to(device=device, dtype=torch.int64)
    else:
        _tA_global = None
        _tB_global = None

    yB = torch.ones(m, device=device, dtype=dtyp, requires_grad=False)
    yA = torch.zeros(n, device=device, dtype=dtyp, requires_grad=False)
    Mb = torch.ones(m, device=device, dtype=dtyp, requires_grad=False) * -1
    Ma = torch.ones(n, device=device, dtype=dtyp, requires_grad=False) * -1

    f = n
    iteration = 0

    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    minus_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    C_value = float(C)
    f_threshold = stopping_condition if stopping_condition is not None else m*delta/C_value
    torch.manual_seed(seed)
    
    geo_cache = _prepare_haversine_cache(xA, xB, C_value, delta)
    
    # Note: Haversine doesn't need precomputed caches like Euclidean
    xAT = None  # Not used for Haversine
    xa2_cached = None  # Not used for Haversine
    
    # Pre-allocate slack tile for reuse
    slack_tile = torch.zeros(k, n, device=device, dtype=torch.int64)
    
    # Existing coarse timing metrics
    slack_compute_total = 0.0
    tile_updates_total = 0.0
    inner_loops_count = 0
    
    # CUDA events for GPU timing (no CPU sync)
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    while f > f_threshold:
        # Get all free B points
        ind_b_all_free = torch.where(Mb == minus_one)[0]

        if stopping_condition is not None and ind_b_all_free.numel() <= stopping_condition:
            break
        
        # Process k points at a time
        for start_idx in range(0, len(ind_b_all_free), k):
            end_idx = min(start_idx + k, len(ind_b_all_free))
            ind_b_free = ind_b_all_free[start_idx:end_idx]
            
            # Time slack tile compute
            if device.type == "cuda":
                start_event.record()
            else:
                t0 = time.perf_counter()
            slack_tile_used = compute_slack_tile(
                ind_b_free,
                xA,
                xB,
                yA,
                yB,
                C,
                delta,
                slack_tile,
                geo_cache=geo_cache,
                xAT=xAT,
                xa2_cached=xa2_cached,
                cmax_int=cmax_int,
            )
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                slack_elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:
                t1 = time.perf_counter()
                slack_elapsed = t1 - t0
            slack_compute_total += slack_elapsed
            
            # Time subsequent per-tile updates
            if device.type == "cuda":
                start_event.record()
            else:
                t2 = time.perf_counter()
            local_ind_S_zero_ind = torch.where(slack_tile_used == 0)

            # Group by local B rows (tile-local), exactly mirroring matching.py semantics
            ind_b_tent_local, group_starts = unique(local_ind_S_zero_ind[0], input_sorted=True)
            group_ends = torch.cat((group_starts[1:], group_starts.new_tensor([local_ind_S_zero_ind[0].shape[0]])))

            # Sample one zero-edge per local B
            rand_n = torch.rand(ind_b_tent_local.shape[0], device=device)
            pick = group_starts + ((group_ends - group_starts) * rand_n).to(dtyp)

            # Candidate A per local B row
            ind_a_tent = local_ind_S_zero_ind[1][pick]

            # Deduplicate A; select aligned local B rows
            ind_a_push, tent_ind = unique(ind_a_tent, input_sorted=False)
            ind_b_push_local = ind_b_tent_local[tent_ind]

            # Convert selected local B rows to global B ids only now
            ind_b_push = ind_b_free[ind_b_push_local]
            
            ind_release = torch.nonzero(Ma[ind_a_push] != -1, as_tuple=True)[0]
            edges_released = (Ma[ind_a_push][ind_release], ind_a_push[ind_release])
            
            f -= len(ind_a_push)-len(ind_release)
            
            Mb[Ma[edges_released[1]]] = minus_one
            
            edges_pushed = (ind_b_push, ind_a_push)
            Ma[ind_a_push] = ind_b_push
            Mb[ind_b_push] = ind_a_push
            yA[ind_a_push] -= one
            
            min_slack, _ = torch.min(slack_tile_used, dim=1)
            min_slack_ind = torch.where(min_slack!=0)[0]
            ind_b_not_pushed = ind_b_free[min_slack_ind]
            yB[ind_b_not_pushed] += min_slack[min_slack_ind]
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                update_elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:
                t3 = time.perf_counter()
                update_elapsed = t3 - t2
            tile_updates_total += update_elapsed
            inner_loops_count += 1

        
        iteration += 1

    yA = yA.cpu().detach()   
    yB = yB.cpu().detach()
    Ma = Ma.cpu().detach()
    Mb = Mb.cpu().detach()

    # Time final matching fill
    t4 = time.perf_counter()
    ind_a = 0
    for ind_b in range(m):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b
    t5 = time.perf_counter()
    final_fill_time = t5 - t4
    
    # Remove infeasible edges (cost >= cmax_int) from matching
    feasible_matches = m
    free_B = 0
    if cmax_int is not None:
        # Compute integerized costs for matched edges using Haversine
        xA_cpu = xA.detach().cpu()
        xB_cpu = xB.detach().cpu()
        mb = Mb.detach().cpu().numpy().astype(np.int64, copy=False)
        
        infeasible_mask = torch.zeros(m, dtype=torch.bool)
        for i in range(m):
            # Extract lat/lon (assuming [lon, lat] format)
            lat_b, lon_b = xB_cpu[i, 1], xB_cpu[i, 0]
            lat_a, lon_a = xA_cpu[mb[i], 1], xA_cpu[mb[i], 0]
            haversine_dist = haversine_distance_cpu(lat_b, lon_b, lat_a, lon_a)
            int_cost = int((3.0 * haversine_dist.item()) / (float(C) * delta))
            if int_cost >= cmax_int:
                infeasible_mask[i] = True
        
        # Remove infeasible matches
        infeasible_b = torch.where(infeasible_mask)[0]
        infeasible_a = Mb[infeasible_b]
        Mb[infeasible_b] = -1
        Ma[infeasible_a] = -1
        
        feasible_matches = m - len(infeasible_b)
        free_B = len(infeasible_b)
    
    # Time total matching cost calculation
    t6 = time.perf_counter()
    mb = Mb.detach().cpu().numpy().astype(np.int64, copy=False)
    
    # Only compute cost for feasible matches using Haversine
    matched_mask = mb != -1
    if matched_mask.any():
        # Convert to CPU tensors for Haversine calculation
        xA_cpu = xA.cpu()
        xB_cpu = xB.cpu()
        total_cost = 0.0
        for i in range(m):
            if matched_mask[i]:
                # Extract lat/lon (assuming [lon, lat] format)
                lat_b, lon_b = xB_cpu[i, 1], xB_cpu[i, 0]
                lat_a, lon_a = xA_cpu[mb[i], 1], xA_cpu[mb[i], 0]
                dist = haversine_distance_cpu(lat_b, lon_b, lat_a, lon_a)
                total_cost += dist.item()
        matching_cost = total_cost
    else:
        matching_cost = 0.0
    matching_cost = torch.as_tensor(matching_cost, dtype=torch.float64)
    t7 = time.perf_counter()
    cost_calc_time = t7 - t6
    
    # Compute averages for coarse metrics
    slack_compute_avg = slack_compute_total / inner_loops_count if inner_loops_count > 0 else 0.0
    tile_updates_avg = tile_updates_total / inner_loops_count if inner_loops_count > 0 else 0.0
    
    # Timing metrics for external A/B testing
    timing_metrics = {
        "slack_compute_total": slack_compute_total,
        "slack_compute_avg": slack_compute_avg,
        "tile_updates_total": tile_updates_total,
        "tile_updates_avg": tile_updates_avg,
        "final_fill_time": final_fill_time,
        "cost_calc_time": cost_calc_time,
        "inner_loops_count": inner_loops_count,
        "feasible_matches": feasible_matches,
        "free_B": free_B
    }

    return Mb, yA, yB, matching_cost, iteration, timing_metrics
