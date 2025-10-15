import numpy as np
import torch
import time
from cuda_deque import CudaDeque
from scipy.spatial.distance import cdist
from scipy.spatial.distance import sqeuclidean
from haversine_utils import fused_slack_haversine, haversine_distance_cpu

# Global times for NYC mask
_tA_global = None
_tB_global = None

# Compiled slack kernel for Haversine computation
def _slack_kernel_haversine(latB, lonB, latA, lonA, yA, yB_idx, C, delta, cmax_int=None, R=6371000.0):
    """Haversine-based slack computation"""
    # Calculate Haversine distances
    if latB.device.type == "cuda":
        # Use GPU-optimized version when available
        try:
            K, N = latB.shape[0], latA.shape[0]
            slack_out = torch.empty(K, N, device=latB.device, dtype=torch.float32)
            fused_slack_haversine(
                latA, lonA, latB, lonB, 
                yA.float(), yB_idx.float(), slack_out,
                use_meters=True, R=R, delta=delta
            )
            # Convert to int64 and apply scaling
            scaled = (3.0 * slack_out) / (float(C) * float(delta))
            c_tile = torch.floor(scaled).to(torch.int64)
            if cmax_int is not None:
                c_tile = torch.clamp_max(c_tile, int(cmax_int))
            return c_tile
        except:
            # Fallback to CPU implementation
            pass
    
    # CPU fallback or when GPU version fails
    K, N = latB.shape[0], latA.shape[0]
    distances = torch.zeros(K, N, device=latB.device, dtype=torch.float32)
    
    for i in range(K):
        for j in range(N):
            dist = haversine_distance_cpu(latB[i], lonB[i], latA[j], lonA[j], R)
            distances[i, j] = dist
    
    # Scale and convert to int64
    scaled = (3.0 * distances) / (float(C) * float(delta))
    c_tile = torch.floor(scaled).to(torch.int64)
    if cmax_int is not None:
        c_tile = torch.clamp_max(c_tile, int(cmax_int))
    
    # Subtract duals
    return c_tile - yA.unsqueeze(0) - yB_idx.unsqueeze(1)

_compiled_slack = torch.compile(_slack_kernel_haversine, mode='reduce-overhead', dynamic=True)


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
    tile_times=None,
    xAT=None,
    xa2_cached=None,
    cmax_int=None
):
    current_k = len(idxB)
    
    if slack_tile is not None and current_k <= slack_tile.shape[0]:
        # Reuse pre-allocated tensor - use compiled fused kernel
        slack_view = slack_tile[:current_k]
        
        # Extract lat/lon coordinates
        latB = xB.index_select(0, idxB)[:, 1]  # latitude is second column
        lonB = xB.index_select(0, idxB)[:, 0]  # longitude is first column
        latA = xA[:, 1]  # latitude is second column
        lonA = xA[:, 0]  # longitude is first column
        yB_idx = yB.index_select(0, idxB)      # [K]
        
        # Call compiled Haversine kernel
        slack_int64 = _compiled_slack(latB, lonB, latA, lonA, yA, yB_idx, C, delta, cmax_int)
        slack_view.copy_(slack_int64)

        # === NYC time mask (vectorized on GPU) ===
        times = tile_times if tile_times is not None else ((_tA_global), (_tB_global))
        if times is not None and times[0] is not None and times[1] is not None:
            tA_all, tB_all = times  # int64 [N], [N]
            tB_sel = tB_all.index_select(0, idxB)  # [K]
            invalid = tB_sel.view(-1, 1) < tA_all.view(1, -1)  # [K,N] bool
            bigM = torch.tensor(10**12, device=slack_view.device, dtype=slack_view.dtype)
            slack_view.masked_fill_(invalid, bigM)
        # === end time mask ===
        return slack_view  # [current_k,N] int64
    else:
        # Fallback branch - use compiled fused kernel
        # Extract lat/lon coordinates
        latB = xB.index_select(0, idxB)[:, 1]  # latitude is second column
        lonB = xB.index_select(0, idxB)[:, 0]  # longitude is first column
        latA = xA[:, 1]  # latitude is second column
        lonA = xA[:, 0]  # longitude is first column
        yB_idx = yB.index_select(0, idxB)      # [K]
        
        # Call compiled Haversine kernel
        slack = _compiled_slack(latB, lonB, latA, lonA, yA, yB_idx, C, delta, cmax_int)

        # === NYC time mask (vectorized on GPU) ===
        times = tile_times if tile_times is not None else ((_tA_global), (_tB_global))
        if times is not None and times[0] is not None and times[1] is not None:
            tA_all, tB_all = times  # int64 [N], [N]
            tB_sel = tB_all.index_select(0, idxB)  # [K]
            invalid = tB_sel.view(-1, 1) < tA_all.view(1, -1)  # [K,N] bool
            bigM = torch.tensor(10**12, device=slack.device, dtype=slack.dtype)
            slack.masked_fill_(invalid, bigM)
        # === end time mask ===
        return slack  # [K,N] int64


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

    f_threshold = stopping_condition if stopping_condition is not None else m*delta/C
    torch.manual_seed(seed)
    
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
        
        # Process k points at a time
        for start_idx in range(0, len(ind_b_all_free), k):
            end_idx = min(start_idx + k, len(ind_b_all_free))
            ind_b_free = ind_b_all_free[start_idx:end_idx]
            
            # Time slack tile compute
            if device.type == "cuda":
                start_event.record()
            else:
                t0 = time.perf_counter()
            slack_tile_used = compute_slack_tile(ind_b_free, xA, xB, yA, yB, C, delta, slack_tile, xAT=xAT, xa2_cached=xa2_cached, cmax_int=cmax_int)
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                slack_compute_total += start_event.elapsed_time(end_event) / 1000.0
            else:
                t1 = time.perf_counter()
                slack_compute_total += t1 - t0
            
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
                tile_updates_total += start_event.elapsed_time(end_event) / 1000.0
            else:
                t3 = time.perf_counter()
                tile_updates_total += t3 - t2
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