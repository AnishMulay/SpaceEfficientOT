import numpy as np
import torch
import time
from cuda_deque import CudaDeque
from scipy.spatial.distance import cdist
from scipy.spatial.distance import sqeuclidean

def feasibility_check_full(
    xA,
    xB,
    Mb,
    Ma,
    yA,
    yB,
    delta,
    device="cpu",
    max_examples=10
):
    """
    Full-matrix feasibility checker (debug-only, not space-efficient).

    Conditions checked (paper's ε-feasible with ε mapped to integer 1 under scaling):
      - For all edges (b,a) not necessarily in M: yA[a] + yB[b] <= C[b,a] + 1
      - For matched edges (b, Mb[b]): yA[Mb[b]] + yB[b] == C[b, Mb[b]]
    Also reports strict violations with +0, min slack, reciprocity, duplicates, and sample edges.

    Returns a dict report and prints a one-line summary.
    """
    with torch.no_grad():
        dtyp = torch.int64

        # Move to device and dtypes
        xA = xA.to(device=device)
        xB = xB.to(device=device)
        yA = yA.to(device=device, dtype=dtyp)
        yB = yB.to(device=device, dtype=dtyp)
        Mb = Mb.to(device=device, dtype=dtyp)
        Ma = Ma.to(device=device, dtype=dtyp)

        m = xB.shape[0]
        n = xA.shape[0]

        # Full integerized cost matrix: C[b,a] = floor(3 * ||xB[b]-xA[a]||^2 / delta)
        xb2 = (xB * xB).sum(dim=1, keepdim=True)       # [m,1]
        xa2 = (xA * xA).sum(dim=1, keepdim=True).T     # [1,n]
        C_float = xb2 + xa2 - 2.0 * (xB @ xA.T)        # [m,n]
        C = torch.floor((3.0 * C_float) / float(delta)).to(dtyp)  # [m,n]

        # Broadcasted dual sum Y[b,a] = yA[a] + yB[b]
        Y = yA.unsqueeze(0) + yB.unsqueeze(1)          # [m,n] int64

        # Slack matrix S = C - Y
        S = C - Y

        # Violations
        # Strict feasibility: Y <= C  (S >= 0)
        strict_viol_mask = (Y > C)
        strict_viol = int(strict_viol_mask.sum().item())

        # ε-feasibility per paper (ε maps to +1): Y <= C + 1  (S >= -1)
        lax_viol_mask = (Y > (C + 1))
        lax_viol = int(lax_viol_mask.sum().item())

        min_slack = int(S.min().item())

        # Matched-edge tightness: Y[b, Mb[b]] == C[b, Mb[b]]  (S[b, Mb[b]] == 0)
        # Only check matched points (Mb[b] != -1)
        matched_mask = (Mb != -1)
        matched_rows = torch.where(matched_mask)[0]
        matched_cols = Mb[matched_mask]
        sum_y_matched = yA.index_select(0, matched_cols) + yB.index_select(0, matched_rows)
        c_matched = C[matched_rows, matched_cols]
        tight_mask = (sum_y_matched == c_matched)
        non_tight_matched = int((~tight_mask).sum().item())

        # Matching integrity diagnostics
        dup_A = int(m - torch.unique(Mb).numel())      # duplicates on A side
        reciprocity_ok = bool(torch.all(Ma.index_select(0, matched_cols) == matched_rows))

        # Example violating edges for debugging (ε-feasibility)
        examples = []
        if lax_viol > 0:
            bi, aj = torch.nonzero(lax_viol_mask, as_tuple=True)
            take = min(max_examples, bi.shape)
            for t in range(take):
                b = int(bi[t].item()); a = int(aj[t].item())
                examples.append((b, a, int(S[b, a].item())))  # negative slack values

        report = {
            "min_slack": min_slack,                # minimum C-Y over all edges
            "strict_violations": strict_viol,      # count where Y > C
            "epsilon_violations": lax_viol,        # count where Y > C+1
            "non_tight_matched": non_tight_matched,
            "dup_A": dup_A,
            "reciprocity_ok": reciprocity_ok,
            "examples": examples,                  # list of (b,a, slack) with slack < -1
        }

        print(
            f"[feas] min_slack={min_slack}, "
            f"strict_viol={strict_viol}, "
            f"eps_viol={lax_viol}, "
            f"non_tight_matched={non_tight_matched}, "
            f"dup_A={dup_A}, reciprocity_ok={reciprocity_ok}"
        )

        return report

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
    delta,
    slack_tile=None,
    xAT=None,
    xa2_cached=None,
    xb2_all=None
):
    current_k = len(idxB)
    
    if slack_tile is not None and current_k <= slack_tile.shape[0]:
        # Reuse pre-allocated tensor (fully overwritten by copy_)
        slack_view = slack_tile[:current_k]
        
        xb = xB.index_select(0, idxB).to(dtype=xA.dtype)               # [K,d]
        if xb2_all is not None:
            xb2 = xb2_all.index_select(0, idxB)                        # [K,1]
        else:
            xb2 = (xb*xb).sum(dim=1, keepdim=True)                     # [K,1]
        if xa2_cached is not None:
            xa2 = xa2_cached                                           # [1,N]
        else:
            xa2 = (xA*xA).sum(dim=1, keepdim=True).T                   # [1,N]
        if xAT is not None:
            cross = xb @ xAT                                           # [K,N]
        else:
            cross = xb @ xA.T                                          # [K,N]
        w_tile = xb2 + xa2 - 2.0*cross                                 # [K,N] float
        c_tile = torch.floor((3.0*w_tile)/float(delta))                # [K,N] float
        c_tile = c_tile.to(dtype=torch.int64)                          # [K,N] int64
        slack_view.copy_(c_tile - yA.unsqueeze(0) - yB.index_select(0, idxB).unsqueeze(1))
        return slack_view                                              # [current_k,N] int64
    else:
        # Original allocation for fallback
        xb = xB.index_select(0, idxB).to(dtype=xA.dtype)               # [K,d]
        if xb2_all is not None:
            xb2 = xb2_all.index_select(0, idxB)                        # [K,1]
        else:
            xb2 = (xb*xb).sum(dim=1, keepdim=True)                     # [K,1]
        if xa2_cached is not None:
            xa2 = xa2_cached                                           # [1,N]
        else:
            xa2 = (xA*xA).sum(dim=1, keepdim=True).T                   # [1,N]
        if xAT is not None:
            cross = xb @ xAT                                           # [K,N]
        else:
            cross = xb @ xA.T                                          # [K,N]
        w_tile = xb2 + xa2 - 2.0*cross                                 # [K,N] float
        c_tile = torch.floor((3.0*w_tile)/float(delta))                # [K,N] float
        c_tile = c_tile.to(dtype=torch.int64)                          # [K,N] int64
        slack = c_tile - yA.unsqueeze(0) - yB.index_select(0, idxB).unsqueeze(1)
        return slack                                                   # [K,N] int64


def spef_matching_torch(
    xA,
    xB,
    C,
    k,
    delta,
    device,
    seed=1
):
    dtyp = torch.int64
    n = xA.shape[0]
    m = xB.shape[0]

    xA = xA.to(device=device)
    xB = xB.to(device=device)

    yB = torch.ones(m, device=device, dtype=dtyp, requires_grad=False)
    yA = torch.zeros(n, device=device, dtype=dtyp, requires_grad=False)
    Mb = torch.ones(m, device=device, dtype=dtyp, requires_grad=False) * -1
    Ma = torch.ones(n, device=device, dtype=dtyp, requires_grad=False) * -1

    f = n
    iteration = 0

    n = xA.shape[0]
    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    minus_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    f_threshold = m*delta/C
    torch.manual_seed(seed)

    dq = CudaDeque(max_size=m, device=device, dtype=dtyp)
    ind_b_free = torch.where(Mb == minus_one)[0]
    dq.push_back(ind_b_free)

    while f > f_threshold:
        ind_b_free = dq.pop_front(k)
        slack_tile = compute_slack_tile(ind_b_free, xA, xB, yA, yB, delta)
        local_ind_S_zero_ind = torch.where(slack_tile == 0)  # (rows in 0..K-1, cols in 0..N-1)

        # Group by local B rows (tile-local), exactly mirroring matching.py semantics
        ind_b_tent_local, group_starts = unique(local_ind_S_zero_ind[0], input_sorted=True)
        group_ends = torch.cat((group_starts[1:], group_starts.new_tensor([local_ind_S_zero_ind[0].shape[0]])))  # right-exclusive

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
        
        min_slack, _ = torch.min(slack_tile, dim=1)
        min_slack_ind = torch.where(min_slack!=0)[0]
        ind_b_not_pushed = ind_b_free[min_slack_ind]
        yB[ind_b_not_pushed] += min_slack[min_slack_ind]
        
        dq.push_back(edges_released[0])
        dq.push_back(ind_b_not_pushed)
        
        iteration += 1
    
    yA = yA.cpu().detach()   
    yB = yB.cpu().detach()
    Ma = Ma.cpu().detach()
    Mb = Mb.cpu().detach()
    
    ind_a = 0
    for ind_b in range(m):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b
    
    xa64 = xA.detach().cpu().numpy().astype(np.float64, copy=False)  # N x d
    xb64 = xB.detach().cpu().numpy().astype(np.float64, copy=False)  # M x d
    mb   = Mb.detach().cpu().numpy().astype(np.int64,   copy=False)  # M

    # Streaming per-pair sqeuclidean in float64 (no full matrix)
    matching_cost = np.add.reduce(
        [sqeuclidean(xb64[i], xa64[mb[i]]) for i in range(xb64.shape[0])],
        dtype=np.float64
    )
    matching_cost = torch.as_tensor(matching_cost, dtype=torch.float64)

    return Mb, yA, yB, matching_cost, iteration

def spef_matching_2(
    xA,
    xB,
    C,
    k,
    delta,
    device,
    seed=1
):
    dtyp = torch.int64
    n = xA.shape[0]
    m = xB.shape[0]

    xA = xA.to(device=device)
    xB = xB.to(device=device)

    yB = torch.ones(m, device=device, dtype=dtyp, requires_grad=False)
    yA = torch.zeros(n, device=device, dtype=dtyp, requires_grad=False)
    Mb = torch.ones(m, device=device, dtype=dtyp, requires_grad=False) * -1
    Ma = torch.ones(n, device=device, dtype=dtyp, requires_grad=False) * -1

    f = n
    iteration = 0

    zero = torch.tensor([0], device=device, dtype=dtyp, requires_grad=False)[0]
    one = torch.tensor([1], device=device, dtype=dtyp, requires_grad=False)[0]
    minus_one = torch.tensor([-1], device=device, dtype=dtyp, requires_grad=False)[0]

    f_threshold = m*delta/C
    torch.manual_seed(seed)
    
    # Compute one-time caches
    xAT = xA.T.contiguous()
    xa2_cached = (xA * xA).sum(dim=1, keepdim=True).T
    xb2_all = (xB * xB).sum(dim=1, keepdim=True)
    
    # Pre-allocate slack tile for reuse
    slack_tile = torch.zeros(k, n, device=device, dtype=torch.int64)
    
    # Timing metrics
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
            slack_tile_used = compute_slack_tile(ind_b_free, xA, xB, yA, yB, delta, slack_tile, xAT=xAT, xa2_cached=xa2_cached, xb2_all=xb2_all)
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
    
    # Time total matching cost calculation
    t6 = time.perf_counter()
    xa64 = xA.detach().cpu().numpy().astype(np.float64, copy=False)  # N x d
    xb64 = xB.detach().cpu().numpy().astype(np.float64, copy=False)  # M x d
    mb   = Mb.detach().cpu().numpy().astype(np.int64,   copy=False)  # M

    # Streaming per-pair sqeuclidean in float64 (no full matrix)
    matching_cost = np.add.reduce(
        [sqeuclidean(xb64[i], xa64[mb[i]]) for i in range(xb64.shape[0])],
        dtype=np.float64
    )
    matching_cost = torch.as_tensor(matching_cost, dtype=torch.float64)
    t7 = time.perf_counter()
    cost_calc_time = t7 - t6
    
    # Compute averages
    slack_compute_avg = slack_compute_total / inner_loops_count if inner_loops_count > 0 else 0.0
    tile_updates_avg = tile_updates_total / inner_loops_count if inner_loops_count > 0 else 0.0
    
    timing_metrics = {
        "slack_compute_total": slack_compute_total,
        "slack_compute_avg": slack_compute_avg,
        "tile_updates_total": tile_updates_total,
        "tile_updates_avg": tile_updates_avg,
        "final_fill_time": final_fill_time,
        "cost_calc_time": cost_calc_time,
        "inner_loops_count": inner_loops_count
    }

    return Mb, yA, yB, matching_cost, iteration, timing_metrics