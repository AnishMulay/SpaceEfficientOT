import numpy as np
import torch
from cuda_deque import CudaDeque
from scipy.spatial.distance import cdist

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
    slack_tile=None
):
    current_k = len(idxB)
    
    if slack_tile is not None and current_k <= slack_tile.shape[0]:
        # Reuse pre-allocated tensor
        slack_view = slack_tile[:current_k]
        slack_view.zero_()
        
        xb = xB.index_select(0, idxB).to(dtype=xA.dtype)               # [K,d]
        xb2 = (xb*xb).sum(dim=1, keepdim=True)                         # [K,1]
        xa2 = (xA*xA).sum(dim=1, keepdim=True).T                       # [1,N]
        cross = xb @ xA.T                                              # [K,N]
        w_tile = xb2 + xa2 - 2.0*cross                                 # [K,N] float
        c_tile = torch.floor((3.0*w_tile)/float(delta))                # [K,N] float
        c_tile = c_tile.to(dtype=torch.int64)                          # [K,N] int64
        slack_view.copy_(c_tile - yA.unsqueeze(0) - yB.index_select(0, idxB).unsqueeze(1))
        return slack_view                                              # [current_k,N] int64
    else:
        # Original allocation for fallback
        xb = xB.index_select(0, idxB).to(dtype=xA.dtype)               # [K,d]
        xb2 = (xb*xb).sum(dim=1, keepdim=True)                         # [K,1]
        xa2 = (xA*xA).sum(dim=1, keepdim=True).T                       # [1,N]
        cross = xb @ xA.T                                              # [K,N]
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
    
    xA_cpu = xA.cpu().float()
    xB_cpu = xB.cpu().float()
    W_cpu = cdist(xB_cpu.numpy(), xA_cpu.numpy(), 'sqeuclidean')
    W_tensor = torch.tensor(W_cpu, dtype=torch.float32)
    matching_cost = torch.sum(W_tensor[torch.arange(m), Mb])

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
    
    # Pre-allocate slack tile for reuse
    slack_tile = torch.zeros(k, n, device=device, dtype=torch.int64)

    while f > f_threshold:
        # Get all free B points
        ind_b_all_free = torch.where(Mb == minus_one)[0]
        
        # Process k points at a time
        for start_idx in range(0, len(ind_b_all_free), k):
            end_idx = min(start_idx + k, len(ind_b_all_free))
            ind_b_free = ind_b_all_free[start_idx:end_idx]
            
            slack_tile_used = compute_slack_tile(ind_b_free, xA, xB, yA, yB, delta, slack_tile)
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
    
    xA_cpu = xA.cpu().float()
    xB_cpu = xB.cpu().float()
    W_cpu = cdist(xB_cpu.numpy(), xA_cpu.numpy(), 'sqeuclidean')
    W_tensor = torch.tensor(W_cpu, dtype=torch.float32)
    matching_cost = torch.sum(W_tensor[torch.arange(m), Mb])

    return Mb, yA, yB, matching_cost, iteration