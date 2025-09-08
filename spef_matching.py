import numpy as np
import torch
from cuda_deque import CudaDeque

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
    delta
):
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
        local_ind_S_zero_ind = torch.where(slack_tile == 0)
        global_ind_S_zero_ind = (ind_b_free[local_ind_S_zero_ind[0]], local_ind_S_zero_ind[1])
        
        if iteration == 0:
            print(f"SPEF - local_ind_S_zero_ind[0] type: {type(local_ind_S_zero_ind[0])}, shape: {local_ind_S_zero_ind[0].shape}, dtype: {local_ind_S_zero_ind[0].dtype}")
            print(f"SPEF - local_ind_S_zero_ind[1] type: {type(local_ind_S_zero_ind[1])}, shape: {local_ind_S_zero_ind[1].shape}, dtype: {local_ind_S_zero_ind[1].dtype}")
            print(f"SPEF - global_ind_S_zero_ind[0] type: {type(global_ind_S_zero_ind[0])}, shape: {global_ind_S_zero_ind[0].shape}, dtype: {global_ind_S_zero_ind[0].dtype}")
            print(f"SPEF - global_ind_S_zero_ind[1] type: {type(global_ind_S_zero_ind[1])}, shape: {global_ind_S_zero_ind[1].shape}, dtype: {global_ind_S_zero_ind[1].dtype}")
            print(f"SPEF - ind_b_free type: {type(ind_b_free)}, shape: {ind_b_free.shape}, dtype: {ind_b_free.dtype}")

        ind_b_tent_ind, free_S_edge_B_ind_range_lt_inclusive = unique(global_ind_S_zero_ind[0], input_sorted=True)
        ind_b_tent = ind_b_free[ind_b_tent_ind]
        
        if iteration == 0:
            print(f"SPEF - free_S_edge_B_ind_range_lt_inclusive: {free_S_edge_B_ind_range_lt_inclusive}")
            print(f"SPEF - free_S_edge_B_ind_range_lt_inclusive[1:]: {free_S_edge_B_ind_range_lt_inclusive[1:]}")
            shape_val = global_ind_S_zero_ind[0].shape[0]
            print(f"SPEF - shape_val: {shape_val}, type: {type(shape_val)}")
            tensor_to_cat = torch.tensor([global_ind_S_zero_ind[0].shape[0]], device=device, dtype=dtyp, requires_grad=False)
            print(f"SPEF - tensor_to_cat: {tensor_to_cat}")
            print(f"SPEF - About to concatenate: {free_S_edge_B_ind_range_lt_inclusive[1:]} with {tensor_to_cat}")
        free_S_edge_B_ind_range_rt_exclusive = torch.cat((free_S_edge_B_ind_range_lt_inclusive[1:], torch.tensor([global_ind_S_zero_ind[0].shape[0]], device=device, dtype=dtyp, requires_grad=False)))
        rand_n = torch.rand(ind_b_tent.shape[0], device=device)
        free_S_edge_B_ind_rand = free_S_edge_B_ind_range_lt_inclusive + ((free_S_edge_B_ind_range_rt_exclusive - free_S_edge_B_ind_range_lt_inclusive)*rand_n).to(dtyp)
        ind_a_tent = global_ind_S_zero_ind[1][free_S_edge_B_ind_rand]
        ind_a_push, tent_ind = unique(ind_a_tent, input_sorted=False)
        ind_b_push = ind_b_tent[tent_ind]
        
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
    for ind_b in range(n):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b
    
    xA_cpu = xA.cpu().float()
    xB_cpu = xB.cpu().float()
    matched_pairs_cost = torch.sum((xA_cpu - xB_cpu[Mb])**2, dim=1)
    matching_cost = torch.sum(matched_pairs_cost)

    return Mb, yA, yB, matching_cost, iteration