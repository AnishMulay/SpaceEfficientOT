import time
from typing import Optional, Tuple

import numpy as np
import torch
from cuda_deque import CudaDeque

_EARTH_RADIUS_KM = 6_371.0
_MASK_SENTINEL_INT = 10**12

_tA_global = None
_tB_global = None


def _to_float_scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def _haversine_distance(
    xB_rad: torch.Tensor,
    xA_rad: torch.Tensor,
) -> torch.Tensor:
    lon_b = xB_rad[:, 0].unsqueeze(1)
    lat_b = xB_rad[:, 1].unsqueeze(1)
    lon_a = xA_rad[:, 0].unsqueeze(0)
    lat_a = xA_rad[:, 1].unsqueeze(0)

    dlon = lon_a - lon_b
    dlat = lat_a - lat_b

    sin_dlat = torch.sin(dlat * 0.5)
    sin_dlon = torch.sin(dlon * 0.5)
    cos_lat_b = torch.cos(lat_b)
    cos_lat_a = torch.cos(lat_a)

    a = sin_dlat.square() + cos_lat_b * cos_lat_a * sin_dlon.square()
    a = torch.clamp(a, min=0.0, max=1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, min=0.0)))
    return (_EARTH_RADIUS_KM * c).to(xB_rad.dtype)


def _slack_kernel(
    xb_deg: torch.Tensor,
    xA_rad: torch.Tensor,
    yA: torch.Tensor,
    yB_idx: torch.Tensor,
    scale_factor: torch.Tensor,
    tA: torch.Tensor,
    tB_tile: torch.Tensor,
    cmax_int: Optional[int] = None,
) -> torch.Tensor:
    xb_rad = torch.deg2rad(xb_deg)
    dist = _haversine_distance(xb_rad, xA_rad)

    scaled = torch.floor(dist * scale_factor).to(torch.int64)

    mask = tB_tile.view(-1, 1) < tA.view(1, -1)
    sentinel = scaled.new_full((), _MASK_SENTINEL_INT)

    if cmax_int is not None:
        clamp_val = scaled.new_full((), int(cmax_int))
        scaled = torch.where(mask, sentinel, torch.minimum(scaled, clamp_val))
    else:
        scaled = torch.where(mask, sentinel, scaled)

    return scaled - yA.unsqueeze(0) - yB_idx.unsqueeze(1)


_compiled_slack = torch.compile(_slack_kernel, mode="reduce-overhead", dynamic=True)


def _haversine_pair_costs(
    xB_deg: torch.Tensor,
    xA_deg: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.zeros(0, device=xB_deg.device, dtype=xB_deg.dtype)
    xb = xB_deg.index_select(0, rows)
    xa = xA_deg.index_select(0, cols)
    xb_rad = torch.deg2rad(xb)
    xa_rad = torch.deg2rad(xa)
    return _haversine_distance(xb_rad, xa_rad)


def feasibility_check_full(
    xA: torch.Tensor,
    xB: torch.Tensor,
    Mb: torch.Tensor,
    Ma: torch.Tensor,
    yA: torch.Tensor,
    yB: torch.Tensor,
    delta,
    device="cpu",
    max_examples: int = 10,
) -> dict:
    with torch.no_grad():
        dtyp = torch.int64

        xA = xA.to(device=device)
        xB = xB.to(device=device)
        yA = yA.to(device=device, dtype=dtyp)
        yB = yB.to(device=device, dtype=dtyp)
        Mb = Mb.to(device=device, dtype=dtyp)
        Ma = Ma.to(device=device, dtype=dtyp)

        xA_rad = torch.deg2rad(xA)
        xB_rad = torch.deg2rad(xB)
        dist = _haversine_distance(xB_rad, xA_rad)
        C = torch.floor((3.0 * dist) / float(delta)).to(dtyp)

        Y = yA.unsqueeze(0) + yB.unsqueeze(1)
        S = C - Y

        strict_viol_mask = (Y > C)
        strict_viol = int(strict_viol_mask.sum().item())

        lax_viol_mask = (Y > (C + 1))
        lax_viol = int(lax_viol_mask.sum().item())

        min_slack = int(S.min().item())

        matched_mask = (Mb != -1)
        matched_rows = torch.where(matched_mask)[0]
        matched_cols = Mb[matched_mask]
        sum_y_matched = yA.index_select(0, matched_cols) + yB.index_select(0, matched_rows)
        c_matched = C[matched_rows, matched_cols]
        tight_mask = (sum_y_matched == c_matched)
        non_tight_matched = int((~tight_mask).sum().item())

        dup_A = int(Mb.shape[0] - torch.unique(Mb).numel())
        reciprocity_ok = bool(torch.all(Ma.index_select(0, matched_cols) == matched_rows))

        examples = []
        if lax_viol > 0:
            bi, aj = torch.nonzero(lax_viol_mask, as_tuple=True)
            take = min(max_examples, bi.shape[0])
            for t in range(take):
                b = int(bi[t].item())
                a = int(aj[t].item())
                examples.append((b, a, int(S[b, a].item())))

        report = {
            "min_slack": min_slack,
            "strict_violations": strict_viol,
            "epsilon_violations": lax_viol,
            "non_tight_matched": non_tight_matched,
            "dup_A": dup_A,
            "reciprocity_ok": reciprocity_ok,
            "examples": examples,
        }

        print(
            f"[feas] min_slack={min_slack}, "
            f"strict_viol={strict_viol}, "
            f"eps_viol={lax_viol}, "
            f"non_tight_matched={non_tight_matched}, "
            f"dup_A={dup_A}, reciprocity_ok={reciprocity_ok}"
        )

        return report


def unique(x, input_sorted: bool = False):
    unique_vals, inverse_ind, unique_count = torch.unique(x, return_inverse=True, return_counts=True)
    unique_ind = unique_count.cumsum(0)
    if unique_ind.size(0) != 0:
        unique_ind = torch.cat((torch.tensor([0], dtype=x.dtype, device=x.device), unique_ind[:-1]))
    if not input_sorted:
        _, sort2ori_ind = torch.sort(inverse_ind, stable=True)
        unique_ind = sort2ori_ind[unique_ind]
    return unique_vals, unique_ind


def _mask_times(
    idxB: torch.Tensor,
    tile_times: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    tA_all, tB_all = tile_times
    tB_tile = tB_all.index_select(0, idxB)
    return tA_all, tB_tile


def compute_slack_tile(
    idxB: torch.Tensor,
    xA: torch.Tensor,
    xB: torch.Tensor,
    yA: torch.Tensor,
    yB: torch.Tensor,
    C,
    delta,
    slack_tile: Optional[torch.Tensor] = None,
    tile_times: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    xA_rad: Optional[torch.Tensor] = None,
    cmax_int: Optional[int] = None,
) -> torch.Tensor:
    current_k = idxB.numel()
    if current_k == 0:
        return torch.empty((0, xA.shape[0]), dtype=torch.int64, device=xA.device)

    xb = xB.index_select(0, idxB).to(dtype=xA.dtype)
    yB_idx = yB.index_select(0, idxB)

    if xA_rad is None:
        xA_rad = torch.deg2rad(xA)

    scale_factor = 3.0 / (_to_float_scalar(C) * _to_float_scalar(delta))
    scale_tensor = xb.new_full((), scale_factor)

    if tile_times is None:
        tA_mask = yA.new_full((xA.shape[0],), torch.iinfo(torch.int64).max)
        tB_tile = yB_idx.new_zeros(current_k, dtype=torch.int64)
    else:
        tA_mask, tB_tile = _mask_times(idxB, tile_times)

    slack = _compiled_slack(
        xb,
        xA_rad,
        yA,
        yB_idx,
        scale_tensor,
        tA_mask,
        tB_tile,
        cmax_int,
    )

    if slack_tile is not None and current_k <= slack_tile.shape[0]:
        reuse = slack_tile[:current_k]
        reuse.copy_(slack)
        return reuse

    return slack


def spef_matching_torch(
    xA: torch.Tensor,
    xB: torch.Tensor,
    C,
    k: int,
    delta,
    device,
    seed: int = 1,
    tA: Optional[torch.Tensor] = None,
    tB: Optional[torch.Tensor] = None,
    cmax_int: Optional[int] = None,
):
    return spef_matching_2(
        xA=xA,
        xB=xB,
        C=C,
        k=k,
        delta=delta,
        device=device,
        seed=seed,
        tA=tA,
        tB=tB,
        cmax_int=cmax_int,
    )


def spef_matching_2(
    xA: torch.Tensor,
    xB: torch.Tensor,
    C,
    k: int,
    delta,
    device,
    seed: int = 1,
    tA: Optional[torch.Tensor] = None,
    tB: Optional[torch.Tensor] = None,
    cmax_int: Optional[int] = None,
    stopping_condition: Optional[float] = None,
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

    f_threshold = stopping_condition if stopping_condition is not None else m * delta / C
    torch.manual_seed(seed)

    slack_tile = torch.zeros(k, n, device=device, dtype=torch.int64)
    xA_rad = torch.deg2rad(xA)
    if _tA_global is not None and _tB_global is not None:
        times_tuple = (_tA_global, _tB_global)
    else:
        tA_default = torch.full(
            (n,),
            torch.iinfo(torch.int64).max,
            device=device,
            dtype=torch.int64,
        )
        tB_default = torch.zeros(m, device=device, dtype=torch.int64)
        times_tuple = (tA_default, tB_default)

    slack_compute_total = 0.0
    tile_updates_total = 0.0
    inner_loops_count = 0

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    while f > f_threshold:
        ind_b_all_free = torch.where(Mb == minus_one)[0]

        for start_idx in range(0, len(ind_b_all_free), k):
            end_idx = min(start_idx + k, len(ind_b_all_free))
            ind_b_free = ind_b_all_free[start_idx:end_idx]

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
                slack_tile=slack_tile,
                tile_times=times_tuple,
                xA_rad=xA_rad,
                cmax_int=cmax_int,
            )
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                slack_compute_total += start_event.elapsed_time(end_event) / 1000.0
            else:
                t1 = time.perf_counter()
                slack_compute_total += t1 - t0

            if device.type == "cuda":
                start_event.record()
            else:
                t2 = time.perf_counter()
            local_ind_S_zero_ind = torch.where(slack_tile_used == 0)

            ind_b_tent_local, group_starts = unique(local_ind_S_zero_ind[0], input_sorted=True)
            group_ends = torch.cat((group_starts[1:], group_starts.new_tensor([local_ind_S_zero_ind[0].shape[0]])))

            rand_n = torch.rand(ind_b_tent_local.shape[0], device=device)
            pick = group_starts + ((group_ends - group_starts) * rand_n).to(dtyp)

            ind_a_tent = local_ind_S_zero_ind[1][pick]

            ind_a_push, tent_ind = unique(ind_a_tent, input_sorted=False)
            ind_b_push_local = ind_b_tent_local[tent_ind]

            ind_b_push = ind_b_free[ind_b_push_local]

            ind_release = torch.nonzero(Ma[ind_a_push] != -1, as_tuple=True)[0]
            edges_released = (Ma[ind_a_push][ind_release], ind_a_push[ind_release])

            f -= len(ind_a_push) - len(ind_release)

            Mb[Ma[edges_released[1]]] = minus_one

            Ma[ind_a_push] = ind_b_push
            Mb[ind_b_push] = ind_a_push
            yA[ind_a_push] -= one

            min_slack, _ = torch.min(slack_tile_used, dim=1)
            min_slack_ind = torch.where(min_slack != 0)[0]
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

    ind_a = 0
    for ind_b in range(m):
        if Mb[ind_b] == -1:
            while Ma[ind_a] != -1:
                ind_a += 1
            Mb[ind_b] = ind_a
            Ma[ind_a] = ind_b

    xb_cpu = xB.detach().cpu()
    xa_cpu = xA.detach().cpu()

    scale_factor = 3.0 / (_to_float_scalar(C) * _to_float_scalar(delta))
    if cmax_int is not None:
        matched_tensor = Mb != -1
        if matched_tensor.any():
            rows = torch.nonzero(matched_tensor, as_tuple=False).squeeze(1)
            cols = Mb.index_select(0, rows)
            xb_sel = xb_cpu.index_select(0, rows)
            xa_sel = xa_cpu.index_select(0, cols)
            xb_rad = torch.deg2rad(xb_sel)
            xa_rad = torch.deg2rad(xa_sel)
            dist = _haversine_distance(xb_rad, xa_rad)
            int_costs = torch.floor(dist * scale_factor).to(torch.int64)
            infeasible = int_costs >= int(cmax_int)
            if infeasible.any():
                infeasible_rows = rows[infeasible]
                infeasible_cols = cols[infeasible]
                Mb[infeasible_rows] = -1
                Ma[infeasible_cols] = -1
        feasible_matches = int((Mb != -1).sum().item())
        free_B = m - feasible_matches
    else:
        feasible_matches = m
        free_B = 0

    mb = Mb.detach().cpu().numpy().astype(np.int64, copy=False)
    matched_mask = mb != -1
    if matched_mask.any():
        rows = torch.from_numpy(np.where(matched_mask)[0]).to(device=xB.device, dtype=torch.int64)
        cols = torch.from_numpy(mb[matched_mask]).to(device=xA.device, dtype=torch.int64)
        costs = _haversine_pair_costs(xB, xA, rows, cols)
        matching_cost = costs.sum(dtype=torch.float64)
    else:
        matching_cost = torch.tensor(0.0, dtype=torch.float64)
    matching_cost = matching_cost.cpu()

    slack_compute_avg = slack_compute_total / inner_loops_count if inner_loops_count > 0 else 0.0
    tile_updates_avg = tile_updates_total / inner_loops_count if inner_loops_count > 0 else 0.0

    timing_metrics = {
        "slack_compute_total": slack_compute_total,
        "slack_compute_avg": slack_compute_avg,
        "tile_updates_total": tile_updates_total,
        "tile_updates_avg": tile_updates_avg,
        "feasible_matches": feasible_matches,
        "free_B": free_B,
        "inner_loops_count": inner_loops_count,
    }

    return Mb, yA, yB, matching_cost, iteration, timing_metrics
