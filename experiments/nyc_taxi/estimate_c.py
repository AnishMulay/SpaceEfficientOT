from __future__ import annotations

from typing import Optional

import torch

EARTH_RADIUS_METERS = 6_371_000.0


def _haversine_distance(xb_rad: torch.Tensor, xa_rad: torch.Tensor) -> torch.Tensor:
    lon_b = xb_rad[:, 0].unsqueeze(1)
    lat_b = xb_rad[:, 1].unsqueeze(1)
    lon_a = xa_rad[:, 0].unsqueeze(0)
    lat_a = xa_rad[:, 1].unsqueeze(0)

    dlon = lon_a - lon_b
    dlat = lat_a - lat_b

    sin_dlat = torch.sin(dlat * 0.5)
    sin_dlon = torch.sin(dlon * 0.5)
    cos_lat_b = torch.cos(lat_b)
    cos_lat_a = torch.cos(lat_a)

    a = sin_dlat.square() + cos_lat_b * cos_lat_a * sin_dlon.square()
    a = torch.clamp(a, 0.0, 1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(torch.clamp(1.0 - a, 0.0, 1.0)))
    return EARTH_RADIUS_METERS * c


@torch.no_grad()
def estimate_c(
    xA: torch.Tensor,
    xB: torch.Tensor,
    *,
    sample_size: int = 64,
    seed: int = 1,
    multiplier: float = 4.0,
) -> float:
    """Estimate scaling constant C using sampled Haversine distances."""

    device = xA.device
    dtype = xA.dtype

    n = xB.shape[0]
    if n == 0:
        raise ValueError("xB is empty")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    sample_size = min(sample_size, n)
    indices = torch.randperm(n, generator=generator, device=device)[:sample_size]

    xb_sample = xB.index_select(0, indices)

    xb_rad = torch.deg2rad(xb_sample.to(dtype=torch.float64))
    xa_rad = torch.deg2rad(xA.to(dtype=torch.float64))

    dist = _haversine_distance(xb_rad, xa_rad)
    max_dist = dist.max().item()

    return float(multiplier * max_dist)
