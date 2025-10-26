from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .loader import ColumnMapping


def _to_unix_seconds(series: pd.Series) -> np.ndarray:
    if series.dt.tz is not None:
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return (series.view("int64") // 10**9).to_numpy(dtype=np.int64)


def prepare_tensors(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert filtered NYC taxi DataFrame to solver tensors."""

    pickup_coords = df[[mapping.pickup_lon, mapping.pickup_lat]].to_numpy(dtype=np.float32)
    dropoff_coords = df[[mapping.dropoff_lon, mapping.dropoff_lat]].to_numpy(dtype=np.float32)

    pickup_times = _to_unix_seconds(df[mapping.pickup_time])
    dropoff_times = _to_unix_seconds(df[mapping.dropoff_time])

    xB = torch.from_numpy(pickup_coords).to(device=device, dtype=dtype)
    xA = torch.from_numpy(dropoff_coords).to(device=device, dtype=dtype)

    tB = torch.from_numpy(pickup_times).to(device=device, dtype=torch.int64)
    tA = torch.from_numpy(dropoff_times).to(device=device, dtype=torch.int64)

    return xA, xB, tA, tB
