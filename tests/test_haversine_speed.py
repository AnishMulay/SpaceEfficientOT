from __future__ import annotations

import math

import pytest
import torch

import spef_ot.kernels.haversine_speed  # noqa: F401 - ensure kernel registration
from spef_ot import match


EARTH_RADIUS_METERS = 6_371_000.0


def _haversine_distance(row: torch.Tensor, col: torch.Tensor) -> float:
    lon_b, lat_b = map(math.radians, row.tolist())
    lon_a, lat_a = map(math.radians, col.tolist())
    dlon = lon_a - lon_b
    dlat = lat_a - lat_b
    sin_dlat = math.sin(dlat * 0.5)
    sin_dlon = math.sin(dlon * 0.5)
    a = sin_dlat * sin_dlat + math.cos(lat_b) * math.cos(lat_a) * sin_dlon * sin_dlon
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c


def test_time_reachability_and_ymax_finalize() -> None:
    xA = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.01],
            [0.01, 0.0],
            [0.02, 0.0],
        ],
        dtype=torch.float32,
    )
    xB = torch.tensor(
        [
            [0.0005, 0.0],   # Close to A0
            [0.0, 0.0105],   # Pickup before drop-off for A1
            [0.0105, 0.0],   # Violates speed constraint
            [0.05, 0.0],     # Beyond y_max
        ],
        dtype=torch.float32,
    )

    times_A = torch.tensor([0, 100, 200, 300], dtype=torch.int64)
    times_B = torch.tensor([60, 80, 210, 5000], dtype=torch.int64)

    speed_mps = 3.0
    y_max_meters = 1000.0

    result = match(
        xA,
        xB,
        kernel="haversine_speed",
        C=1.0,
        k=2,
        delta=0.01,
        times_A=times_A,
        times_B=times_B,
        speed_mps=speed_mps,
        y_max_meters=y_max_meters,
        future_only=True,
        fill_policy="none",
    )

    Mb = result.Mb
    matched_rows = torch.nonzero(Mb != -1, as_tuple=False).squeeze(1)
    matched_cols = Mb.index_select(0, matched_rows)

    for row, col in zip(matched_rows.tolist(), matched_cols.tolist()):
        dt = int(times_B[row].item() - times_A[col].item())
        assert dt >= 0, "Matched pair violates future-only constraint"
        dist = _haversine_distance(xB[row], xA[col])
        assert dist < y_max_meters, "Matched pair exceeds y_max meters"
        assert dt >= dist / speed_mps - 1e-6, "Matched pair violates speed constraint"

    feasible_matches = float(result.metrics.get("feasible_matches", 0.0))
    free_B = float(result.metrics.get("free_B", 0.0))
    assert pytest.approx(feasible_matches + free_B) == float(xB.shape[0])


def test_disabled_limits_behaves_reasonably() -> None:
    xA = torch.tensor(
        [
            [0.0, 0.0],
            [0.01, 0.0],
        ],
        dtype=torch.float32,
    )
    xB = torch.tensor(
        [
            [0.0005, 0.0],
            [0.0105, 0.0],
        ],
        dtype=torch.float32,
    )

    times_A = torch.tensor([0, 100], dtype=torch.int64)
    times_B = torch.tensor([200, 320], dtype=torch.int64)

    result = match(
        xA,
        xB,
        kernel="haversine_speed",
        C=1.0,
        k=2,
        delta=0.01,
        times_A=times_A,
        times_B=times_B,
        speed_mps=None,
        y_max_meters=None,
        future_only=True,
        fill_policy="none",
    )

    Mb = result.Mb
    matched_rows = torch.nonzero(Mb != -1, as_tuple=False).squeeze(1)
    matched_cols = Mb.index_select(0, matched_rows)

    distances = [
        _haversine_distance(xB[row], xA[col])
        for row, col in zip(matched_rows.tolist(), matched_cols.tolist())
    ]
    total_cost = sum(distances, 0.0)

    assert pytest.approx(float(result.matching_cost), rel=1e-5, abs=1e-5) == total_cost
    assert float(result.metrics.get("removed_by_speed", 0.0)) == 0.0
    assert float(result.metrics.get("removed_by_ymax", 0.0)) == 0.0
