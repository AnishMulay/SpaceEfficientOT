from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

import torch

from spef_ot import match


def _load_archive_solver() -> callable:
    archive_dir = Path(__file__).resolve().parents[1] / "Archive"
    module_path = archive_dir / "spef_matching_nyc_2.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Archive solver not found at {module_path}")

    if str(archive_dir) not in sys.path:
        sys.path.append(str(archive_dir))

    spec = importlib.util.spec_from_file_location("archive_spef_matching_nyc_2", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.spef_matching_2


class HaversineParityTest(unittest.TestCase):
    def test_haversine_kernel_parity(self) -> None:
        archive_solver = _load_archive_solver()

        torch.manual_seed(2024)
        n = 12
        xA = torch.rand(n, 2) * torch.tensor([1.0, 1.0]) + torch.tensor([-74.0, 40.0])
        xB = torch.rand(n, 2) * torch.tensor([1.0, 1.0]) + torch.tensor([-74.5, 40.5])

        times_A = torch.randint(0, 10, (n,), dtype=torch.int64)
        times_B = torch.randint(0, 10, (n,), dtype=torch.int64)

        C = 5.0
        delta = 1.0
        k = 4
        device = "cpu"
        seed = 7
        cmax_int = 5000

        Mb_old, yA_old, yB_old, cost_old, iter_old, metrics_old = archive_solver(
            xA.clone(),
            xB.clone(),
            C,
            k,
            delta,
            device=device,
            seed=seed,
            tA=times_A.clone(),
            tB=times_B.clone(),
            cmax_int=cmax_int,
        )

        result_new = match(
            xA.clone(),
            xB.clone(),
            kernel="haversine",
            C=C,
            k=k,
            delta=delta,
            device=device,
            seed=seed,
            times_A=times_A.clone(),
            times_B=times_B.clone(),
            cmax_int=cmax_int,
        )

        torch.testing.assert_close(result_new.Mb, Mb_old.cpu(), msg="Mb mismatch")
        torch.testing.assert_close(result_new.yA, yA_old.cpu(), msg="yA mismatch")
        torch.testing.assert_close(result_new.yB, yB_old.cpu(), msg="yB mismatch")
        torch.testing.assert_close(
            result_new.matching_cost,
            cost_old.cpu(),
            rtol=1e-6,
            atol=1e-6,
            msg="Cost mismatch",
        )
        self.assertEqual(result_new.iterations, iter_old)

        feasible_old = float(metrics_old.get("feasible_matches", len(Mb_old)))
        free_B_old = float(metrics_old.get("free_B", 0))
        self.assertAlmostEqual(result_new.metrics.get("feasible_matches"), feasible_old)
        self.assertAlmostEqual(result_new.metrics.get("free_B"), free_B_old)


if __name__ == "__main__":
    unittest.main()
