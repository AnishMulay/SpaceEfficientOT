from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

import torch

from spef_ot import match


def _load_archive_solver() -> callable:
    archive_dir = Path(__file__).resolve().parents[1] / "Archive"
    module_path = archive_dir / "spef_matching_v2.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Archive solver not found at {module_path}")

    if str(archive_dir) not in sys.path:
        sys.path.append(str(archive_dir))

    spec = importlib.util.spec_from_file_location("archive_spef_matching_v2", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.spef_matching_2


class EuclideanParityTest(unittest.TestCase):
    def test_spef_matching_parity(self) -> None:
        archive_solver = _load_archive_solver()

        torch.manual_seed(1234)
        n, d = 16, 3
        xA = torch.rand(n, d)
        xB = torch.rand(n, d)
        C = 10.0
        delta = 1.0
        k = 4
        device = "cpu"
        seed = 42

        Mb_old, yA_old, yB_old, cost_old, iter_old = archive_solver(
            xA.clone(),
            xB.clone(),
            C,
            k,
            delta,
            device=device,
            seed=seed,
        )

        result_new = match(
            xA.clone(),
            xB.clone(),
            kernel="euclidean_sq",
            C=C,
            k=k,
            delta=delta,
            device=device,
            seed=seed,
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


if __name__ == "__main__":
    unittest.main()
