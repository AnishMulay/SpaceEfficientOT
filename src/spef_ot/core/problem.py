from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class Problem:
    """
    Immutable container for solver inputs that are common across kernels.

    Kernel-specific data should be handled inside the kernel implementation.
    """

    xA: torch.Tensor
    xB: torch.Tensor
    C: Union[float, torch.Tensor]
    delta: Union[float, torch.Tensor]
    device: torch.device

    @property
    def n(self) -> int:
        """Number of points on the A side."""
        return int(self.xA.shape[0])

    @property
    def m(self) -> int:
        """Number of points on the B side."""
        return int(self.xB.shape[0])

    @property
    def C_value(self) -> float:
        """Scalar value of C as a Python float."""
        return float(torch.as_tensor(self.C, device=self.device, dtype=torch.float64).item())

    @property
    def delta_value(self) -> float:
        """Scalar value of delta as a Python float."""
        return float(torch.as_tensor(self.delta, device=self.device, dtype=torch.float64).item())
