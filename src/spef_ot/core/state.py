from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SolverState:
    """
    Mutable solver state tracked across iterations of the matching algorithm.

    The fields will be populated when the solver implementation is introduced.
    """

    yA: torch.Tensor | None = None
    yB: torch.Tensor | None = None
    Ma: torch.Tensor | None = None
    Mb: torch.Tensor | None = None
    iteration: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    generator: torch.Generator | None = None
