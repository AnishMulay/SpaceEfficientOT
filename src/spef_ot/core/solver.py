from __future__ import annotations

from typing import Any, Union

import torch

from .problem import Problem
from .state import SolverState
from ..kernels.base import SlackKernel
from ..kernels.registry import get_kernel


def match(
    xA: torch.Tensor,
    xB: torch.Tensor,
    *,
    kernel: Union[str, type[SlackKernel], SlackKernel] = "euclidean_sq",
    C: Union[float, torch.Tensor] = 1.0,
    k: int = 1,
    delta: Union[float, torch.Tensor] = 1.0,
    device: Union[str, torch.device, None] = None,
    seed: int = 1,
    **kernel_kwargs: Any,
) -> Any:
    """
    Solve the space-efficient matching problem between point sets `xA` and `xB`.

    This function currently provides only the public-facing API scaffold.
    The algorithmic implementation will be introduced in subsequent steps.
    """

    if device is None:
        device = xA.device
    torch_device = torch.device(device)

    kernel_instance = get_kernel(kernel, **kernel_kwargs)

    problem = Problem(
        xA=xA,
        xB=xB,
        C=C,
        delta=delta,
        device=torch_device,
    )
    state = SolverState()

    raise NotImplementedError("Solver core not implemented yet.")
