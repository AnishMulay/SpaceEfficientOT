from __future__ import annotations

import abc
from typing import Any, Optional, Protocol

import torch

from ..core.problem import Problem
from ..core.state import SolverState


class Workspace(Protocol):
    """Marker protocol for kernel-specific cached data."""


class SlackKernel(abc.ABC):
    """
    Abstract base class for slack kernels used by the solver.

    Concrete kernels must implement tile-level slack computation and provide
    a streamed pair-cost evaluation for the final matching cost aggregation.
    Kernel-specific data (e.g., masks, scaling parameters, cached tensors)
    should be prepared inside :meth:`prepare`.
    """

    def __init__(self, **config: Any) -> None:
        self._config = dict(config)

    @abc.abstractmethod
    def prepare(self, problem: Problem) -> Workspace:
        """
        Perform one-time preparation work before the solver loop starts.
        Returns a workspace object that will be passed to subsequent calls.
        """

    @abc.abstractmethod
    def compute_slack_tile(
        self,
        idxB: torch.Tensor,
        state: SolverState,
        workspace: Workspace,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the slack matrix for a tile of B-indices.

        Parameters
        ----------
        idxB:
            1D tensor of indices (length K) selecting rows from xB.
        state:
            Current solver state, giving access to dual variables and matches.
        workspace:
            Kernel-specific cached data returned by :meth:`prepare`.
        out:
            Optional preallocated tensor with shape [K, N] and dtype int64.

        Returns
        -------
        torch.Tensor
            Slack values with shape [K, N] in int64 dtype on the solver device.
        """

    @abc.abstractmethod
    def pair_cost(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        problem: Problem,
        workspace: Workspace,
    ) -> torch.Tensor:
        """
        Compute exact pairwise costs for matched edges without materialising
        the full cost matrix.
        """

    def finalize(
        self,
        problem: Problem,
        state: SolverState,
        workspace: Workspace,
    ) -> Optional[dict[str, Any]]:
        """
        Optional hook executed after the main solver loop and final matching
        fill. Kernels can override this to drop infeasible matches or record
        additional metrics. By default no action is taken.
        """
        return None
