"""
Public interface for the space-efficient optimal transport library.

`match` is the main entry point for running the space-efficient matching solver.
`scaling_match` provides the epsilon-scaling (cost scaling) implementation.
Kernel registration helpers are re-exported from `spef_ot.kernels.registry`
so that users can discover and register custom slack kernels.
"""

from .solver import MatchResult, match
from .solver_scaling import ScalingMatchResult, scaling_match
from .kernels.registry import available_kernels, get_kernel, register_kernel

__all__ = [
    "match", 
    "MatchResult", 
    "scaling_match", 
    "ScalingMatchResult",
    "available_kernels", 
    "get_kernel", 
    "register_kernel"
]
