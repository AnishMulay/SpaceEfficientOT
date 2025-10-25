"""
Public interface for the space-efficient optimal transport library.

`match` is the main entry point for running the space-efficient matching solver.
Kernel registration helpers are re-exported from `spef_ot.kernels.registry`
so that users can discover and register custom slack kernels.
"""

from .core.solver import match
from .kernels.registry import available_kernels, get_kernel, register_kernel

__all__ = ["match", "available_kernels", "get_kernel", "register_kernel"]
