from .base import SlackKernel
from .registry import available_kernels, get_kernel, register_kernel
from . import euclidean_sq  # noqa: F401 - ensures default kernel registration

__all__ = ["SlackKernel", "available_kernels", "get_kernel", "register_kernel"]
