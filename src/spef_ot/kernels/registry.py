from __future__ import annotations

from typing import Dict, Tuple, Type, Union

from .base import SlackKernel

_REGISTRY: Dict[str, Type[SlackKernel]] = {}


def register_kernel(name: str, kernel_cls: Type[SlackKernel]) -> None:
    """
    Register a kernel class under a human-readable name.
    """
    if not issubclass(kernel_cls, SlackKernel):
        raise TypeError("kernel_cls must inherit from SlackKernel")
    _REGISTRY[name] = kernel_cls


def available_kernels() -> Tuple[str, ...]:
    """
    Return the set of kernel names currently registered.
    """
    return tuple(sorted(_REGISTRY.keys()))


def get_kernel(
    kernel: Union[str, Type[SlackKernel], SlackKernel],
    **kwargs,
) -> SlackKernel:
    """
    Resolve a kernel specifier to an instantiated SlackKernel.
    """
    if isinstance(kernel, SlackKernel):
        if kwargs:
            raise ValueError("Cannot pass keyword arguments when supplying a kernel instance.")
        return kernel

    if isinstance(kernel, str):
        try:
            kernel_cls = _REGISTRY[kernel]
        except KeyError as exc:
            available = ", ".join(sorted(_REGISTRY)) or "<empty>"
            raise KeyError(f"Unknown kernel '{kernel}'. Registered kernels: {available}") from exc
    elif isinstance(kernel, type) and issubclass(kernel, SlackKernel):
        kernel_cls = kernel
    else:
        raise TypeError("kernel must be a string name, SlackKernel subclass, or SlackKernel instance.")

    return kernel_cls(**kwargs)
