"""Fast CP inference regions and transports.

Model adapters load on demand. Importing transports does not load model kernels.
The package root is the supported consumer API.
"""

from .api import FastCPAttention, FastCPRuntime, init_fast_cp
from .runtime import (
    AsyncFastAgkvHandle,
    FastAgkvTransport,
    FastUlyssesAllToAll,
    FastUlyssesSubgroupTransport,
    FastUlyssesTransport,
    probe_fast_ulysses,
)


def __getattr__(name):
    if name in {"WanAdapter", "QwenImageAdapter", "ZImageAdapter"}:
        from . import adapters

        value = getattr(adapters, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AsyncFastAgkvHandle",
    "FastAgkvTransport",
    "FastUlyssesAllToAll",
    "FastUlyssesSubgroupTransport",
    "FastUlyssesTransport",
    "probe_fast_ulysses",
    "FastCPAttention",
    "FastCPRuntime",
    "init_fast_cp",
    "WanAdapter",
    "QwenImageAdapter",
    "ZImageAdapter",
]
