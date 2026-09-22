"""Explicit model contracts and prepared execution templates, loaded on demand."""

from importlib import import_module

__all__ = ["WanAdapter", "QwenImageAdapter", "ZImageAdapter"]

_MODULES = {
    "WanAdapter": ".wan",
    "QwenImageAdapter": ".qwen_image",
    "ZImageAdapter": ".zimage",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
