"""Resolve public CP choices without changing the runtime transport API."""

from __future__ import annotations

CP_METHODS = ("agkv", "ulysses")
CP_BACKENDS = ("fast", "nccl")


def resolve_cp_options(
    *,
    method: str | None = None,
    backend: str | None = None,
    attention_mode: str | None = None,
    agkv_transport: str | None = None,
    ulysses_transport: str | None = None,
) -> tuple[str, str | None]:
    """Validate new/legacy options and return the method and backend.

    An omitted backend preserves the runtime's environment-based default.
    Legacy ``auto`` remains available for existing callers.
    """
    if method is not None and attention_mode is not None and method != attention_mode:
        raise ValueError("cp-method conflicts with legacy attention-mode")
    selected = method if method is not None else attention_mode
    selected = "agkv" if selected is None else selected
    if selected not in CP_METHODS:
        raise ValueError("cp-method must be one of: agkv, ulysses")
    if backend is not None and backend not in CP_BACKENDS:
        raise ValueError("cp-backend must be one of: fast, nccl")

    transports = {"agkv": agkv_transport, "ulysses": ulysses_transport}
    for name, transport in transports.items():
        if name != selected and transport is not None:
            raise ValueError(
                f"{name}-transport requires cp-method={name}; "
                "use cp-backend for the selected method"
            )
    legacy = transports[selected]
    if legacy is not None:
        aliases = {"torch": "nccl", f"fast_{selected}": "fast"}
        legacy = aliases.get(legacy, legacy)
        if legacy not in (*CP_BACKENDS, "auto"):
            raise ValueError(f"invalid {selected}-transport: {legacy!r}")
        if backend is not None and backend != legacy:
            raise ValueError(f"cp-backend conflicts with legacy {selected}-transport")
    return selected, backend if backend is not None else legacy
