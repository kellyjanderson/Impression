from __future__ import annotations

import warnings

_WARNED_MESSAGES: set[tuple[str, str]] = set()


def _warn_once(key: tuple[str, str], message: str, *, stacklevel: int) -> None:
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def warn_mesh_primary_backend(
    api_name: str,
    *,
    replacement: str = "backend='surface'",
    extra: str | None = None,
    stacklevel: int = 3,
) -> None:
    message = (
        f"{api_name} is still using the legacy mesh-primary backend path and is deprecated. "
        f"Prefer {replacement} and tessellate only at mesh-consumer boundaries."
    )
    if extra:
        message = f"{message} {extra}"
    _warn_once(("backend", api_name), message, stacklevel=stacklevel)


def warn_mesh_primary_api(
    api_name: str,
    *,
    replacement: str | None = None,
    extra: str | None = None,
    stacklevel: int = 3,
) -> None:
    message = f"{api_name} returns legacy mesh-primary geometry and is deprecated."
    if replacement:
        message = f"{message} Prefer {replacement}."
    if extra:
        message = f"{message} {extra}"
    _warn_once(("api", api_name), message, stacklevel=stacklevel)


def reset_legacy_mesh_deprecation_warnings() -> None:
    _WARNED_MESSAGES.clear()
