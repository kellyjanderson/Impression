from __future__ import annotations

import numpy as np


class ValidationError(ValueError):
    """Raised when validation constraints are violated."""


def validate_periodic_profile(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValidationError("Profile points must be Nx2 points.")
    if np.any(~np.isfinite(arr)):
        raise ValidationError("Profile points contain invalid values.")
    phases = arr[:, 0]
    values = arr[:, 1]
    if np.any(phases < 0.0) or np.any(phases > 1.0):
        raise ValidationError("Profile phases must be in [0, 1].")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValidationError("Profile values must be in [0, 1].")

    order = np.argsort(phases)
    phases = phases[order]
    values = values[order]
    for i in range(1, len(phases)):
        if np.isclose(phases[i], phases[i - 1], atol=1e-6) and not np.isclose(values[i], values[i - 1], atol=1e-3):
            raise ValidationError("Profile has duplicate phases with different values.")

    has_zero = np.any(np.isclose(phases, 0.0, atol=1e-6))
    has_one = np.any(np.isclose(phases, 1.0, atol=1e-6))
    if has_zero and has_one:
        v0 = values[np.where(np.isclose(phases, 0.0, atol=1e-6))[0][0]]
        v1 = values[np.where(np.isclose(phases, 1.0, atol=1e-6))[0][0]]
        if not np.isclose(v0, v1, atol=1e-3):
            raise ValidationError("Profile must be continuous at phase 0/1.")

    if np.isclose(values.max(), values.min(), atol=1e-6):
        raise ValidationError("Profile must vary between root and crest.")
