from __future__ import annotations

import warnings


def warn_min_feature(name: str, value: float, nozzle_diameter: float) -> None:
    if nozzle_diameter <= 0:
        return
    if value < nozzle_diameter:
        warnings.warn(
            f"{name} {value:.3f}mm is below nozzle diameter {nozzle_diameter:.3f}mm.",
            RuntimeWarning,
        )
