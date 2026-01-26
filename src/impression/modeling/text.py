from __future__ import annotations

from typing import Sequence

from impression.mesh import Mesh


# Text modeling is temporarily disabled.

def make_text(
    content: str,
    depth: float = 0.2,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, -1.0, 0.0),
    font_size: float = 1.0,
    justify: str = "center",
    font: str = "Arial",
    font_path: str | None = None,
    font_style: object | str | None = None,
    tolerance: float = 0.05,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    raise RuntimeError("make_text is disabled in this build.")


__all__ = ["make_text"]
