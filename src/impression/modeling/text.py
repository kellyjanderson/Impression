from __future__ import annotations

from typing import Any, Literal, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from impression.mesh import Mesh

from ..cad import shape_to_polydata
from ._color import set_mesh_color

if TYPE_CHECKING:  # pragma: no cover - typing only
    from build123d import FontStyle as FontStyle
else:
    FontStyle = Any

Justify = Literal["left", "center", "right"]
VARIATION_SELECTORS = {chr(code) for code in range(0xFE00, 0xFE0F + 1)}
VARIATION_SELECTORS.update(chr(code) for code in range(0xE0100, 0xE01EF + 1))


def make_text(
    content: str,
    depth: float = 0.2,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, -1.0, 0.0),
    font_size: float = 1.0,
    justify: Justify = "center",
    font: str = "Arial",
    font_path: str | None = None,
    font_style: FontStyle | str | None = None,
    tolerance: float = 0.05,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Create a text mesh by tessellating build123d geometry instead of PyVista primitives."""

    try:
        from build123d import BuildSketch, FontStyle, Text as BText, extrude
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Text support requires build123d. Install it or avoid make_text() during preview."
        ) from exc

    content = _strip_variation_selectors(content)
    if not content:
        raise ValueError("content must be a non-empty string after stripping variation selectors")

    depth_value = max(depth, 0.0)
    with BuildSketch() as sketch:
        BText(
            content,
            font_size=font_size,
            font=font,
            font_path=font_path,
            font_style=_coerce_font_style(font_style, FontStyle),
        )

    if depth_value > 0:
        shape = extrude(sketch.sketch, amount=depth_value)
    else:
        shape = sketch.sketch

    text = shape_to_polydata(shape, tolerance=tolerance)
    text.rotate_vector((0.0, 0.0, 1.0), 180.0, inplace=True)
    text.rotate_vector((1.0, 0.0, 0.0), 90.0, inplace=True)

    text = _justify_and_center(text, justify)
    text = _orient_mesh(text, direction, default=(0.0, 1.0, 0.0))
    text.translate(center, inplace=True)

    if color is not None:
        set_mesh_color(text, color)

    return text


def _justify_and_center(mesh: Mesh, justify: Justify) -> Mesh:
    bounds = mesh.bounds
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    if justify == "left":
        anchor_x = min_x
    elif justify == "right":
        anchor_x = max_x
    else:
        anchor_x = (min_x + max_x) / 2.0

    offset = (
        anchor_x,
        (min_y + max_y) / 2.0,
        (min_z + max_z) / 2.0,
    )
    mesh.translate((-offset[0], -offset[1], -offset[2]), inplace=True)
    return mesh


def _orient_mesh(
    mesh: Mesh,
    direction: Sequence[float],
    default: Sequence[float] = (0.0, 0.0, 1.0),
) -> Mesh:
    target_vec = np.array(_normalize(direction))
    default_vec = np.array(_normalize(default))

    if np.allclose(target_vec, default_vec):
        return mesh.copy()

    axis = np.cross(default_vec, target_vec)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        dot = np.dot(default_vec, target_vec)
        if dot > 0:
            return mesh.copy()
        axis = np.cross(default_vec, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(default_vec, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        angle_deg = 180.0
    else:
        axis = axis / axis_norm
        angle_rad = np.arccos(np.clip(np.dot(default_vec, target_vec), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

    rotated = mesh.copy()
    rotated.rotate_vector(axis, angle_deg, point=(0.0, 0.0, 0.0), inplace=True)
    return rotated


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def _coerce_font_style(value: object | str | None, font_style_cls: Any):
    if value is None:
        return font_style_cls.REGULAR
    if isinstance(value, font_style_cls):
        return value
    try:
        normalized = value.replace("-", "_").replace(" ", "_").upper()
        return font_style_cls[normalized]
    except KeyError as exc:  # pragma: no cover - defensive
        valid = ", ".join(style.name for style in font_style_cls)
        raise ValueError(f"Unknown font style '{value}'. Valid options: {valid}.") from exc


def _strip_variation_selectors(text: str) -> str:
    if not text:
        return text
    stripped = "".join(ch for ch in text if ch not in VARIATION_SELECTORS)
    return stripped or text
