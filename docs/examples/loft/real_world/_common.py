from __future__ import annotations

from typing import Iterable
import warnings

from impression.modeling import make_circle, make_rect, make_text, translate
from impression.modeling.drawing2d import PlanarShape2D

LABEL_COLOR = "#ff00ff"
DEFAULT_LABEL_DIRECTION = (0.6, 0.6, 0.5)


def make_ring_circle(outer_radius: float, inner_radius: float, color: str) -> PlanarShape2D:
    outer = make_circle(radius=outer_radius, color=color).outer
    inner = make_circle(radius=inner_radius).outer
    return PlanarShape2D(outer=outer, holes=[inner]).with_color(color)


def make_ring_rect(
    outer_size: tuple[float, float],
    wall: float,
    color: str,
) -> PlanarShape2D:
    outer = make_rect(size=outer_size, color=color).outer
    inner = make_rect(size=(outer_size[0] - 2.0 * wall, outer_size[1] - 2.0 * wall)).outer
    return PlanarShape2D(outer=outer, holes=[inner]).with_color(color)


def make_label(
    text: str,
    center: tuple[float, float, float],
    font_size: float = 5.2,
    depth: float = 0.8,
    direction: tuple[float, float, float] = DEFAULT_LABEL_DIRECTION,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        label = make_text(
            text,
            depth=depth,
            center=center,
            direction=direction,
            font_size=font_size,
            justify="center",
            valign="middle",
            color=LABEL_COLOR,
        )
    if hasattr(label, "face_colors"):
        label.face_colors = None
    if hasattr(label, "color"):
        label.color = (1.0, 0.0, 1.0, 1.0)
    return label


def translate_many(items: Iterable, offset: tuple[float, float, float]) -> list:
    translated = []
    for item in items:
        translate(item, offset)
        translated.append(item)
    return translated
